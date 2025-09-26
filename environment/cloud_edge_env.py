# environment/cloud_edge_env.py
"""
云边端三层架构卸载环境 - 简化版设备模型
- UE: CPU频率 + 任务负载
- ES: CPU频率 + 任务负载
- CS: CPU频率（资源无限）
- 考虑差异化通信延迟：边缘通信快，云端通信慢
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from .device_models import UserEquipment, EdgeServer, CloudServer
from .task_generator import TaskGenerator, Task


class TaskExecution:
    """任务执行状态跟踪"""
    def __init__(self, task_id, device_id, task_workload, data_size, start_time, execution_time, node_type, node_id, original_task_deadline):
        self.task_id = task_id
        self.device_id = device_id  # 发起任务的设备ID
        self.task_workload = task_workload    # CPU周期数
        self.data_size = data_size  # 数据大小
        self.start_time = start_time
        self.execution_time = execution_time
        self.remaining_time = execution_time
        self.node_type = node_type  # 'local', 'edge', 'cloud'
        self.node_id = node_id      # 节点ID
        self.completed = False
        self.original_task_deadline = original_task_deadline  # 原始任务的截止时间
        self.creation_step = 0  # 任务创建的step
        
    def is_deadline_violated(self, current_time):
        """检查是否违反截止时间"""
        expected_completion_time = self.start_time + self.execution_time
        return expected_completion_time > self.original_task_deadline
    
    def get_progress(self):
        """获取执行进度 (0-1)"""
        if self.execution_time == 0:
            return 1.0
        return max(0, (self.execution_time - self.remaining_time) / self.execution_time)


class CloudEdgeDeviceEnv(gym.Env):
    """云边端三层架构卸载环境 - 简化设备模型版本"""
    
    def __init__(self, config):
        super(CloudEdgeDeviceEnv, self).__init__()

        self.config = config
        
        # 基础配置
        self.num_devices = config.get('environment', {}).get('num_devices', 10)
        self.num_edges = config.get('environment', {}).get('num_edges', 5)
        self.num_clouds = config.get('environment', {}).get('num_clouds', 1)
        
        # 🚀 真实边缘环境任务生成配置
        self.task_generation_config = {
            # 基础泊松参数
            'base_arrival_rate': 0.8,      # 基础任务到达率（每设备每step）
            'poisson_lambda': 1.2,         # 泊松分布参数
            
            # 时间模式配置
            'time_pattern_enabled': True,   # 启用时间模式
            'peak_hours': [20, 40, 60, 80], # 高峰时段（step）
            'peak_multiplier': 2.5,         # 高峰期倍率
            'low_multiplier': 0.4,          # 低峰期倍率
            'pattern_cycle': 20,            # 模式周期（step）
            
            # 突发任务配置
            'burst_probability': 0.05,      # 突发概率（每step 5%）
            'burst_intensity': 3.0,         # 突发强度倍率
            'burst_duration': [2, 5],       # 突发持续时间范围
            
            # 负载控制
            'max_concurrent_tasks': 200,    # 增加并发限制
            'device_load_threshold': 60.0,  # 设备负载阈值（秒）
            'system_load_threshold': 800.0, # 系统负载阈值（秒）
            'emergency_threshold': 1000.0,  # 紧急阈值
            
            # 应用场景混合
            'application_mix': {
                'iot_sensors': 0.4,         # IoT传感器数据
                'mobile_apps': 0.3,         # 移动应用
                'video_stream': 0.15,       # 视频流处理
                'ai_inference': 0.1,        # AI推理
                'emergency': 0.05           # 紧急任务
            }
        }
        
        # 🚀 动态截止时间配置
        self.deadline_config = {
            'adaptive_deadline': True,
            'base_factors': {
                'iot_sensors': (2.0, 4.0),     # IoT: 宽松截止时间
                'mobile_apps': (1.5, 3.0),     # 移动应用: 中等截止时间
                'video_stream': (1.2, 2.0),    # 视频: 严格截止时间
                'ai_inference': (1.8, 3.5),    # AI: 中等偏宽松
                'emergency': (1.1, 1.5)        # 紧急: 极严格截止时间
            },
            'load_adjustment': True,
            'min_deadline': 2.0,
            'congestion_penalty': 1.5
        }

        # 创建设备
        self._create_devices()

        # 任务生成器
        self.task_generator = TaskGenerator(config)
        
        # 🚀 真实场景任务类型重新定义
        self.task_generator.task_type_weights = {
            'small': 0.5,   # 小任务：IoT传感器、文本处理
            'medium': 0.35, # 中任务：图像处理、轻度AI
            'large': 0.15   # 大任务：视频分析、复杂AI
        }
        
        # 🚀 应用场景特定的任务参数
        self.application_task_configs = {
            'iot_sensors': {
                'size_range': (0.1, 2.0),      # 0.1-2MB
                'compute_density': 0.05e9,      # 低计算密度
                'priority': 'low'
            },
            'mobile_apps': {
                'size_range': (1.0, 20.0),     # 1-20MB
                'compute_density': 0.15e9,      # 中计算密度
                'priority': 'medium'
            },
            'video_stream': {
                'size_range': (50.0, 150.0),   # 50-150MB
                'compute_density': 0.3e9,       # 高计算密度
                'priority': 'high'
            },
            'ai_inference': {
                'size_range': (10.0, 80.0),    # 10-80MB
                'compute_density': 0.25e9,      # 高计算密度
                'priority': 'medium'
            },
            'emergency': {
                'size_range': (0.5, 30.0),     # 0.5-30MB
                'compute_density': 0.1e9,       # 可变计算密度
                'priority': 'critical'
            }
        }
        
        # 🚀 真实环境任务生成状态
        self.task_generation_state = {
            'last_generation_step': -1,     # 上次生成步数
            'total_concurrent_tasks': 0,    # 当前并发任务数
            'burst_active': False,          # 是否在突发期
            'burst_end_step': 0,           # 突发结束步数
            'current_pattern_phase': 'normal', # 当前模式：normal/peak/low
            'generation_history': [],       # 生成历史
            'daily_task_count': 0          # 每日任务计数
        }

        # 状态空间维度计算
        # UE: 3个特征 × num_devices
        # ES: 2个特征 × num_edges  
        # CS: 1个特征 × num_clouds
        # 任务: 6个特征 × num_devices
        self.state_dim = (3 * self.num_devices + 
                         2 * self.num_edges + 
                         1 * self.num_clouds + 
                         6 * self.num_devices)

        # 定义观察和动作空间
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

        # 动作空间定义：每个设备的三元分割决策 [α1, α2, α3, edge_id]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0]),      # [α1, α2, α3, edge_id]
            high=np.array([1.0, 1.0, 1.0, self.num_edges - 1]),
            dtype=np.float32
        )

        # 任务执行跟踪
        self.current_tasks = None
        self.task_executions = defaultdict(list)  # 按节点分组的执行队列
        self.completed_tasks_history = []  # 已完成任务的历史记录
        self.global_time = 0.0  # 全局时间步
        self.time_step_duration = 1.0  # 每个step的持续时间（秒）
        
        # Episode控制
        self.episode_step = 0
        
        # 从配置中读取max_steps，而不是硬编码为100
        # 优先从maddpg配置读取，如果不存在则从training配置读取，如果都不存在则默认为200
        self.max_steps = config.get('maddpg', {}).get('max_steps', 
                          config.get('training', {}).get('max_steps_per_episode', 200))
        print(f"环境初始化: 最大步数设置为 {self.max_steps}")
        
        # 🆕 任务生成控制
        self.last_generation_step = 0  # 上次生成任务的步数
        self.total_concurrent_tasks = 0  # 当前并发任务数
        
        # 统计信息
        self.step_stats = {
            'tasks_completed': 0,
            'tasks_timeout': 0,
            'total_latency': 0.0,
            'total_energy': 0.0,
            'communication_latency': 0.0,
            'computation_latency': 0.0
        }
        
        # 新增：任务完成率统计
        self.task_completion_stats = {
            'total_tasks_generated': 0,        # 总生成任务数
            'tasks_completed_on_time': 0,      # 按时完成的任务数
            'tasks_completed_late': 0,         # 超时完成的任务数
            'tasks_failed': 0,                 # 失败任务数
            'completion_times': [],            # 任务完成时间记录
            'deadline_violations': [],         # 截止时间违反记录
            'timeout_reasons': []              # 超时原因记录
        }

    def _create_devices(self):
        """创建云边端三层设备"""
        # 创建端侧设备（异构CPU频率：0.5-1.0 GHz）
        self.user_equipments = []
        for i in range(self.num_devices):
            ue = UserEquipment(i)
            self.user_equipments.append(ue)
            
        # 创建边缘服务器（异构配置：{5, 6, 7, 8, 9} GHz）
        edge_frequencies = [5, 6, 7, 8, 9]
        self.edge_servers = []
        for i in range(self.num_edges):
            es = EdgeServer(i, edge_frequencies[i % len(edge_frequencies)])
            self.edge_servers.append(es)
            
        # 创建云服务器（20 GHz）
        self.cloud_servers = []
        for i in range(self.num_clouds):
            cs = CloudServer(i)
            self.cloud_servers.append(cs)

    @property
    def devices(self):
        """返回所有用户设备（用于LLM咨询）"""
        return self.user_equipments

    @property 
    def edge_servers_list(self):
        """返回边缘服务器列表"""
        return self.edge_servers
        
    @property
    def cloud_servers_list(self):
        """返回云服务器列表"""
        return self.cloud_servers

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
            
        # 重置所有设备状态
        for ue in self.user_equipments:
            ue.reset()
        for es in self.edge_servers:
            es.reset()
        for cs in self.cloud_servers:
            cs.reset()

        # 重置任务执行跟踪
        self.task_executions.clear()
        self.completed_tasks_history.clear()
        self.global_time = 0.0
        self.episode_step = 0
        
        # 🚀 重置真实环境任务生成状态
        self.task_generation_state = {
            'last_generation_step': -1,     # 上次生成步数
            'total_concurrent_tasks': 0,    # 当前并发任务数
            'burst_active': False,          # 是否在突发期
            'burst_end_step': 0,           # 突发结束步数
            'current_pattern_phase': 'normal', # 当前模式：normal/peak/low
            'generation_history': [],       # 生成历史
            'daily_task_count': 0          # 每日任务计数
        }
        
        # 重置统计信息
        self.step_stats = {
            'tasks_completed': 0,
            'tasks_timeout': 0,
            'total_latency': 0.0,
            'total_energy': 0.0,
            'communication_latency': 0.0,
            'computation_latency': 0.0
        }
        
        # 新增：任务完成率统计
        self.task_completion_stats = {
            'total_tasks_generated': 0,        # 总生成任务数
            'tasks_completed_on_time': 0,      # 按时完成的任务数
            'tasks_completed_late': 0,         # 超时完成的任务数
            'tasks_failed': 0,                 # 失败任务数
            'completion_times': [],            # 任务完成时间记录
            'deadline_violations': [],         # 截止时间违反记录
            'timeout_reasons': []              # 超时原因记录
        }
        
        # 生成第一批任务
        self._generate_new_tasks()

        return self._get_observation(), {}

    def _generate_new_tasks(self):
        """🚀 使用基础任务生成器生成任务 - 泊松分布 + 时间模式"""
        print(f"\n[Step {self.episode_step}] 🌟 生成新任务...")
        
        # 使用基础任务生成器的泊松生成逻辑
        if not hasattr(self, 'task_generator'):
            # 初始化任务生成器
            from environment.task_generator import TaskGenerator
            
            # 从配置中提取任务生成器相关配置
            task_config = {}
            if 'tasks' in self.config:
                task_config = self.config['tasks']
            
            # 配置泊松参数
            poisson_config = {
                'base_arrival_rate': self.task_generation_config.get('base_arrival_rate', 0.5),
                'time_pattern_enabled': self.task_generation_config.get('time_pattern_enabled', True),
                'pattern_cycle': self.task_generation_config.get('pattern_cycle', 20),
                'peak_hours': self.task_generation_config.get('peak_hours', [5, 15]),
                'peak_multiplier': self.task_generation_config.get('peak_multiplier', 2.0),
                'low_multiplier': self.task_generation_config.get('low_multiplier', 0.5),
            }
            
            task_config['poisson_config'] = poisson_config
            self.task_generator = TaskGenerator(task_config)
        
        # 使用基础任务生成器生成任务
        device_tasks_dict = self.task_generator.generate_poisson_tasks(
            num_devices=self.num_devices,
            step=self.episode_step
        )
        
        # 创建任务对象
        self.current_tasks = []
        for device_id in range(self.num_devices):
            if device_id in device_tasks_dict and device_tasks_dict[device_id]:
                # 取该设备的第一个任务（如果有多个，只保留第一个）
                task_data = device_tasks_dict[device_id][0]
                task = Task(task_data)
                task.creation_step = self.episode_step
                self.current_tasks.append(task)
                
                # 更新统计
                self.task_completion_stats['total_tasks_generated'] += 1
                self.task_generation_state['total_concurrent_tasks'] += 1
                self.task_generation_state['daily_task_count'] += 1
                
                # 如果该设备有多个任务，打印提示（仅测试用）
                if len(device_tasks_dict[device_id]) > 1:
                    print(f"   设备 {device_id} 有 {len(device_tasks_dict[device_id])} 个任务，只保留第一个")
            else:
                self.current_tasks.append(None)
        
        # 打印生成结果
        valid_tasks = sum(1 for task in self.current_tasks if task is not None)
        print(f"   📊 生成结果: {valid_tasks}/{self.num_devices}个设备有任务")
        print(f"   💼 当前并发任务: {self.task_generation_state['total_concurrent_tasks']}")
        print(f"   📈 累计生成任务: {self.task_completion_stats['total_tasks_generated']}")
        
        # 更新生成历史（保持与原代码的兼容性）
        self.task_generation_state['generation_history'].append({
            'step': self.episode_step,
            'total_generated': valid_tasks,
            'pattern_phase': self.task_generator.current_pattern_phase,
            'burst_active': self.task_generation_state.get('burst_active', False),
            'arrival_rate': self.task_generator.poisson_config['base_arrival_rate']
        })

    def _calculate_time_pattern_multiplier(self):
        """计算时间模式倍率"""
        if not self.task_generation_config['time_pattern_enabled']:
            return 1.0
            
        # 计算在周期中的位置
        cycle_position = self.episode_step % self.task_generation_config['pattern_cycle']
        
        # 判断当前是否在高峰期
        is_peak = any(abs(self.episode_step - peak) <= 2 for peak in self.task_generation_config['peak_hours'])
        
        if is_peak:
            self.task_generation_state['current_pattern_phase'] = 'peak'
            return self.task_generation_config['peak_multiplier']
        elif cycle_position < 5:  # 周期前25%为低峰期
            self.task_generation_state['current_pattern_phase'] = 'low'
            return self.task_generation_config['low_multiplier']
        else:
            self.task_generation_state['current_pattern_phase'] = 'normal'
            return 1.0
    
    def _handle_burst_events(self):
        """处理突发事件"""
        # 检查当前突发是否结束
        if self.task_generation_state['burst_active']:
            if self.episode_step >= self.task_generation_state['burst_end_step']:
                self.task_generation_state['burst_active'] = False
                print(f"   🔥 突发事件结束 (step {self.episode_step})")
                return 1.0
            else:
                remaining = self.task_generation_state['burst_end_step'] - self.episode_step
                print(f"   🔥 突发事件进行中 (剩余{remaining}步)")
                return self.task_generation_config['burst_intensity']
        
        # 检查是否触发新的突发事件
        if np.random.random() < self.task_generation_config['burst_probability']:
            duration = np.random.randint(*self.task_generation_config['burst_duration'])
            self.task_generation_state['burst_active'] = True
            self.task_generation_state['burst_end_step'] = self.episode_step + duration
            
            print(f"   🔥 新突发事件触发！持续{duration}步")
            return self.task_generation_config['burst_intensity']
        
        return 1.0
    
    def _generate_poisson_tasks(self, device_id, arrival_rate):
        """为单个设备使用泊松分布生成任务"""
        # 使用泊松分布确定任务数量
        num_tasks = np.random.poisson(arrival_rate)
        
        # 考虑设备负载限制
        device_load = self.user_equipments[device_id].calculate_task_load()
        if device_load > self.task_generation_config['device_load_threshold']:
            # 设备过载，减少任务生成
            reduction_factor = min(device_load / self.task_generation_config['device_load_threshold'], 3.0)
            num_tasks = max(0, int(num_tasks / reduction_factor))
        
        # 检查系统总负载
        system_load = self._calculate_system_load()
        if system_load > self.task_generation_config['system_load_threshold']:
            num_tasks = 0  # 系统过载，停止生成
        
        if num_tasks == 0:
            return []
        
        # 生成具体任务
        tasks = []
        for i in range(num_tasks):
            task_data = self._generate_realistic_task(device_id)
            tasks.append(task_data)
        
        return tasks
    
    def _generate_realistic_task(self, device_id):
        """生成符合真实场景的任务"""
        # 1. 选择应用类型
        app_types = list(self.task_generation_config['application_mix'].keys())
        app_probs = list(self.task_generation_config['application_mix'].values())
        app_type = np.random.choice(app_types, p=app_probs)
        
        # 2. 获取应用配置
        app_config = self.application_task_configs[app_type]
        
        # 3. 生成任务大小
        min_size, max_size = app_config['size_range']
        data_size = np.random.uniform(min_size, max_size)
        
        # 4. 计算CPU周期需求
        cpu_cycles = data_size * app_config['compute_density']
        
        # 5. 设置截止时间
        deadline = self._calculate_realistic_deadline(app_type, data_size, cpu_cycles, device_id)
        
        # 6. 生成全局唯一任务ID
        task_id = f"{device_id}_{self.episode_step}_{self.task_completion_stats['total_tasks_generated']}"
        
        return {
            'task_id': task_id,
            'device_id': device_id,
            'type': app_type,
            'data_size': data_size,
            'cpu_cycles': cpu_cycles,
            'deadline': deadline,
            'priority': app_config['priority'],
            'arrival_time': self.global_time,
            'application_type': app_type
        }
    
    def _calculate_realistic_deadline(self, app_type, data_size, cpu_cycles, device_id):
        """计算符合真实场景的截止时间"""
        # 1. 获取基础截止时间因子
        base_factors = self.deadline_config['base_factors'][app_type]
        base_factor = np.random.uniform(*base_factors)
        
        # 2. 计算本地执行时间（使用该设备的CPU频率）
        device_cpu_freq = self.user_equipments[device_id].cpu_frequency * 1e9  # Hz
        local_execution_time = cpu_cycles / device_cpu_freq
        
        # 3. 基础截止时间
        base_deadline = local_execution_time * base_factor
        
        # 4. 负载调整
        if self.deadline_config['load_adjustment']:
            system_load = self._calculate_system_load()
            if system_load > self.task_generation_config['system_load_threshold'] * 0.7:
                # 系统负载较高，适当放宽截止时间
                congestion_factor = self.deadline_config['congestion_penalty']
                base_deadline *= congestion_factor
        
        # 5. 确保最小截止时间
        final_deadline = max(base_deadline, self.deadline_config['min_deadline'])
        
        return final_deadline

    def _calculate_system_load(self):
        """计算系统整体负载"""
        total_load = 0.0
        
        # UE负载
        for ue in self.user_equipments:
            total_load += ue.calculate_task_load()
            
        # ES负载
        for es in self.edge_servers:
            total_load += es.calculate_task_load()
            
        return total_load

    def step(self, actions, llm_actions=None):
        """
        执行一步动作 - 考虑差异化通信延迟
        
        Args:
            actions: Agent的动作 shape=(num_devices, 4) [α1, α2, α3, edge_id] 或 list
            llm_actions: LLM专家动作 shape=(num_devices, 4) 或 list
        
        Returns:
            observation, rewards, terminated, truncated, info
        """
        print(f"\n{'='*80}")
        print(f"开始执行 Step {self.episode_step + 1}")
        print(f"{'='*80}")
        
        self.episode_step += 1
        
        # 1. 推进全局时间，更新所有设备的任务状态
        self.global_time += self.time_step_duration
        print(f"[Step {self.episode_step}] 时间推进到: {self.global_time:.1f}s")
        
        # 更新所有设备的任务执行状态
        self._update_all_devices(self.time_step_duration)
        
        # 🔧 确保actions是NumPy数组格式
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        # 2. 显示MADDPG动作解析过程
        print(f"\n🔄 MADDPG动作环境交互过程:")
        print(f"{'='*80}")
        print(f"接收到的MADDPG动作维度: {actions.shape}")
        print(f"动作内容:")
        for i, action in enumerate(actions):
            alpha1, alpha2, alpha3, edge_id_raw = action
            edge_id = int(np.clip(edge_id_raw, 0, self.num_edges - 1))
            
            # 归一化分割比例
            total = alpha1 + alpha2 + alpha3
            if total > 0:
                alpha1_norm, alpha2_norm, alpha3_norm = alpha1/total, alpha2/total, alpha3/total
            else:
                alpha1_norm, alpha2_norm, alpha3_norm = 1.0, 0.0, 0.0
            
            print(f"  Device{i}: 原始[{alpha1:.3f}, {alpha2:.3f}, {alpha3:.3f}, {edge_id_raw:.3f}]")
            print(f"           → 解析为[本地:{alpha1_norm:.3f}, 边缘:{alpha2_norm:.3f}, 云端:{alpha3_norm:.3f}, Edge{edge_id}]")
        
        # 3. 处理新任务的卸载决策
        print(f"\n[Step {self.episode_step}] 🚀 执行MADDPG卸载决策...")
        rewards = np.zeros(self.num_devices)
        total_latencies = []
        total_energies = []
        communication_latencies = []
        computation_latencies = []
        has_task_list = []  # 新增：记录每个设备是否有任务

        for i in range(self.num_devices):
            # 🔧 安全地获取单个设备的动作
            if len(actions.shape) > 1:
                action = actions[i]
            else:
                action = actions
            
            reward, metrics = self._execute_offloading_decision(i, action)
            rewards[i] = reward
            
            # 记录该设备是否有任务
            has_task = reward > 0.0  # 如果奖励为0，说明没有任务
            has_task_list.append(has_task)
            
            total_latencies.append(metrics['total_latency'])  # 总时延
            total_energies.append(metrics['total_energy'])  # 总能耗
            communication_latencies.append(metrics['communication_latency'])  # 通信时延
            computation_latencies.append(metrics['computation_latency'])  # 计算时延
            
            # 更新统计信息
            self.step_stats['total_latency'] += metrics['total_latency']
            self.step_stats['total_energy'] += metrics['total_energy']
            self.step_stats['communication_latency'] += metrics['communication_latency']
            self.step_stats['computation_latency'] += metrics['computation_latency']

        # 显示奖励反馈
        print(f"\n💰 MADDPG动作奖励反馈:")
        print(f"{'='*80}")
        for i, reward in enumerate(rewards):
            print(f"  Device{i}: 奖励值 = {reward:.3f}")
        
        # 计算有任务设备的平均奖励
        valid_rewards = [r for r, has_task in zip(rewards, has_task_list) if has_task]
        if valid_rewards:
            avg_reward = np.mean(valid_rewards)
            min_reward = np.min(valid_rewards)
            max_reward = np.max(valid_rewards)
        else:
            avg_reward = 0.0
            min_reward = 0.0
            max_reward = 0.0
            
        print(f"  平均奖励: {avg_reward:.3f} (仅计算有任务的设备)")
        print(f"  奖励范围: [{min_reward:.3f}, {max_reward:.3f}]")

        # 4. 检查终止条件
        max_steps_reached = self.episode_step >= self.max_steps
        terminated = False
        truncated = max_steps_reached

        # 5. 如果还没结束，为下一步生成新任务
        if not (terminated or truncated):
            self._generate_new_tasks()

        # 6. 打印当前状态总结
        self._print_step_summary()

        # 构建info字典
        info = {
            'total_latencies': total_latencies,
            'total_energies': total_energies,
            'communication_latencies': communication_latencies,
            'computation_latencies': computation_latencies,
            'episode_step': self.episode_step,
            'global_time': self.global_time,
            'step_stats': self.step_stats.copy(),
            'llm_actions': llm_actions if llm_actions is not None else [],
            # 新增：任务完成率统计
            'task_completion_stats': self.get_task_completion_rate(),
            'deadline_violations': self.task_completion_stats['deadline_violations'].copy(),
            'timeout_reasons': self.task_completion_stats['timeout_reasons'].copy(),
            # 新增：MADDPG动作信息
            'maddpg_actions': actions.tolist(),
            'maddpg_rewards': rewards.tolist(),
            # 新增：每个设备是否有任务的标志
            'has_task_list': has_task_list
        }

        return self._get_observation(), rewards, terminated, truncated, info

    def _update_all_devices(self, time_elapsed):
        """更新所有设备的任务执行状态"""
        # 更新端侧设备
        for ue in self.user_equipments:
            ue.update_tasks(time_elapsed)
            
        # 更新边缘服务器
        for es in self.edge_servers:
            es.update_tasks(time_elapsed)
        
        # 云服务器无需更新（资源无限，任务立即执行）

    def _execute_offloading_decision(self, device_idx, action):
        """执行单个设备的卸载决策 - 考虑差异化通信延迟"""
        # 🆕 检查是否有任务需要处理
        if self.current_tasks is None or device_idx >= len(self.current_tasks):
            # 没有任务，返回零奖励
            return 0.0, {
                'total_latency': 0.0,
                'total_energy': 0.0, 
                'communication_latency': 0.0,
                'computation_latency': 0.0,
                'local_baseline': (0.0, 0.0)
            }
        
        task = self.current_tasks[device_idx]
        if task is None:
            # 该设备没有任务，返回零奖励
            print(f"  Device{device_idx}: 无任务分配")
            return 0.0, {
                'total_latency': 0.0,
                'total_energy': 0.0,
                'communication_latency': 0.0, 
                'computation_latency': 0.0,
                'local_baseline': (0.0, 0.0)
            }
        
        # 解析动作
        alpha1, alpha2, alpha3, edge_id = action
        edge_id = int(np.clip(edge_id, 0, self.num_edges - 1))
        
        # 归一化分割比例，确保和为1
        total = alpha1 + alpha2 + alpha3
        if total > 0:
            alpha1, alpha2, alpha3 = alpha1/total, alpha2/total, alpha3/total
        else:
            alpha1, alpha2, alpha3 = 1.0, 0.0, 0.0  # 默认全本地
            
        # 获取设备和任务
        ue = self.user_equipments[device_idx]
        task.set_split_ratios(alpha1, alpha2, alpha3)
        
        print(f"  Device{device_idx}: Task{task.task_id} 分割比例 "
              f"[本地:{alpha1:.2f}, 边缘:{alpha2:.2f}, 云端:{alpha3:.2f}] → Edge{edge_id}")
        
        # 分割任务并分配到不同节点
        total_latency, total_energy, comm_latency, comp_latency = self._schedule_task_execution_optimized(
            ue, task, edge_id, device_idx)
        
        # 计算本地基准
        baseline_latency, baseline_energy = self._calculate_local_baseline(ue, task)
        
        # 计算奖励函数
        reward = self._calculate_reward(
            total_latency, total_energy, baseline_latency, baseline_energy, task.deadline, edge_id)
        
        metrics = {
            'total_latency': total_latency,
            'total_energy': total_energy,
            'communication_latency': comm_latency,
            'computation_latency': comp_latency,
            'local_baseline': (baseline_latency, baseline_energy)
        }
        
        return reward, metrics

    def _schedule_task_execution_optimized(self, ue, task, edge_id, device_idx):
        """
        优化的任务调度 - 考虑差异化通信延迟和任务负载
        
        返回: (总延迟, 总能耗, 通信延迟, 计算延迟)
        """
        workloads = task.get_split_workloads()  # [本地, 边缘, 云端]工作负载
        data_sizes = task.get_split_data_sizes()  # [本地, 边缘, 云端]数据大小
        
        latencies = []
        energies = []
        comm_latencies = []
        comp_latencies = []
        
        # 1. 本地计算部分
        if workloads[0] > 0:
            # 获取当前任务负载（等待时间）
            current_load = ue.calculate_task_load()
            exec_time = ue.calculate_execution_time(workloads[0])
            energy = ue.calculate_energy_consumption(workloads[0])
            
            # 添加任务到设备队列
            ue.add_task(f"{task.task_id}_local", workloads[0], self.global_time)
            
            total_time = current_load + exec_time
            latencies.append(total_time)
            energies.append(energy)
            comm_latencies.append(0.0)  # 本地无通信延迟
            comp_latencies.append(exec_time)
            

            
            print(f"    本地执行: {workloads[0]/1e9:.2f}Gcycles, "
                  f"等待{current_load:.2f}s + 计算{exec_time:.2f}s")
        else:
            latencies.append(0.0)
            energies.append(0.0)
            comm_latencies.append(0.0)
            comp_latencies.append(0.0)
            
        # 2. 边缘计算部分
        if workloads[1] > 0:
            es = self.edge_servers[edge_id]
            
            # 通信延迟（UE到边缘）
            comm_time = ue.calculate_transmission_time_to_edge(data_sizes[1])
            comm_energy = ue.calculate_transmission_energy(comm_time)
            
            # 边缘服务器的任务负载（等待时间）
            edge_load = es.calculate_task_load()
            exec_time = es.calculate_execution_time(workloads[1])
            
            # 添加任务到边缘队列
            es.add_task(f"{task.task_id}_edge", workloads[1], self.global_time + comm_time)
            
            total_time = comm_time + edge_load + exec_time
            latencies.append(total_time)
            energies.append(comm_energy)  # 只计算UE的传输能耗
            comm_latencies.append(comm_time)
            comp_latencies.append(exec_time)
            

            
            print(f"    边缘执行: {workloads[1]/1e9:.2f}Gcycles → ES{edge_id}, "
                  f"通信{comm_time:.2f}s + 等待{edge_load:.2f}s + 计算{exec_time:.2f}s")
        else:
            latencies.append(0.0)
            energies.append(0.0)
            comm_latencies.append(0.0)
            comp_latencies.append(0.0)
            
        # 3. 云计算部分（差异化通信延迟）
        if workloads[2] > 0:
            cs = self.cloud_servers[0]
            
            # 通信延迟（UE→边缘→云，总延迟更高）
            comm_time = ue.calculate_transmission_time_to_cloud(data_sizes[2])
            comm_energy = ue.calculate_transmission_energy(comm_time * 0.6)  # 部分传输时间的能耗
            
            # 云计算时间（无等待，资源无限）
            exec_time = cs.calculate_execution_time(workloads[2])
            
            total_time = comm_time + exec_time
            latencies.append(total_time)
            energies.append(comm_energy)
            comm_latencies.append(comm_time)
            comp_latencies.append(exec_time)
            

            
            print(f"    云端执行: {workloads[2]/1e9:.2f}Gcycles → Cloud, "
                  f"通信{comm_time:.2f}s + 计算{exec_time:.2f}s")
        else:
            latencies.append(0.0)
            energies.append(0.0)
            comm_latencies.append(0.0)
            comp_latencies.append(0.0)
            
        # 计算总延迟（取最大值，因为可以并行执行）和总能耗（求和）
        total_latency = max(latencies)
        total_energy = sum(energies)
        total_comm_latency = max(comm_latencies)
        total_comp_latency = max(comp_latencies)
        
        # 检查任务完成状态
        self._check_task_completion(task, total_latency)
        
        return total_latency, total_energy, total_comm_latency, total_comp_latency

    def _calculate_local_baseline(self, ue, task):
        """计算全本地执行的基准时延和能耗"""
        current_load = ue.calculate_task_load()
        exec_time = ue.calculate_execution_time(task.task_workload)
        energy = ue.calculate_energy_consumption(task.task_workload)
        return current_load + exec_time, energy

    def _calculate_reward(self, offload_latency, offload_energy, 
                         baseline_latency, baseline_energy, deadline, edge_id=None):
        """
        计算奖励函数，包含负载均衡奖励项
        
        Args:
            offload_latency: 卸载延迟
            offload_energy: 卸载能耗
            baseline_latency: 基准延迟（全本地执行）
            baseline_energy: 基准能耗（全本地执行）
            deadline: 任务截止时间
            edge_id: 选择的边缘服务器ID（如果有）
        
        Returns:
            float: 奖励值
        """
        # 基本奖励：延迟和能耗的倒数加权
        # 防止除零
        latency_term = 1.0 / offload_latency if offload_latency > 1e-8 else 0.0
        energy_term = 1.0 / offload_energy if offload_energy > 1e-8 else 0.0
        basic_reward = 0.5 * latency_term + 0.5 * energy_term
        
        # 负载均衡奖励项
        load_balancing_reward = 0.0
        if edge_id is not None:
            # 计算所有边缘服务器的负载差异
            loads = [es.calculate_task_load() for es in self.edge_servers]
            
            if sum(loads) > 0:  # 确保有负载
                # 计算负载标准差，标准差越小表示越均衡
                mean_load = sum(loads) / len(loads)
                load_variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
                load_std = load_variance ** 0.5
                
                # 负载标准差越小，奖励越大
                # 使用指数衰减函数，当标准差为0时奖励最大为0.2
                load_balancing_reward = 0.2 * np.exp(-2.0 * load_std)
                
                # 额外奖励：选择负载最低的服务器
                min_load_idx = np.argmin(loads)
                if edge_id == min_load_idx:
                    load_balancing_reward += 0.1
                    
                # 调试输出
                print(f"    负载均衡: 各服务器负载={[f'{load:.1f}' for load in loads]}, "
                      f"标准差={load_std:.2f}, 奖励={load_balancing_reward:.2f}")
        
        # 总奖励 = 基本奖励 + 负载均衡奖励
        total_reward = basic_reward + load_balancing_reward
        
        return float(total_reward)

    def _print_step_summary(self):
        """打印当前步骤的状态总结"""
        print(f"\n[Step {self.episode_step}] 状态总结:")
        print(f"  已完成任务: {self.step_stats['tasks_completed']}")
        print(f"  超时任务: {self.step_stats['tasks_timeout']}")
        
        # 打印设备负载状态
        print("  端侧设备任务负载:")
        for i in range(min(3, self.num_devices)):
            ue = self.user_equipments[i]
            load = ue.calculate_task_load()
            print(f"    UE{i}: 任务负载={load:.1f}s")
        
        print("  边缘服务器负载:")
        for i, es in enumerate(self.edge_servers):
            load = es.calculate_task_load()
            print(f"    ES{i}: CPU={es.cpu_frequency}GHz, 负载={load:.1f}s")

    def _get_observation(self):
        """
        获取环境观察（简化版）
        
        状态组成：
        1. UE状态：CPU频率、任务负载
        2. ES状态：CPU频率、任务负载  
        3. CS状态：CPU频率
        4. 任务状态：类型、数据大小、CPU周期、截止时间、剩余时间、紧急程度
        """
        observation = []
        
        # 1. UE状态 (每个设备2个特征)
        for ue in self.user_equipments:
            ue_state = ue.get_state()  # [CPU频率, 任务负载]
            observation.extend(ue_state)
            
        # 2. ES状态 (每个服务器2个特征)
        for es in self.edge_servers:
            es_state = es.get_state()  # [CPU频率, 任务负载]
            observation.extend(es_state)
            
        # 3. CS状态 (每个服务器1个特征)
        for cs in self.cloud_servers:
            cs_state = cs.get_state()  # [CPU频率]
            observation.extend(cs_state)
            
        # 4. 任务状态 (每个任务6个特征)
        if self.current_tasks:
            for i, task in enumerate(self.current_tasks):
                if task is not None:
                    # 任务类型归一化
                    if task.task_type == 'small':
                        task_type_norm = 0.0
                    elif task.task_type == 'medium':
                        task_type_norm = 0.5
                    else:
                        task_type_norm = 1.0
                        
                    # 数据大小归一化 - 修复属性名称
                    data_size_norm = min(task.task_data_size / 200.0, 1.0)
                    
                    # CPU周期归一化 - 修复属性名称
                    workload_norm = min(task.task_workload / 1e10, 1.0)
                    
                    # 截止时间归一化
                    deadline_norm = min(task.deadline / 100.0, 1.0)
                    
                    # 剩余时间归一化
                    remaining_time = max(task.deadline - self.global_time, 0)
                    remaining_time_norm = min(remaining_time / 100.0, 1.0)
                    
                    # 紧急程度
                    urgency = 1.0 - (remaining_time / task.deadline if task.deadline > 0 else 0)
                    
                    task_state = [
                        task_type_norm,
                        data_size_norm,
                        workload_norm,
                        deadline_norm,
                        remaining_time_norm,
                        urgency
                    ]
                else:
                    # 没有任务，填充零
                    task_state = [0.0] * 6
                
                observation.extend(task_state)
        else:
            # 如果没有任务，填充零
            for _ in range(self.num_devices):
                observation.extend([0.0] * 6)
                
        return np.array(observation, dtype=np.float32)

    def extract_agent_state(self, global_state, agent_id):
        """
        正确提取单个Agent的观察状态
        
        Args:
            global_state: 全局状态向量 (101维)
            agent_id: Agent ID (0到num_devices-1)
            
        Returns:
            agent_state: Agent的局部状态 (20维)
            
        状态结构说明:
        - 全局状态: [UE状态(30维) + ES状态(10维) + CS状态(1维) + 任务状态(60维)] = 101维
        - Agent状态: [自己UE状态(3维) + 所有ES状态(10维) + CS状态(1维) + 自己任务状态(6维)] = 20维
        """
        if agent_id < 0 or agent_id >= self.num_devices:
            raise ValueError(f"Agent ID {agent_id} 超出范围 [0, {self.num_devices-1}]")
        
        # 状态分割点计算
        ue_states_end = self.num_devices * 2
        es_states_end = ue_states_end + self.num_edges * 2
        cs_states_end = es_states_end + self.num_clouds * 1
        task_states_end = cs_states_end + self.num_devices * 6
        
        # 1. 提取当前Agent的UE状态 (2维)
        agent_ue_start = agent_id * 2
        agent_ue_state = global_state[agent_ue_start:agent_ue_start + 2]
        
        # 2. 提取所有边缘服务器状态 (10维) - 共享信息
        es_state = global_state[ue_states_end:es_states_end]
        
        # 3. 提取云服务器状态 (1维) - 共享信息  
        cs_state = global_state[es_states_end:cs_states_end]
        
        # 4. 提取当前Agent的任务状态 (6维)
        agent_task_start = cs_states_end + agent_id * 6
        agent_task_state = global_state[agent_task_start:agent_task_start + 6]
        
        # 组合Agent的完整状态
        agent_state = np.concatenate([
            agent_ue_state,    # 3维：自己的设备状态
            es_state,          # 10维：所有边缘服务器状态  
            cs_state,          # 1维：云服务器状态
            agent_task_state   # 6维：自己的任务状态
        ])
        
        return agent_state.astype(np.float32)

    def get_agent_state_dim(self):
        """获取单个Agent的状态维度
        
        Agent状态结构：
        - 自己UE状态: 2维 (CPU频率, 任务负载)
        - 所有ES状态: 2×5=10维 (CPU频率, 任务负载)
        - CS状态: 1维 (CPU频率)
        - 自己任务状态: 6维 (任务类型, 数据大小, CPU周期, 截止时间, 剩余时间, 紧急程度)
        
        总计: 2 + 10 + 1 + 6 = 19维
        """
        return 2 + (self.num_edges * 2) + (self.num_clouds * 1) + 6

    def get_device_info(self):
        """获取设备信息（用于LLM咨询）"""
        device_info = []
        for i, ue in enumerate(self.user_equipments):
            info = {
                'device_id': i,
                'cpu_frequency': ue.cpu_frequency,
                'task_load': ue.calculate_task_load()
            }
            device_info.append(info)
        return device_info

    def get_edge_info(self):
        """获取边缘服务器信息（用于LLM咨询）"""
        edge_info = []
        for i, es in enumerate(self.edge_servers):
            info = {
                'server_id': i,
                'cpu_frequency': es.cpu_frequency,
                'task_load': es.calculate_task_load()
            }
            edge_info.append(info)
        return edge_info

    def get_cloud_info(self):
        """获取云服务器信息（用于LLM咨询）"""
        cloud_info = []
        for i, cs in enumerate(self.cloud_servers):
            info = {
                'server_id': i,
                'cpu_frequency': cs.cpu_frequency,
                'is_available': True  # 云资源始终可用
            }
            cloud_info.append(info)
        return cloud_info

    def get_current_tasks_info(self):
        """获取当前任务信息（用于LLM咨询）"""
        tasks_info = []
        if self.current_tasks:
            for i, task in enumerate(self.current_tasks):
                # 🔧 修复：过滤掉None任务
                if task is not None:
                    info = {
                        'task_id': task.task_id,
                        'device_id': i,
                        'task_type': task.task_type,
                        'data_size': task.task_data_size,  # 修复属性名称
                        'cpu_cycles': task.task_workload,     # 修复属性名称
                        'deadline': task.deadline,
                        'remaining_time': max(task.deadline - self.global_time, 0)
                    }
                    tasks_info.append(info)
        return tasks_info

    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print("\n=== 环境状态（简化版）===")
            print(f"Episode Step: {self.episode_step}")
            print(f"Global Time: {self.global_time:.1f}s")
            
            # 显示设备状态
            print("\n设备状态:")
            for i in range(min(3, self.num_devices)):
                ue = self.user_equipments[i]
                load = ue.calculate_task_load()
                print(f"  UE{i}: CPU={ue.cpu_frequency:.1f}GHz, "
                      f"负载={load:.1f}s")
            
            # 显示边缘服务器状态
            print("\n边缘服务器状态:")
            for i, es in enumerate(self.edge_servers):
                load = es.calculate_task_load()
                print(f"  ES{i}: CPU={es.cpu_frequency}GHz, 负载={load:.1f}s")
            
            # 显示云服务器状态
            print("\n云服务器状态:")
            for i, cs in enumerate(self.cloud_servers):
                print(f"  CS{i}: CPU={cs.cpu_frequency}GHz (资源无限)")

    def close(self):
        """关闭环境"""
        pass

    def _check_task_completion(self, task, actual_latency):
        """
        检查并记录任务完成状态
        
        Args:
            task: 任务对象
            actual_latency: 实际完成延迟
        """
        # 🆕 更新并发任务计数
        if self.task_generation_state['total_concurrent_tasks'] > 0:
            self.task_generation_state['total_concurrent_tasks'] -= 1
        
        # 记录任务完成时间
        completion_time = self.global_time + actual_latency
        self.task_completion_stats['completion_times'].append(completion_time)
        
        # 检查是否超过截止时间
        if actual_latency <= task.deadline:
            # 按时完成
            self.task_completion_stats['tasks_completed_on_time'] += 1
            self.step_stats['tasks_completed'] += 1
            
            print(f"    ✅ 任务{task.task_id}按时完成: {actual_latency:.2f}s <= {task.deadline:.2f}s")
        else:
            # 超时完成
            overtime = actual_latency - task.deadline
            self.task_completion_stats['tasks_completed_late'] += 1
            self.task_completion_stats['deadline_violations'].append({
                'task_id': task.task_id,
                'task_type': task.task_type,
                'deadline': task.deadline,
                'actual_time': actual_latency,
                'overtime': overtime,
                'step': self.episode_step
            })
            self.step_stats['tasks_timeout'] += 1
            
            # 记录超时原因
            if overtime < 1.0:
                reason = "轻微超时"
            elif overtime < 5.0:
                reason = "中度超时"
            else:
                reason = "严重超时"
            
            self.task_completion_stats['timeout_reasons'].append({
                'task_id': task.task_id,
                'reason': reason,
                'overtime': overtime
            })
            
            print(f"    ⚠️ 任务{task.task_id}超时完成: {actual_latency:.2f}s > {task.deadline:.2f}s (超时{overtime:.2f}s)")

    def get_task_completion_rate(self):
        """
        计算任务完成率
        
        Returns:
            dict: 包含各种完成率指标的字典
        """
        stats = self.task_completion_stats
        total_attempted = stats['tasks_completed_on_time'] + stats['tasks_completed_late'] + stats['tasks_failed']
        
        if total_attempted == 0:
            return {
                'overall_completion_rate': 1.0,
                'on_time_completion_rate': 1.0,
                'timeout_rate': 0.0,
                'failure_rate': 0.0,
                'total_tasks': 0,
                'completed_on_time': 0,
                'completed_late': 0,
                'failed': 0
            }
        
        # 计算各项指标
        overall_completion_rate = (stats['tasks_completed_on_time'] + stats['tasks_completed_late']) / total_attempted
        on_time_completion_rate = stats['tasks_completed_on_time'] / total_attempted
        timeout_rate = stats['tasks_completed_late'] / total_attempted
        failure_rate = stats['tasks_failed'] / total_attempted
        
        return {
            'overall_completion_rate': overall_completion_rate,        # 总完成率（按时+超时）
            'on_time_completion_rate': on_time_completion_rate,        # 按时完成率
            'timeout_rate': timeout_rate,                              # 超时完成率
            'failure_rate': failure_rate,                              # 失败率
            'total_tasks': total_attempted,                            # 总任务数
            'completed_on_time': stats['tasks_completed_on_time'],     # 按时完成数
            'completed_late': stats['tasks_completed_late'],           # 超时完成数
            'failed': stats['tasks_failed'],                           # 失败任务数
            'avg_completion_time': np.mean(stats['completion_times']) if stats['completion_times'] else 0,
            'avg_overtime': np.mean([v['overtime'] for v in stats['deadline_violations']]) if stats['deadline_violations'] else 0
        }