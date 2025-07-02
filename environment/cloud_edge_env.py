# environment/cloud_edge_env.py
"""
云边端三层架构卸载环境 - 简化版设备模型
- UE: CPU频率 + 电池 + 任务负载
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

        # 系统架构配置
        self.num_devices = 10      # 10个端侧设备
        self.num_edges = 5         # 5个边缘服务器  
        self.num_clouds = 1        # 1个云服务器

        # 创建设备
        self._create_devices()
        
        # 任务生成器
        task_config = config.get('task_config', {})
        if not task_config:
            # 使用默认任务配置
            task_config = {
                'task_types': {
                    'small': {'data_range': [1, 5], 'probability': 0.4},
                    'medium': {'data_range': [10, 50], 'probability': 0.4},
                    'large': {'data_range': [100, 200], 'probability': 0.2}
                },
                'processing_density': 0.2e9,
                'deadline_multiplier': 2.0
            }
        self.task_generator = TaskGenerator(task_config)

        # 状态空间定义 (简化后的状态)
        self.state_dim = (
            self.num_devices * 3 +     # UE状态: CPU频率, 电池, 任务负载
            self.num_edges * 2 +       # ES状态: CPU频率, 任务负载
            self.num_clouds * 1 +      # CS状态: CPU频率
            self.num_devices * 6       # 任务状态: 类型, 数据大小, CPU周期, 截止时间, 剩余时间, 紧急程度
        )

        # 单个Agent的状态维度
        self.agent_state_dim = (
            3 +                        # 自己的UE状态: CPU频率, 电池, 任务负载
            self.num_edges * 2 +       # 所有ES状态: CPU频率, 任务负载 (共享信息)
            self.num_clouds * 1 +      # 所有CS状态: CPU频率 (共享信息)
            6                          # 自己的任务状态: 类型, 数据大小, 任务所需的CPU周期, 截止时间, 剩余时间, 紧急程度
        )

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
        self.max_steps = 100
        
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
            
        # 创建边缘服务器（异构配置：{5, 6, 8, 10, 12} GHz）
        edge_frequencies = [5, 6, 8, 10, 12]
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
        """为每个设备生成新任务"""
        print(f"[Step {self.episode_step}] 生成新任务...")
        task_data_list = self.task_generator.generate_tasks(self.num_devices)
        self.current_tasks = [Task(task_data) for task_data in task_data_list]
        
        # 为新任务设置创建步骤
        for i, task in enumerate(self.current_tasks):
            task.creation_step = self.episode_step
            
        # 更新任务生成统计
        self.task_completion_stats['total_tasks_generated'] += len(self.current_tasks)

    def step(self, actions, llm_actions=None):
        """
        执行一步动作 - 考虑差异化通信延迟
        
        Args:
            actions: Agent的动作 shape=(num_devices, 4) [α1, α2, α3, edge_id]
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
        
        # 2. 处理新任务的卸载决策
        print(f"\n[Step {self.episode_step}] 🚀 执行MADDPG卸载决策...")
        rewards = np.zeros(self.num_devices)
        total_latencies = []
        total_energies = []
        communication_latencies = []
        computation_latencies = []
        
        for i in range(self.num_devices):
            action = actions[i] if len(actions.shape) > 1 else actions
            reward, metrics = self._execute_offloading_decision(i, action)
            rewards[i] = reward
            
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
        print(f"  平均奖励: {np.mean(rewards):.3f}")
        print(f"  奖励范围: [{np.min(rewards):.3f}, {np.max(rewards):.3f}]")

        # 3. 检查终止条件
        max_steps_reached = self.episode_step >= self.max_steps
        terminated = False
        truncated = max_steps_reached

        # 4. 如果还没结束，为下一步生成新任务
        if not (terminated or truncated):
            self._generate_new_tasks()

        # 5. 打印当前状态总结
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
            'maddpg_rewards': rewards.tolist()
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
        task = self.current_tasks[device_idx]
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
            total_latency, total_energy, baseline_latency, baseline_energy, task.deadline
        )
        
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
            
            # 消耗电池
            ue.consume_battery(energy)
            
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
            
            # 消耗电池（传输能耗）
            ue.consume_battery(comm_energy)
            
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
            
            # 消耗电池（传输能耗）
            ue.consume_battery(comm_energy)
            
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
                         baseline_latency, baseline_energy, deadline):
        """
        计算奖励函数 - 考虑差异化通信延迟的影响
        
        奖励设计：
        1. 时延改善奖励
        2. 能耗改善奖励  
        3. 截止时间满足奖励
        4. 负载均衡奖励
        5. 通信效率奖励（鼓励减少不必要的云端卸载）
        """
        # 基础奖励计算
        if baseline_latency > 0:
            latency_improvement = (baseline_latency - offload_latency) / baseline_latency
        else:
            latency_improvement = 0
            
        if baseline_energy > 0:
            energy_improvement = (baseline_energy - offload_energy) / baseline_energy
        else:
            energy_improvement = 0
            
        # 时延和能耗奖励
        latency_reward = latency_improvement * 10.0
        energy_reward = energy_improvement * 5.0
        
        # 截止时间满足奖励
        if offload_latency <= deadline:
            deadline_reward = 5.0
        else:
            overtime_ratio = (offload_latency - deadline) / deadline
            deadline_reward = -10.0 * overtime_ratio
        
        # 负载均衡奖励
        edge_loads = [es.calculate_task_load() for es in self.edge_servers]
        if len(edge_loads) > 1:
            load_variance = np.var(edge_loads)
            balance_reward = -load_variance * 0.01
        else:
            balance_reward = 0
        
        # 总奖励
        total_reward = latency_reward + energy_reward + deadline_reward + balance_reward
        
        return total_reward

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
            battery = ue.get_battery_percentage()
            print(f"    UE{i}: 任务负载={load:.1f}s, 电池={battery:.0%}")
        
        print("  边缘服务器负载:")
        for i, es in enumerate(self.edge_servers):
            load = es.calculate_task_load()
            print(f"    ES{i}: CPU={es.cpu_frequency}GHz, 负载={load:.1f}s")

    def _get_observation(self):
        """
        获取环境观察（简化版）
        
        状态组成：
        1. UE状态：CPU频率、电池、任务负载
        2. ES状态：CPU频率、任务负载  
        3. CS状态：CPU频率
        4. 任务状态：类型、数据大小、CPU周期、截止时间、剩余时间、紧急程度
        """
        observation = []
        
        # 1. UE状态 (每个设备3个特征)
        for ue in self.user_equipments:
            ue_state = ue.get_state()  # [CPU频率, 电池, 任务负载]
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
            for task in self.current_tasks:
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
        ue_states_end = self.num_devices * 3  # 30
        es_states_end = ue_states_end + self.num_edges * 2  # 40  
        cs_states_end = es_states_end + self.num_clouds * 1  # 41
        task_states_end = cs_states_end + self.num_devices * 6  # 101
        
        # 1. 提取当前Agent的UE状态 (3维)
        agent_ue_start = agent_id * 3
        agent_ue_state = global_state[agent_ue_start:agent_ue_start + 3]
        
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
        """获取单个Agent的状态维度"""
        return self.agent_state_dim

    def get_device_info(self):
        """获取设备信息（用于LLM咨询）"""
        device_info = []
        for i, ue in enumerate(self.user_equipments):
            info = {
                'device_id': i,
                'cpu_frequency': ue.cpu_frequency,
                'battery_percentage': ue.get_battery_percentage(),
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
                battery = ue.get_battery_percentage()
                print(f"  UE{i}: CPU={ue.cpu_frequency:.1f}GHz, "
                      f"负载={load:.1f}s, 电池={battery:.0%}")
            
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