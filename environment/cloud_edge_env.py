# environment/cloud_edge_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .device_models import UserEquipment, EdgeServer, CloudServer
from .task_generator import TaskGenerator, Task


class CloudEdgeDeviceEnv(gym.Env):
    """云边端三层架构卸载环境 - 支持每step生成新任务"""
    
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
        self.task_generator = TaskGenerator(task_config)

        # 状态空间定义 (归一化后的状态)
        self.state_dim = (
            self.num_devices * 4 +     # UE状态: CPU频率, CPU负载, 电池, 内存
            self.num_edges * 3 +       # ES状态: CPU频率, CPU负载, 内存  
            self.num_clouds * 3 +      # CS状态: CPU频率, CPU负载, 内存
            self.num_devices * 4       # 任务状态: 类型, 数据大小, CPU周期, 截止时间
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

        # 动作空间定义：每个设备的三元分割决策 [α1, α2, α3]
        # α1: 本地处理比例, α2: 边缘处理比例, α3: 云端处理比例
        # 并且需要选择具体的边缘服务器
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0]),      # [α1, α2, α3, edge_id]
            high=np.array([1.0, 1.0, 1.0, self.num_edges - 1]),
            dtype=np.float32
        )

        # 当前任务（每step更新）
        self.current_tasks = None
        self.episode_step = 0
        self.max_steps = 100

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
            es = EdgeServer(i, edge_frequencies[i])
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
        """重置环境 - 不预生成任务，等待第一个step"""
        if seed is not None:
            np.random.seed(seed)
            
        # 重置所有设备状态
        for ue in self.user_equipments:
            ue.reset()
        for es in self.edge_servers:
            es.reset()
        for cs in self.cloud_servers:
            cs.reset()

        # 重置episode计数器
        self.episode_step = 0
        
        # 生成第一步的任务（每step都会生成新任务）
        self._generate_new_tasks()

        return self._get_observation(), {}

    def _generate_new_tasks(self):
        """为每个设备生成新任务"""
        print(f"  生成新任务 (Step {self.episode_step + 1})")
        task_data_list = self.task_generator.generate_tasks(self.num_devices)
        self.current_tasks = [Task(task_data) for task_data in task_data_list]

    def step(self, actions, llm_actions=None):
        """
        执行一步动作
        
        Args:
            actions: Agent的动作 shape=(num_devices, 4) [α1, α2, α3, edge_id]
            llm_actions: LLM专家动作 shape=(num_devices, 4) 或 list
        
        Returns:
            observation, rewards, terminated, truncated, info
        """
        self.episode_step += 1
        
        # 解析动作：actions是形状为(num_devices, 4)的数组
        rewards = np.zeros(self.num_devices)
        
        # 性能指标统计
        total_latencies = []
        total_energies = []
        local_baselines = []
        
        # 为每个设备执行卸载决策
        for i in range(self.num_devices):
            action = actions[i] if len(actions.shape) > 1 else actions
            reward, metrics = self._execute_offloading_decision(i, action)
            rewards[i] = reward
            
            total_latencies.append(metrics['total_latency'])
            total_energies.append(metrics['total_energy'])
            local_baselines.append(metrics['local_baseline'])

        # 检查终止条件
        max_steps_reached = self.episode_step >= self.max_steps
        
        # 在新的逻辑中，每个step处理完任务后，不需要任务完成状态跟踪
        # Episode只在达到最大步数时终止
        terminated = False
        truncated = max_steps_reached

        # 如果还没结束，为下一步生成新任务
        if not (terminated or truncated):
            self._generate_new_tasks()

        # 构建info字典
        info = {
            'total_latencies': total_latencies,
            'total_energies': total_energies, 
            'local_baselines': local_baselines,
            'episode_step': self.episode_step,
            'llm_actions': llm_actions if llm_actions is not None else []
        }

        return self._get_observation(), rewards, terminated, truncated, info

    def _execute_offloading_decision(self, device_idx, action):
        """执行单个设备的卸载决策"""
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
        
        # 计算卸载策略的时延和能耗
        total_latency, total_energy = self._calculate_offloading_performance(
            ue, task, edge_id
        )
        
        # 计算本地基准
        baseline_latency, baseline_energy = self._calculate_local_baseline(ue, task)
        
        # 计算奖励函数
        reward = self._calculate_reward(
            total_latency, total_energy, baseline_latency, baseline_energy
        )
        
        # 检查任务是否在截止时间内完成
        if total_latency > task.deadline:
            # 任务超时惩罚
            reward -= 1.0
            
        # 更新设备状态
        ue.battery -= total_energy * 0.001  # 简化的电池消耗更新
        
        metrics = {
            'total_latency': total_latency,
            'total_energy': total_energy,
            'local_baseline': (baseline_latency, baseline_energy)
        }
        
        return reward, metrics

    def _calculate_offloading_performance(self, ue, task, edge_id):
        """计算卸载策略的总时延和总能耗"""
        workloads = task.get_split_workloads()  # [本地, 边缘, 云端]工作负载
        data_sizes = task.get_split_data_sizes()  # [本地, 边缘, 云端]数据大小
        
        es = self.edge_servers[edge_id]
        cs = self.cloud_servers[0]  # 假设只有一个云服务器
        
        # 计算各部分时延和能耗
        latencies = []
        energies = []
        
        # 1. 本地计算部分
        if workloads[0] > 0:
            local_latency = ue.calculate_execution_time(workloads[0])
            local_energy = ue.calculate_energy_consumption(workloads[0])
            latencies.append(local_latency)
            energies.append(local_energy)
        else:
            latencies.append(0.0)
            energies.append(0.0)
            
        # 2. 边缘计算部分
        if workloads[1] > 0:
            # UE到ES传输时间和能耗
            trans_time_ue_es = ue.calculate_transmission_time(data_sizes[1])
            trans_energy_ue_es = ue.calculate_transmission_energy(trans_time_ue_es)
            
            # ES计算时间和能耗
            edge_exec_time = es.calculate_execution_time(workloads[1])
            edge_energy = es.calculate_energy_consumption(workloads[1])
            
            # 边缘部分总时延 = 传输时间 + 计算时间
            edge_total_latency = trans_time_ue_es + edge_exec_time
            edge_total_energy = trans_energy_ue_es + edge_energy
            
            latencies.append(edge_total_latency)
            energies.append(edge_total_energy)
        else:
            latencies.append(0.0)
            energies.append(0.0)
            
        # 3. 云计算部分
        if workloads[2] > 0:
            # UE到ES传输
            trans_time_ue_es = ue.calculate_transmission_time(data_sizes[2])
            trans_energy_ue_es = ue.calculate_transmission_energy(trans_time_ue_es)
            
            # ES到CS传输
            trans_time_es_cs = es.calculate_transmission_time_to_cloud(data_sizes[2])
            
            # CS计算时间和能耗
            cloud_exec_time = cs.calculate_execution_time(workloads[2])
            cloud_energy = cs.calculate_energy_consumption(workloads[2])
            
            # 云部分总时延 = UE到ES传输 + ES到CS传输 + 云计算时间
            cloud_total_latency = trans_time_ue_es + trans_time_es_cs + cloud_exec_time
            cloud_total_energy = trans_energy_ue_es + cloud_energy  # 云端能耗很小
            
            latencies.append(cloud_total_latency)
            energies.append(cloud_total_energy)
        else:
            latencies.append(0.0)
            energies.append(0.0)
        
        # 总时延 = 最大时延（并行执行）
        total_latency = max(latencies)
        
        # 总能耗 = 各部分能耗之和
        total_energy = sum(energies)
        
        return total_latency, total_energy

    def _calculate_local_baseline(self, ue, task):
        """计算本地全部执行的基准时延和能耗"""
        baseline_latency = ue.calculate_execution_time(task.cpu_cycles)
        baseline_energy = ue.calculate_energy_consumption(task.cpu_cycles)
        return baseline_latency, baseline_energy

    def _calculate_reward(self, offload_latency, offload_energy, 
                         baseline_latency, baseline_energy):
        """
        计算奖励函数
        
        奖励 = (本地时延 - 卸载时延) / 本地时延 + (本地能耗 - 卸载能耗) / 本地能耗
        """
        # 避免除零错误
        if baseline_latency == 0:
            latency_improvement = 0
        else:
            latency_improvement = (baseline_latency - offload_latency) / baseline_latency
            
        if baseline_energy == 0:
            energy_improvement = 0
        else:
            energy_improvement = (baseline_energy - offload_energy) / baseline_energy
        
        # 奖励函数
        reward = latency_improvement + energy_improvement
        
        # 确保奖励在合理范围内
        reward = np.clip(reward, -2.0, 2.0)
        
        return reward

    def _get_observation(self):
        """构建观测状态"""
        # UE状态
        ue_states = []
        for ue in self.user_equipments:
            ue_states.extend(ue.get_state())
            
        # ES状态  
        es_states = []
        for es in self.edge_servers:
            es_states.extend(es.get_state())
            
        # CS状态
        cs_states = []
        for cs in self.cloud_servers:
            cs_states.extend(cs.get_state())
            
        # 当前任务状态（归一化）
        task_states = []
        for task in self.current_tasks:
            # 任务类型编码
            type_encoding = {'small': 0.33, 'medium': 0.67, 'large': 1.0}
            task_type_val = type_encoding.get(task.task_type, 0.67)
            
            # 数据大小归一化 (最大200MB)
            data_size_norm = min(task.data_size_mb / 200.0, 1.0)
            
            # CPU周期归一化 (最大40Gcycles，对应200MB任务)
            cpu_cycles_norm = min(task.cpu_cycles / (200 * 0.2e9), 1.0)
            
            # 截止时间归一化 (最大100秒)
            deadline_norm = min(task.deadline / 100.0, 1.0)
            
            task_states.extend([
                task_type_val, data_size_norm, cpu_cycles_norm, deadline_norm
            ])
            
        # 合并所有状态
        full_state = np.array(ue_states + es_states + cs_states + task_states, 
                             dtype=np.float32)
        
        return full_state

    def get_device_info(self):
        """获取设备信息，用于LLM咨询"""
        device_info = []
        for ue in self.user_equipments:
            device_info.append({
                "cpu": ue.cpu_frequency,
                "memory": 2.0,  # 假设2GB内存
                "battery": ue.battery / ue.battery_capacity
            })
        return device_info
    
    def get_edge_info(self):
        """获取边缘服务器信息"""
        edge_info = []
        for es in self.edge_servers:
            edge_info.append({
                "cpu": es.cpu_frequency,
                "memory": es.memory_capacity
            })
        return edge_info
    
    def get_cloud_info(self):
        """获取云服务器信息"""
        cloud_info = []
        for cs in self.cloud_servers:
            cloud_info.append({
                "cpu": cs.cpu_frequency,
                "memory": cs.memory_capacity
            })
        return cloud_info

    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"\n=== Episode Step {self.episode_step} ===")
            
            # 显示设备状态
            print("\nUE States:")
            for i, ue in enumerate(self.user_equipments):
                print(f"  UE{i}: CPU={ue.cpu_frequency:.2f}GHz, Load={ue.current_cpu_load:.2f}, Battery={ue.battery:.0f}mAh")
                
            print("\nES States:")
            for i, es in enumerate(self.edge_servers):
                print(f"  ES{i}: CPU={es.cpu_frequency:.0f}GHz, Load={es.current_cpu_load:.2f}")
                
            print("\nCurrent Tasks:")
            for i, task in enumerate(self.current_tasks):
                print(f"  Task{i}: {task.task_type}, {task.data_size_mb:.1f}MB, {task.deadline:.2f}s")

    def close(self):
        """关闭环境"""
        pass