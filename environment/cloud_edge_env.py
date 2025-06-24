# environment/cloud_edge_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .device_models import Device, EdgeServer, CloudServer
from .task_generator import TaskGenerator


class CloudEdgeDeviceEnv(gym.Env):
    def __init__(self, config):
        super(CloudEdgeDeviceEnv, self).__init__()

        # 环境参数
        self.num_devices = config['num_devices']
        self.num_edges = config['num_edges']
        self.num_clouds = config['num_clouds']

        # 创建设备、边缘服务器和云端服务器
        self.devices = [Device(i, **config['device_config']) for i in range(self.num_devices)]
        self.edge_servers = [EdgeServer(i, **config['edge_config']) for i in range(self.num_edges)]
        self.cloud_servers = [CloudServer(i, **config['cloud_config']) for i in range(self.num_clouds)]

        # 任务生成器
        self.task_generator = TaskGenerator(config['task_config'])

        # 状态空间定义
        self.state_dim = (
                self.num_devices * 4 +  # 设备状态: CPU, 内存, 电池, 负载
                self.num_edges * 3 +  # 边缘状态: CPU, 内存, 负载
                self.num_clouds * 3 +  # 云端状态: CPU, 内存, 负载
                self.num_devices * 3  # 任务状态: 计算量, 数据量, 截止时间
        )

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.state_dim,), dtype=np.float32
        )

        # 动作空间定义 (每个设备决定卸载比例和目标)
        self.action_dim = 2  # 卸载比例, 目标节点
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([1, self.num_edges + self.num_clouds - 1]),
            dtype=np.float32
        )

        # 初始化环境
        self.reset()

    def reset(self):
        # 重置所有设备、边缘和云端状态
        for device in self.devices:
            device.reset()

        for edge in self.edge_servers:
            edge.reset()

        for cloud in self.cloud_servers:
            cloud.reset()

        # 生成新任务
        self.tasks = self.task_generator.generate_tasks(self.num_devices)
        # 新增：初始化任务完成状态
        self.task_done_flags = [False for _ in range(self.num_devices)]

        return self._get_observation()

    def step(self, actions):
        rewards = np.zeros(self.num_devices)
        done = False

        # 执行卸载决策
        for i, device in enumerate(self.devices):
            action = actions[i]
            offload_ratio = action[0]
            target_idx = int(action[1])

            task = self.tasks[i]

            # 计算本地执行部分
            local_ratio = 1.0 - offload_ratio
            local_workload = task['computation'] * local_ratio

            # 执行本地计算
            local_delay = device.execute(local_workload)
            local_energy = device.consume_energy(local_workload)

            # 卸载部分任务
            if offload_ratio > 0:
                offload_workload = task['computation'] * offload_ratio

                if target_idx < self.num_edges:  # 卸载到边缘
                    edge = self.edge_servers[target_idx]
                    edge_delay = edge.execute(offload_workload)
                    transmission_delay = self._calculate_transmission_delay(
                        device, edge, task['data_size'] * offload_ratio
                    )
                    total_delay = local_delay + max(edge_delay, transmission_delay)

                else:  # 卸载到云端
                    cloud_idx = target_idx - self.num_edges
                    cloud = self.cloud_servers[cloud_idx]
                    cloud_delay = cloud.execute(offload_workload)
                    transmission_delay = self._calculate_transmission_delay(
                        device, cloud, task['data_size'] * offload_ratio
                    )
                    total_delay = local_delay + max(cloud_delay, transmission_delay)
            else:
                total_delay = local_delay

            # 计算奖励 (基于延迟和能耗)
            rewards[i] = self._calculate_reward(total_delay, local_energy, task)

            # 更新设备电池
            device.battery -= local_energy

            # 检查任务是否完成
            if total_delay > task['deadline']:
                rewards[i] -= 10  # 任务超时惩罚
                self.task_done_flags[i] = False  # 超时视为未完成
            elif total_delay <= task['deadline']:
                self.task_done_flags[i] = True  # 正常完成

        # 检查是否所有设备电池耗尽或任务完成
        battery_depleted = all(device.battery <= 0 for device in self.devices)
        tasks_completed = all(self._is_task_completed(i) for i in range(self.num_devices))

        done = battery_depleted or tasks_completed

        # 更新环境状态
        next_state = self._get_observation()

        return next_state, rewards, done, {}

    def _get_observation(self):
        # 构建完整的观测状态
        device_states = np.array([
            device.get_state() for device in self.devices
        ]).flatten()

        edge_states = np.array([
            edge.get_state() for edge in self.edge_servers
        ]).flatten()

        cloud_states = np.array([
            cloud.get_state() for cloud in self.cloud_servers
        ]).flatten()

        task_states = np.array([
            [task['computation'], task['data_size'], task['deadline']]
            for task in self.tasks
        ]).flatten()

        return np.concatenate([device_states, edge_states, cloud_states, task_states])

    def _calculate_reward(self, delay, energy, task):
        # 计算奖励 (延迟越小越好，能耗越低越好)
        delay_penalty = -delay / task['deadline']
        energy_penalty = -energy / 100  # 假设最大能耗为100

        # 负载均衡奖励 (如果卸载到负载较低的节点)
        return delay_penalty + energy_penalty

    def _calculate_transmission_delay(self, source, destination, data_size):
        # 计算传输延迟
        bandwidth = 10  # Mbps，简化处理
        return data_size / bandwidth  # 秒

    def _is_task_completed(self, device_idx):
        # 检查任务是否完成
        return self.task_done_flags[device_idx]