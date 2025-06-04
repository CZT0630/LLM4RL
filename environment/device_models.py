# environment/device_models.py
class Device:
    def __init__(self, device_id, cpu_capacity=2.0, memory_capacity=4.0, battery_capacity=100.0):
        self.device_id = device_id
        self.cpu_capacity = cpu_capacity  # GHz
        self.memory_capacity = memory_capacity  # GB
        self.battery_capacity = battery_capacity  # mAh

        # 动态状态
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        self.battery = battery_capacity

    def reset(self):
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        self.battery = self.battery_capacity

    def execute(self, workload):
        # 执行计算任务，返回执行延迟
        execution_time = workload / self.cpu_capacity
        self.current_cpu_load += workload / self.cpu_capacity

        # 限制负载不超过100%
        if self.current_cpu_load > 1.0:
            self.current_cpu_load = 1.0

        return execution_time

    def consume_energy(self, workload):
        # 计算执行任务消耗的能量
        energy_per_computation = 0.1  # 简化的能量模型
        return workload * energy_per_computation

    def get_state(self):
        # 返回设备状态
        return [
            self.cpu_capacity,
            self.current_cpu_load,
            self.battery / self.battery_capacity,
            self.current_memory_usage / self.memory_capacity
        ]


class EdgeServer:
    def __init__(self, server_id, cpu_capacity=8.0, memory_capacity=16.0):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity  # GHz
        self.memory_capacity = memory_capacity  # GB
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0

    def reset(self):
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0

    def execute(self, workload):
        # 执行计算任务，返回执行延迟
        execution_time = workload / self.cpu_capacity
        self.current_cpu_load += workload / self.cpu_capacity

        # 限制负载不超过100%
        if self.current_cpu_load > 1.0:
            self.current_cpu_load = 1.0

        return execution_time

    def get_state(self):
        # 返回服务器状态
        return [
            self.cpu_capacity,
            self.current_cpu_load,
            self.current_memory_usage / self.memory_capacity
        ]


class CloudServer:
    def __init__(self, server_id, cpu_capacity=32.0, memory_capacity=64.0):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity  # GHz
        self.memory_capacity = memory_capacity  # GB
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0

    def reset(self):
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0

    def execute(self, workload):
        # 执行计算任务，返回执行延迟
        execution_time = workload / self.cpu_capacity
        self.current_cpu_load += workload / self.cpu_capacity

        # 限制负载不超过100%
        if self.current_cpu_load > 1.0:
            self.current_cpu_load = 1.0

        return execution_time

    def get_state(self):
        # 返回服务器状态
        return [
            self.cpu_capacity,
            self.current_cpu_load,
            self.current_memory_usage / self.memory_capacity
        ]