# environment/device_models.py
"""
简化的云边端设备模型
- UE: CPU频率 + 电池 + 任务负载
- ES: CPU频率 + 任务负载  
- CS: CPU频率（资源无限）
- 考虑差异化的通信延迟
"""
import numpy as np
import random


class TaskExecution:
    """任务执行记录"""
    def __init__(self, task_id, task_workload, start_time, execution_time):
        self.task_id = task_id
        self.task_workload = task_workload
        self.start_time = start_time
        self.execution_time = execution_time
        self.remaining_time = execution_time
        self.completed = False


class UserEquipment:
    """
    端侧设备（User Equipment, UE）
    简化状态：[CPU频率, 电池, 任务负载]
    """
    
    def __init__(self, device_id, cpu_frequency=None):
        self.device_id = device_id
        
        # CPU频率：0.5-1.0 GHz异构配置
        if cpu_frequency is None:
            self.cpu_frequency = random.uniform(0.5, 1.0)  # GHz
        else:
            self.cpu_frequency = cpu_frequency
            
        # # 计算限制：单任务最大可处理 CPU 周期数 500 Mcycles
        # self.max_cpu_cycles = 500e6  # 500 * 10^6 cycles
        
        # 能耗参数
        self.alpha_ue = 1e-26  # J/cycle
        self.transmission_power = 0.5  # W (传输功率)
        
        # 网络参数（差异化通信延迟）
        self.transmission_rate_to_edge = 1e9  # 1 Gbps到边缘（无线6G）
        self.transmission_rate_to_cloud = 100e6  # 100 Mbps到云端（经边缘中转，总延迟更高）
        
        # 电池状态
        self.battery_capacity = 4000.0  # mAh
        self.battery = self.battery_capacity
        
        # 任务执行队列
        self.task_queue = []  # 当前执行和等待的任务
        self.current_execution = None  # 当前正在执行的任务
        
    def reset(self):
        """重置设备状态"""
        self.battery = self.battery_capacity
        self.task_queue.clear()
        self.current_execution = None
        
    def calculate_task_load(self):
        """
        计算任务负载：当前执行任务剩余时间 + 队列中所有任务的处理时间总和
        
        返回: 任务负载（秒）
        """
        total_load = 0.0
        
        # 当前执行任务的剩余时间
        if self.current_execution and not self.current_execution.completed:
            total_load += self.current_execution.remaining_time
            
        # 队列中等待任务的处理时间
        for task in self.task_queue:
            if not task.completed:
                total_load += task.execution_time
                
        return total_load
    
    def add_task(self, task_id, cpu_cycles, current_time):
        """
        添加新任务到队列
        
        参数:
        - task_id: 任务ID
        - cpu_cycles: CPU周期数
        - current_time: 当前时间
        """
        execution_time = self.calculate_execution_time(cpu_cycles)
        
        if self.current_execution is None or self.current_execution.completed:
            # 没有正在执行的任务，立即开始执行
            self.current_execution = TaskExecution(task_id, cpu_cycles, current_time, execution_time)
        else:
            # 添加到等待队列
            task = TaskExecution(task_id, cpu_cycles, current_time, execution_time)
            self.task_queue.append(task)
    
    def update_tasks(self, time_elapsed):
        """
        更新任务执行状态
        
        参数:
        - time_elapsed: 流逝的时间（秒）
        """
        # 更新当前执行任务
        if self.current_execution and not self.current_execution.completed:
            self.current_execution.remaining_time -= time_elapsed
            if self.current_execution.remaining_time <= 0:
                self.current_execution.completed = True
                self.current_execution = None
                
                # 开始执行队列中的下一个任务
                if self.task_queue:
                    self.current_execution = self.task_queue.pop(0)
    
    def calculate_execution_time(self, cpu_cycles):
        """计算执行时间"""
        cpu_frequency_hz = self.cpu_frequency * 1e9  # GHz转Hz
        execution_time = cpu_cycles / cpu_frequency_hz
        return execution_time
    
    def calculate_energy_consumption(self, cpu_cycles):
        """计算计算能耗"""
        energy = self.alpha_ue * cpu_cycles
        return energy
    
    def calculate_transmission_time_to_edge(self, data_size_mb):
        """计算到边缘服务器的传输时间"""
        data_size_bits = data_size_mb * 8 * 1e6  # MB转bits
        transmission_time = data_size_bits / self.transmission_rate_to_edge
        return transmission_time
    
    def calculate_transmission_time_to_cloud(self, data_size_mb):
        """
        计算到云服务器的传输时间（经边缘中转）
        包含UE→Edge→Cloud的总传输时间
        """
        data_size_bits = data_size_mb * 8 * 1e6  # MB转bits
        
        # UE到边缘的传输时间
        time_ue_to_edge = data_size_bits / self.transmission_rate_to_edge
        
        # 边缘到云的传输时间（假设边缘到云带宽为10Gbps）
        edge_to_cloud_rate = 10e9  # 10 Gbps
        time_edge_to_cloud = data_size_bits / edge_to_cloud_rate
        
        # 总传输时间
        total_transmission_time = time_ue_to_edge + time_edge_to_cloud
        return total_transmission_time
    
    def calculate_transmission_energy(self, transmission_time):
        """计算传输能耗"""
        energy = self.transmission_power * transmission_time
        return energy
    
    def consume_battery(self, energy_joules):
        """消耗电池电量"""
        # 简化转换：将焦耳转换为mAh消耗（假设3.7V）
        mah_consumed = energy_joules / (3.7 * 3600) * 1000
        self.battery = max(0, self.battery - mah_consumed)
    
    def get_battery_percentage(self):
        """获取电池百分比"""
        return self.battery / self.battery_capacity
    
    def get_state(self):
        """
        返回简化的设备状态
        返回: [CPU频率(归一化), 电池百分比, 任务负载(归一化)]
        """
        # 任务负载归一化（假设最大60秒的任务负载）
        task_load = self.calculate_task_load()
        task_load_norm = min(task_load / 60.0, 1.0)
        
        return [
            self.cpu_frequency / 1.0,      # CPU频率归一化（最大1.0GHz）
            self.get_battery_percentage(),  # 电池百分比
            task_load_norm                  # 任务负载归一化
        ]


class EdgeServer:
    """
    边缘服务器（Edge Server, ES）
    简化状态：[CPU频率, 任务负载]
    """
    
    def __init__(self, server_id, cpu_frequency):
        self.server_id = server_id
        self.cpu_frequency = cpu_frequency  # GHz
        
        # 能耗参数
        self.alpha_es = 3e-26  # J/cycle
        
        # 网络参数
        self.transmission_rate_to_cloud = 10e9  # 10 Gbps到云端
        
        # 任务执行队列
        self.task_queue = []
        self.current_execution = None
        
    def reset(self):
        """重置服务器状态"""
        self.task_queue.clear()
        self.current_execution = None
        
    def calculate_task_load(self):
        """
        计算任务负载：当前执行任务剩余时间 + 队列中所有任务的处理时间总和
        """
        total_load = 0.0
        
        # 当前执行任务的剩余时间
        if self.current_execution and not self.current_execution.completed:
            total_load += self.current_execution.remaining_time
            
        # 队列中等待任务的处理时间
        for task in self.task_queue:
            if not task.completed:
                total_load += task.execution_time
                
        return total_load
    
    def add_task(self, task_id, cpu_cycles, current_time):
        """添加新任务到队列"""
        execution_time = self.calculate_execution_time(cpu_cycles)
        
        if self.current_execution is None or self.current_execution.completed:
            self.current_execution = TaskExecution(task_id, cpu_cycles, current_time, execution_time)
        else:
            task = TaskExecution(task_id, cpu_cycles, current_time, execution_time)
            self.task_queue.append(task)
    
    def update_tasks(self, time_elapsed):
        """更新任务执行状态"""
        if self.current_execution and not self.current_execution.completed:
            self.current_execution.remaining_time -= time_elapsed
            if self.current_execution.remaining_time <= 0:
                self.current_execution.completed = True
                self.current_execution = None
                
                if self.task_queue:
                    self.current_execution = self.task_queue.pop(0)
    
    def calculate_execution_time(self, cpu_cycles):
        """计算执行时间"""
        cpu_frequency_hz = self.cpu_frequency * 1e9  # GHz转Hz
        execution_time = cpu_cycles / cpu_frequency_hz
        return execution_time
    
    def calculate_energy_consumption(self, cpu_cycles):
        """计算能耗"""
        energy = self.alpha_es * cpu_cycles
        return energy
    
    def calculate_transmission_time_to_cloud(self, data_size_mb):
        """计算到云服务器的传输时间"""
        data_size_bits = data_size_mb * 8 * 1e6  # MB转bits
        transmission_time = data_size_bits / self.transmission_rate_to_cloud
        return transmission_time
    
    def get_expected_completion_time(self, cpu_cycles):
        """
        获取新任务的预期完成时间
        包括等待时间和执行时间
        """
        execution_time = self.calculate_execution_time(cpu_cycles)
        current_load = self.calculate_task_load()
        return current_load + execution_time
    
    def get_state(self):
        """
        返回简化的服务器状态
        返回: [CPU频率(归一化), 任务负载(归一化)]
        """
        # 任务负载归一化（假设最大120秒的任务负载）
        task_load = self.calculate_task_load()
        task_load_norm = min(task_load / 120.0, 1.0)
        
        return [
            self.cpu_frequency / 12.0,  # CPU频率归一化（最大12GHz）
            task_load_norm              # 任务负载归一化
        ]


class CloudServer:
    """
    云服务器（Cloud Server, CS）
    简化状态：[CPU频率]（资源无限，无任务负载）
    """
    
    def __init__(self, server_id=0):
        self.server_id = server_id
        self.cpu_frequency = 20.0  # 20 GHz
        
        # 能耗参数（极低）
        self.alpha_cs = 1e-27  # J/cycle
        
        # 并行处理能力
        self.parallel_factor = 8.0   # 强大的并行处理能力
        
    def reset(self):
        """重置服务器状态（云服务器无需重置）"""
        pass
        
    def calculate_execution_time(self, cpu_cycles):
        """
        计算云服务器执行时间
        云服务器具有强大的并行处理能力，任务可以立即执行
        """
        cpu_frequency_hz = self.cpu_frequency * 1e9  # GHz转Hz
        # 考虑并行处理能力
        effective_frequency = cpu_frequency_hz * self.parallel_factor
        execution_time = cpu_cycles / effective_frequency
        return execution_time
    
    def calculate_energy_consumption(self, cpu_cycles):
        """计算云服务器能耗（极低）"""
        energy = self.alpha_cs * cpu_cycles
        return energy
    
    def get_expected_completion_time(self, cpu_cycles):
        """
        获取新任务的预期完成时间
        云服务器资源无限，无等待时间
        """
        return self.calculate_execution_time(cpu_cycles)
    
    def get_state(self):
        """
        返回简化的服务器状态
        返回: [CPU频率(归一化)]
        """
        return [
            self.cpu_frequency / 20.0  # CPU频率归一化（最大20GHz）
        ]


# 兼容性保持
Device = UserEquipment  # 别名，保持向后兼容