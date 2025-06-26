# environment/device_models.py
import numpy as np
import random


class UserEquipment:
    """端侧设备（User Equipment, UE）"""
    
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
        
        # 网络参数
        self.transmission_rate_to_edge = 1e9  # 1 Gbps = 1e9 bps
        
        # 动态状态
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        self.battery_capacity = 4000.0  # mAh (更现实的电池容量)
        self.battery = self.battery_capacity
        
    def reset(self):
        """重置设备状态"""
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        self.battery = self.battery_capacity
        
    def calculate_execution_time(self, cpu_cycles):
        """
        计算执行时间
        
        公式: t = C / f
        其中:
        - C: CPU周期数 (cycles)
        - f: CPU频率 (Hz)
        - t: 执行时间 (秒)
        """
        # if cpu_cycles > self.max_cpu_cycles:
        #     return float('inf')  # 超出处理能力
            
        cpu_frequency_hz = self.cpu_frequency * 1e9  # GHz转Hz
        execution_time = cpu_cycles / cpu_frequency_hz
        return execution_time
    
    def calculate_energy_consumption(self, cpu_cycles):
        """
        计算计算能耗
        
        公式: e = α * C
        其中:
        - α: 能耗系数 (J/cycle) 
        - C: CPU周期数 (cycles)
        """
        energy = self.alpha_ue * cpu_cycles
        return energy
    
    def calculate_transmission_time(self, data_size_mb):
        """
        计算传输时间（UE到边缘服务器）
        
        参数:
        - data_size_mb: 数据大小 (MB)
        
        返回: 传输时间 (秒)
        """
        data_size_bits = data_size_mb * 8 * 1e6  # MB转bits
        transmission_time = data_size_bits / self.transmission_rate_to_edge
        return transmission_time
    
    def calculate_transmission_energy(self, transmission_time):
        """
        计算传输能耗
        
        公式: e = P * t
        其中:
        - P: 传输功率 (W)
        - t: 传输时间 (s)
        """
        energy = self.transmission_power * transmission_time
        return energy
    
    def execute(self, cpu_cycles):
        """执行计算任务，返回执行时延"""
        execution_time = self.calculate_execution_time(cpu_cycles)
        
        # 更新CPU负载
        load_increment = execution_time / 10.0  # 假设10秒为满负载基准
        self.current_cpu_load = min(self.current_cpu_load + load_increment, 1.0)
        
        return execution_time
    
    def consume_energy(self, cpu_cycles):
        """计算并消耗能量"""
        energy = self.calculate_energy_consumption(cpu_cycles)
        # 这里可以更新电池状态
        return energy
    
    def get_state(self):
        """返回设备状态"""
        return [
            self.cpu_frequency / 1.0,  # 归一化CPU频率 (最大1.0GHz)
            self.current_cpu_load,     # CPU负载 (0-1)
            self.battery / self.battery_capacity,  # 电池比例 (0-1)
            self.current_memory_usage  # 内存使用率 (0-1)
        ]


class EdgeServer:
    """边缘服务器（Edge Server, ES）"""
    
    def __init__(self, server_id, cpu_frequency):
        self.server_id = server_id
        self.cpu_frequency = cpu_frequency  # GHz，从{5, 6, 8, 10, 12}中选择
        
        # 能耗参数
        self.alpha_es = 3e-26  # J/cycle
        
        # 网络参数
        self.transmission_rate_to_cloud = 10e9  # 10 Gbps (边缘到云)
        
        # 动态状态
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        self.memory_capacity = 16.0  # GB
        
    def reset(self):
        """重置服务器状态"""
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        
    def calculate_execution_time(self, cpu_cycles):
        """
        计算边缘服务器执行时间
        
        公式: t = C / f
        """
        cpu_frequency_hz = self.cpu_frequency * 1e9  # GHz转Hz
        execution_time = cpu_cycles / cpu_frequency_hz
        return execution_time
    
    def calculate_energy_consumption(self, cpu_cycles):
        """计算边缘服务器能耗"""
        energy = self.alpha_es * cpu_cycles
        return energy
    
    def calculate_transmission_time_to_cloud(self, data_size_mb):
        """计算边缘到云的传输时间"""
        data_size_bits = data_size_mb * 8 * 1e6  # MB转bits
        transmission_time = data_size_bits / self.transmission_rate_to_cloud
        return transmission_time
    
    def execute(self, cpu_cycles):
        """执行计算任务"""
        execution_time = self.calculate_execution_time(cpu_cycles)
        
        # 更新CPU负载
        load_increment = execution_time / 20.0  # 20秒为满负载基准
        self.current_cpu_load = min(self.current_cpu_load + load_increment, 1.0)
        
        return execution_time
    
    def get_state(self):
        """返回服务器状态"""
        return [
            self.cpu_frequency / 12.0,  # 归一化CPU频率 (最大12GHz)
            self.current_cpu_load,      # CPU负载 (0-1)
            self.current_memory_usage / self.memory_capacity  # 内存使用率 (0-1)
        ]


class CloudServer:
    """云服务器（Cloud Server, CS）"""
    
    def __init__(self, server_id=0):
        self.server_id = server_id
        self.cpu_frequency = 20.0  # 20 GHz
        
        # 能耗参数（简化为极低，体现规模效应）
        self.alpha_cs = 1e-27  # J/cycle
        
        # 云服务器参数
        self.memory_capacity = 64.0  # GB
        self.parallel_factor = 4.0   # 并行处理能力
        
        # 动态状态
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        
    def reset(self):
        """重置服务器状态"""
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        
    def calculate_execution_time(self, cpu_cycles):
        """
        计算云服务器执行时间（考虑并行处理能力）
        
        云服务器视为无计算资源瓶颈，支持并行处理
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
    
    def execute(self, cpu_cycles):
        """执行计算任务"""
        execution_time = self.calculate_execution_time(cpu_cycles)
        
        # 更新CPU负载（云端有很强的并行能力）
        load_increment = execution_time / 50.0  # 50秒为满负载基准
        self.current_cpu_load = min(self.current_cpu_load + load_increment, 1.0)
        
        return execution_time
    
    def get_state(self):
        """返回服务器状态"""
        return [
            self.cpu_frequency / 20.0,  # 归一化CPU频率 (最大20GHz)
            self.current_cpu_load,      # CPU负载 (0-1)
            self.current_memory_usage / self.memory_capacity  # 内存使用率 (0-1)
        ]


# 兼容性保持：保留原有的Device、EdgeServer、CloudServer类名
Device = UserEquipment  # 别名，保持向后兼容