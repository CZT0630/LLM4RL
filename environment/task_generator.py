# environment/task_generator.py
import numpy as np
import random
import time


class TaskGenerator:
    """任务生成器 - 根据实验设置生成不同类型的任务"""
    
    def __init__(self, config=None):
        # 任务数据大小范围 (MB)
        self.task_sizes = {
            'small': (1, 5),      # 小任务：1-5 MB (传感器数据、文本处理)
            'medium': (10, 50),   # 中任务：10-50 MB (短视频预处理、轻度图像识别)
            'large': (100, 200)   # 大任务：100-200 MB (高清视频分析、复杂AI推理)
        }
        
        # 处理密度：0.2 Gcycles/MB
        self.processing_density = 0.2e9  # cycles/MB (0.2 * 10^9 cycles/MB)
        
        # 任务类型分布权重
        self.task_type_weights = {
            'small': 0.5,   # 50%小任务
            'medium': 0.4,  # 40%中任务  
            'large': 0.1    # 10%大任务
        }
        
        # 截止时间参数（基于任务大小动态设置）
        self.deadline_factors = {
            'small': (2.0, 5.0),    # 小任务：2-5倍本地执行时间
            'medium': (1.5, 3.0),   # 中任务：1.5-3倍本地执行时间
            'large': (1.2, 2.0)     # 大任务：1.2-2倍本地执行时间
        }
        
        # 新增泊松到达过程配置
        self.poisson_config = {
            'base_arrival_rate': 0.5,  # 基础到达率（每步每设备的平均任务数）
            'time_pattern_enabled': False,  # 默认不启用时间模式
            'pattern_cycle': 20,     # 时间模式周期（步数）
            'peak_hours': [5, 15],   # 高峰时段（在周期内的位置）
            'peak_multiplier': 2.0,  # 高峰期到达率倍数
            'low_multiplier': 0.5,   # 低峰期到达率倍数
        }
        
        # 初始化当前状态
        self.current_step = 0
        self.current_pattern_phase = 'normal'  # 'normal', 'peak', 'low'
        
        # 如果提供了配置，更新默认值
        if config:
            if 'task_types' in config:
                for task_type, properties in config['task_types'].items():
                    if 'data_range' in properties and task_type in self.task_sizes:
                        self.task_sizes[task_type] = tuple(properties['data_range'])
                    if 'probability' in properties and task_type in self.task_type_weights:
                        self.task_type_weights[task_type] = properties['probability']
                    if 'deadline_multiplier' in properties and task_type in self.deadline_factors:
                        min_factor = properties['deadline_multiplier'] * 0.8
                        max_factor = properties['deadline_multiplier'] * 1.2
                        self.deadline_factors[task_type] = (min_factor, max_factor)
            
            if 'processing_density' in config:
                self.processing_density = config['processing_density']
            
            # 更新泊松配置
            if 'poisson_config' in config:
                for key, value in config['poisson_config'].items():
                    if key in self.poisson_config:
                        self.poisson_config[key] = value

    def generate_single_task(self, task_id=0, device_id=0):
        """生成单个任务"""
        # 选择任务类型
        task_type = np.random.choice(
            list(self.task_type_weights.keys()),
            p=list(self.task_type_weights.values())
        )
        
        # 确保task_type是普通字符串，不是numpy字符串
        task_type = str(task_type)
        
        # 生成数据大小
        min_size, max_size = self.task_sizes[task_type]
        data_size = random.uniform(min_size, max_size)
        
        # 计算CPU周期需求：C = ε * D
        cpu_cycles = data_size * self.processing_density
        
        # 生成截止时间（基于最慢设备的本地执行时间）
        # 假设最慢设备CPU频率为0.5GHz
        slowest_cpu_frequency = 0.5e9  # Hz
        local_execution_time = cpu_cycles / slowest_cpu_frequency
        
        # 根据任务类型设置截止时间因子
        min_factor, max_factor = self.deadline_factors[task_type]
        deadline_factor = random.uniform(min_factor, max_factor)
        deadline = local_execution_time * deadline_factor
        
        return {
            'task_id': f"{device_id}_{self.current_step}_{task_id}",
            'device_id': device_id,
            'type': task_type,
            'data_size': data_size,       # 数据大小 (MB)
            'cpu_cycles': cpu_cycles,     # CPU周期需求 (cycles)
            'deadline': deadline,         # 截止时间 (秒)
            'arrival_time': time.time()   # 到达时间戳
        }
    
    def generate_poisson_tasks(self, num_devices, step=None):
        """
        使用泊松分布生成任务，支持时间模式
        
        参数:
            num_devices: 设备数量
            step: 当前步数（如果不提供，则使用内部计数器）
            
        返回:
            一个字典，键为设备ID，值为该设备的任务列表
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # 计算当前时间模式倍率
        arrival_rate_multiplier = self._calculate_time_pattern_multiplier()
        
        # 计算最终到达率
        final_arrival_rate = self.poisson_config['base_arrival_rate'] * arrival_rate_multiplier
        
        # 按设备生成任务
        device_tasks = {}
        total_tasks = 0
        
        for device_id in range(num_devices):
            # 使用泊松分布确定该设备的任务数
            num_tasks = np.random.poisson(final_arrival_rate)
            
            if num_tasks > 0:
                tasks = []
                for i in range(num_tasks):
                    task = self.generate_single_task(task_id=total_tasks+i, device_id=device_id)
                    tasks.append(task)
                
                device_tasks[device_id] = tasks
                total_tasks += num_tasks
            else:
                device_tasks[device_id] = []
        
        print(f"[Step {self.current_step}] 生成任务统计: 时间模式='{self.current_pattern_phase}',"
              f" 倍率={arrival_rate_multiplier:.2f}, 总任务数={total_tasks}")
        
        return device_tasks
    
    def _calculate_time_pattern_multiplier(self):
        """计算时间模式倍率"""
        if not self.poisson_config['time_pattern_enabled']:
            return 1.0
            
        # 计算在周期中的位置
        cycle_position = self.current_step % self.poisson_config['pattern_cycle']
        
        # 判断当前是否在高峰期
        is_peak = any(abs(cycle_position - peak) <= 2 for peak in self.poisson_config['peak_hours'])
        
        if is_peak:
            self.current_pattern_phase = 'peak'
            return self.poisson_config['peak_multiplier']
        elif cycle_position < 5:  # 周期前25%为低峰期
            self.current_pattern_phase = 'low'
            return self.poisson_config['low_multiplier']
        else:
            self.current_pattern_phase = 'normal'
            return 1.0
    
    def generate_tasks(self, num_tasks):
        """生成多个任务"""
        tasks = []
        for i in range(num_tasks):
            task = self.generate_single_task(task_id=i)
            tasks.append(task)
        return tasks
    
    def generate_task_batch(self, num_batches, tasks_per_batch):
        """生成任务批次"""
        all_tasks = []
        for batch_id in range(num_batches):
            batch_tasks = []
            for task_id in range(tasks_per_batch):
                task = self.generate_single_task(task_id=batch_id * tasks_per_batch + task_id)
                batch_tasks.append(task)
            all_tasks.append(batch_tasks)
        return all_tasks
    
    def get_task_statistics(self, tasks):
        """获取任务统计信息"""
        if not tasks:
            return {}
            
        stats = {
            'total_tasks': len(tasks),
            'task_types': {},
            'data_size_stats': {},
            'cpu_cycles_stats': {},
            'deadline_stats': {}
        }
        
        # 按类型统计
        for task_type in self.task_sizes.keys():
            type_tasks = [t for t in tasks if t['type'] == task_type]
            stats['task_types'][task_type] = len(type_tasks)
        
        # 数据大小统计
        data_sizes = [t['data_size'] for t in tasks]
        stats['data_size_stats'] = {
            'min': min(data_sizes),
            'max': max(data_sizes),
            'mean': np.mean(data_sizes),
            'std': np.std(data_sizes)
        }
        
        # CPU周期统计
        cpu_cycles = [t['cpu_cycles'] for t in tasks]
        stats['cpu_cycles_stats'] = {
            'min': min(cpu_cycles),
            'max': max(cpu_cycles),
            'mean': np.mean(cpu_cycles),
            'std': np.std(cpu_cycles)
        }
        
        # 截止时间统计
        deadlines = [t['deadline'] for t in tasks]
        stats['deadline_stats'] = {
            'min': min(deadlines),
            'max': max(deadlines),
            'mean': np.mean(deadlines),
            'std': np.std(deadlines)
        }
        
        return stats


class Task:
    """任务类 - 支持任务分割"""
    
    def __init__(self, task_data):
        """从任务数据初始化任务对象"""
        self.task_id = task_data.get('task_id', 0)
        self.task_type = task_data.get('type', 'medium')
        self.task_data_size = task_data.get('data_size', task_data.get('data_size', 50))
        self.task_workload = task_data.get('cpu_cycles', task_data.get('computation', 50) * 1e6)
        self.deadline = task_data.get('deadline', 10.0)
        self.device_id = task_data.get('device_id', 0)
        self.creation_step = 0  # 会在环境中设置
        self.arrival_time = task_data.get('arrival_time', time.time())
        
        # 任务分割比例 [α1, α2, α3] 其中 α1+α2+α3=1
        self.split_ratios = None
        
    def set_split_ratios(self, alpha1, alpha2, alpha3):
        """
        设置任务分割比例
        
        参数:
        - alpha1: 端侧本地处理比例
        - alpha2: 边缘服务器处理比例  
        - alpha3: 云侧处理比例
        """
        if abs(alpha1 + alpha2 + alpha3 - 1.0) > 1e-6:
            raise ValueError("分割比例之和必须等于1")
        self.split_ratios = [alpha1, alpha2, alpha3]
        
    def get_split_workloads(self):
        """获取分割后的工作负载（CPU周期数）"""
        if self.split_ratios is None:
            raise ValueError("请先设置分割比例")
            
        workloads = [
            self.task_workload * ratio for ratio in self.split_ratios
        ]
        return workloads
    
    def get_split_data_sizes(self):
        """获取分割后的数据大小（MB）"""
        if self.split_ratios is None:
            raise ValueError("请先设置分割比例")
            
        data_sizes = [
            self.task_data_size * ratio for ratio in self.split_ratios
        ]
        return data_sizes
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'task_id': self.task_id,
            'type': self.task_type,
            'data_size_mb': self.task_data_size,
            'cpu_cycles': self.task_workload,
            'deadline': self.deadline,
            'split_ratios': self.split_ratios,
            'device_id': self.device_id,
            'creation_step': self.creation_step,
            'arrival_time': self.arrival_time
        }