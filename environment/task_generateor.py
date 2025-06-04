# environment/task_generator.py
import numpy as np


class TaskGenerator:
    def __init__(self, config):
        self.task_types = config['types']
        self.min_computation = config['min_computation']  # 计算量最小值 (MI)
        self.max_computation = config['max_computation']  # 计算量最大值 (MI)
        self.min_data_size = config['min_data_size']  # 数据量最小值 (MB)
        self.max_data_size = config['max_data_size']  # 数据量最大值 (MB)
        self.min_deadline = config['min_deadline']  # 截止时间最小值 (秒)
        self.max_deadline = config['max_deadline']  # 截止时间最大值 (秒)

    def generate_tasks(self, num_tasks):
        tasks = []
        for _ in range(num_tasks):
            task_type = np.random.choice(self.task_types)
            computation = np.random.uniform(self.min_computation, self.max_computation)
            data_size = np.random.uniform(self.min_data_size, self.max_data_size)
            deadline = np.random.uniform(self.min_deadline, self.max_deadline)

            tasks.append({
                'type': task_type,
                'computation': computation,  # 计算量 (MI)
                'data_size': data_size,  # 数据量 (MB)
                'deadline': deadline  # 截止时间 (秒)
            })

        return tasks