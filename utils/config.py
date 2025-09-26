# utils/config.py
import yaml

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # 如果没有seed则加上默认seed
        if 'seed' not in config:
            config['seed'] = 42
        return config
    except FileNotFoundError:
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return get_default_config()

def get_default_config():
    """返回默认配置"""
    return {
        "environment": {
            "num_devices": 5,
            "num_edges": 2,
            "num_clouds": 1,
            "device_config": {
                "cpu_capacity": 2.0,
                "memory_capacity": 4.0
            },
            "edge_config": {
                "cpu_capacity": 8.0,
                "memory_capacity": 16.0
            },
            "cloud_config": {
                "cpu_capacity": 32.0,
                "memory_capacity": 64.0
            },
            "task_config": {
                "types": ["computation_intensive", "data_intensive", "balanced"],
                "min_computation": 100,
                "max_computation": 1000,
                "min_data_size": 1,
                "max_data_size": 100,
                "min_deadline": 10,
                "max_deadline": 60
            }
        },
        "llm": {
            "api_key": "",
            "model_name": "qwen3-14b",
            "query_frequency": 1  # 每多少个episode查询一次LLM
        },
        "maddpg": {
            "lr_actor": 1e-4,
            "lr_critic": 1e-3,
            "gamma": 0.99,
            "tau": 0.001,
            "buffer_size": int(1e6),
            "batch_size": 100,
            "max_episodes": 1000,
            "max_steps": 200
        },
        "seed": 42
    }