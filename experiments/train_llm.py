import numpy as np
import os
import random
import torch
import gymnasium as gym
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from llm_assistant.llm_client import LLMClient
from llm_assistant.response_parser import ResponseParser
from utils.plotting import Plotter
from utils.metrics import MetricsTracker
from utils.config import load_config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_llm(config=None):
    # 加载配置
    if config is None:
        config = load_config()
    set_seed(config.get('seed', 42))

    # 创建保存目录
    save_dir = config.get('save_dir', 'results_llm')
    os.makedirs(save_dir, exist_ok=True)

    # 创建环境
    env = CloudEdgeDeviceEnv(config['environment'])

    # 创建LLM客户端
    llm_client = LLMClient(
        api_key=config['llm'].get('api_key', ''),
        model_name=config['llm']['model_name'],
        server_url=config['llm'].get('server_url', 'http://10.200.1.35:8888/v1/completions'),
        timeout_connect=config['llm'].get('timeout_connect', 120),
        timeout_read=config['llm'].get('timeout_read', 300),
        use_mock=config['llm'].get('use_mock_when_unavailable', True)
    )

    # 训练参数
    max_episodes = config['maddpg']['max_episodes']
    max_steps = config['maddpg']['max_steps']
    num_agents = env.num_devices

    plotter = Plotter(save_dir)
    metrics_tracker = MetricsTracker()
    all_actions = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0
        for step in range(max_steps):
            # 获取LLM策略并执行
            device_info = [{"cpu": d.cpu_capacity, "memory": d.memory_capacity} for d in env.devices]
            edge_info = [{"cpu": e.cpu_capacity, "memory": e.memory_capacity} for e in env.edge_servers]
            cloud_info = [{"cpu": c.cpu_capacity, "memory": c.memory_capacity} for c in env.cloud_servers]
            llm_strategies = llm_client.get_unload_strategy(state, device_info, edge_info, cloud_info)
            
            # 解析LLM响应
            actions = ResponseParser.parse_unload_strategy(
                llm_strategies,
                env.num_devices,
                env.num_edges,
                env.num_clouds
            )
            
            # 将策略转换为环境可接受的动作格式
            action_list = []
            for device_idx in range(env.num_devices):
                device_action = next((a for a in actions if a["task_id"] == device_idx), 
                                    {"offload_ratio": 0.0, "target_node": 0})
                action_list.append([
                    device_action["offload_ratio"],
                    device_action["target_node"]
                ])
            actions = np.array(action_list)
            all_actions.append(actions)
            # 执行动作
            next_state, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            state = next_state
            episode_reward += sum(rewards)
            # 可选：统计延迟、能耗等
            # episode_delay += ...
            # episode_energy += ...
            if done:
                break
        metrics_tracker.add_episode(episode_reward, episode_delay, episode_energy, None)
        if (episode + 1) % 10 == 0:
            avg_metrics = metrics_tracker.get_average_metrics(last_n=10)
            print(f"[LLM] Episode {episode + 1}/{max_episodes}")
            print(f"  平均奖励: {avg_metrics['avg_reward']:.2f}")
        if (episode + 1) % 100 == 0:
            plotter.plot_rewards(metrics_tracker.episode_rewards)
            if all_actions:
                plotter.plot_action_distribution(np.array(all_actions).reshape(-1, 2))
    plotter.plot_rewards(metrics_tracker.episode_rewards)
    if all_actions:
        plotter.plot_action_distribution(np.array(all_actions).reshape(-1, 2))
    print("[LLM] 训练完成!")
    return metrics_tracker.episode_rewards 