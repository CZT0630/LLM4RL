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
from utils.path_manager import get_path_manager
from utils.csv_saver import save_training_metrics_csv

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

    # 🆕 使用路径管理器
    path_manager = get_path_manager()
    
    # 创建保存目录
    model_dir = path_manager.get_model_path("llm")
    data_dir = path_manager.get_data_path("csv")
    json_dir = path_manager.get_data_path("json")
    plot_dir = path_manager.get_plot_path()
    log_dir = path_manager.get_log_path()
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 创建环境
    env = CloudEdgeDeviceEnv(config)

    # 创建LLM客户端
    llm_client = LLMClient(
        config=config,
        use_mock=config['llm'].get('use_mock_when_unavailable', True)
    )

    # 训练参数 - 优先使用命令行指定的episodes，否则使用配置文件中的值
    max_episodes = config.get('training', {}).get('episodes', config['maddpg']['max_episodes'])
    max_steps = config['maddpg']['max_steps']
    num_agents = env.num_devices

    plotter = Plotter(plot_dir)
    metrics_tracker = MetricsTracker()
    all_actions = []
    episode_completion_rates = []  # 新增：记录每个episode的任务完成率

    print(f"🔧 [LLM] 训练配置:")
    print(f"  训练轮数: {max_episodes}")
    print(f"  每轮最大步数: {max_steps}")
    print(f"  设备数量: {num_agents}")
    print(f"  结果保存路径: {path_manager.get_experiment_dir()}")

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_latencies = []  # 🆕 收集每步的延迟
        episode_energies = []   # 🆕 收集每步的能耗
        step_means = []  # 新增：收集每个step所有智能体reward的均值
        
        for step in range(max_steps):
            # 获取LLM策略并执行
            device_info = [{
                "device_id": d.device_id,
                "cpu_frequency": d.cpu_frequency,
                "battery_percentage": d.get_battery_percentage(),
                "task_load": d.calculate_task_load()
            } for d in env.devices]
            edge_info = [{
                "server_id": e.server_id if hasattr(e, 'server_id') else i,
                "cpu_frequency": e.cpu_frequency,
                "task_load": e.calculate_task_load() if hasattr(e, 'calculate_task_load') else 0.0
            } for i, e in enumerate(env.edge_servers_list)]
            cloud_info = [{
                "server_id": c.server_id if hasattr(c, 'server_id') else i,
                "cpu_frequency": c.cpu_frequency
            } for i, c in enumerate(env.cloud_servers_list)]
            
            # 获取格式1的LLM策略
            llm_strategies = llm_client.get_unload_strategy(state, device_info, edge_info, cloud_info)
            
            # 解析LLM响应（格式1）
            parsed_strategies = ResponseParser.parse_unload_strategy(
                llm_strategies,
                env.num_devices,
                env.num_edges,
                env.num_clouds
            )
            
            # 将格式1策略转换为环境可接受的动作格式 [α1, α2, α3, edge_id]
            action_list = []
            for device_idx in range(env.num_devices):
                device_strategy = next(
                    (s for s in parsed_strategies if s["device_id"] == device_idx), 
                    {
                        "device_id": device_idx,
                        "local_ratio": 1.0,
                        "edge_ratio": 0.0,
                        "cloud_ratio": 0.0,
                        "target_edge": 0
                    }
                )
                
                # 转换为环境动作格式
                action = [
                    device_strategy["local_ratio"],   # α1: 本地执行比例
                    device_strategy["edge_ratio"],    # α2: 边缘执行比例  
                    device_strategy["cloud_ratio"],   # α3: 云端执行比例
                    device_strategy["target_edge"]    # edge_id: 目标边缘服务器
                ]
                action_list.append(action)
                
            actions = np.array(action_list)
            all_actions.append(actions)
            
            # 执行动作
            next_state, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            state = next_state
            
            # 记录本step所有智能体reward的均值（只考虑有任务的设备）
            if info and 'has_task_list' in info:
                valid_rewards = [r for r, has_task in zip(rewards, info['has_task_list']) if has_task]
                if valid_rewards:
                    step_means.append(np.mean(valid_rewards))
            else:
                # 如果没有has_task_list，则过滤掉0奖励（假设0奖励表示无任务）
                valid_rewards = [r for r in rewards if r > 0]
                if valid_rewards:
                    step_means.append(np.mean(valid_rewards))
                else:
                    step_means.append(np.mean(rewards))  # 如果所有奖励都为0，则使用全部均值
            
            # 🆕 从info中提取延迟和能耗，过滤零值
            if info:
                step_latencies = info.get('total_latencies', [])
                step_energies = info.get('total_energies', [])
                has_task_list = info.get('has_task_list', [True] * len(step_latencies))  # 默认所有设备都有任务
                
                # 只保留有任务的设备的延迟和能耗数据
                valid_latencies = [lat for lat, has_task in zip(step_latencies, has_task_list) if has_task and lat > 0]
                valid_energies = [eng for eng, has_task in zip(step_energies, has_task_list) if has_task and eng > 0]
                
                if valid_latencies:
                    episode_latencies.extend(valid_latencies)
                if valid_energies:
                    episode_energies.extend(valid_energies)
            
            if done:
                break
        
        # 统一episode reward计算方式
        episode_reward = np.mean(step_means) if step_means else 0.0
        
        # Episode 结束，统计指标
        # 使用实际任务完成率而不是固定值
        completion_stats = env.get_task_completion_rate()
        episode_completion_rate = completion_stats.get('on_time_completion_rate', 0.0)
        
        # 记录本轮的任务完成率
        episode_completion_rates.append(episode_completion_rate)
        
        # 计算平均延迟和能耗
        avg_latency = np.mean(episode_latencies) if episode_latencies else 0.0
        avg_energy = np.mean(episode_energies) if episode_energies else 0.0
        
        # 🆕 正确传入延迟和能耗指标
        metrics_tracker.add_episode(episode_reward, avg_latency, avg_energy, True)
        
        if (episode + 1) % 10 == 0:
            avg_metrics = metrics_tracker.get_average_metrics(last_n=10)
            print(f"[LLM] Episode {episode + 1}/{max_episodes}")
            print(f"  平均奖励: {avg_metrics['avg_reward']:.2f}")
            print(f"  平均延迟: {avg_metrics['avg_delay']:.2f}s")
            print(f"  平均能耗: {avg_metrics['avg_energy']:.2f}J")
            
        if (episode + 1) % 100 == 0:
            plotter.plot_rewards(metrics_tracker.episode_rewards)
            if all_actions:
                # 绘制动作分布（4维动作）
                plotter.plot_action_distribution(np.array(all_actions).reshape(-1, 4))
                
    plotter.plot_rewards(metrics_tracker.episode_rewards)
    if all_actions:
        plotter.plot_action_distribution(np.array(all_actions).reshape(-1, 4))
        
    # 🆕 保存核心指标到CSV表格 - 使用正确的路径
    print("保存训练指标到CSV表格...")
    try:
        csv_filepath = save_training_metrics_csv(
            episode_rewards=metrics_tracker.episode_rewards,
            episode_latencies=metrics_tracker.episode_delays,
            episode_energies=metrics_tracker.episode_energy,
            episode_completion_rates=episode_completion_rates,  # 使用动态记录的完成率列表
            algorithm_name="Pure_LLM",
            save_dir=data_dir  # 🆕 使用路径管理器的目录
        )
        print(f"✅ CSV文件已保存: {csv_filepath}")
    except Exception as e:
        print(f"❌ 保存CSV文件失败: {e}")
        
    print("[LLM] 训练完成!")
    print(f"📁 结果保存在: {path_manager.get_experiment_dir()}")
    
    # 返回完整结果
    return {
        'episode_rewards': metrics_tracker.episode_rewards,
        'episode_latencies': metrics_tracker.episode_delays,
        'episode_energies': metrics_tracker.episode_energy,
        'episode_completion_rates': episode_completion_rates,  # 使用动态记录的完成率列表
        'training_losses': [],  # LLM没有训练损失
        'global_step_count': max_episodes * max_steps
    }

if __name__ == "__main__":
    train_llm() 