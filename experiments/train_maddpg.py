import numpy as np
import os
import random
import torch
import gymnasium as gym
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from algos.maddpg_agent import MADDPGAgent
from algos.replay_buffer import ReplayBuffer
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

def train_maddpg(config=None):
    # 加载配置
    if config is None:
        config = load_config()
    set_seed(config.get('seed', 42))

    # 使用路径管理器
    path_manager = get_path_manager()
    
    # 创建保存目录 - 使用正确的算法名称
    model_dir = path_manager.get_model_path("maddpg")
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
    env = CloudEdgeDeviceEnv(config['environment'])

    # 创建MADDPG智能体
    # 使用正确的单个Agent状态维度
    state_dim = env.get_agent_state_dim()  # 20维：3(自己UE) + 10(所有ES) + 1(CS) + 6(自己任务)
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[1] + 1  # 目标节点数量
    num_agents = env.num_devices

    print(f"🔧 [MADDPG] Agent配置信息:")
    print(f"  单个Agent状态维度: {state_dim}")
    print(f"  全局状态维度: {env.observation_space.shape[0]}")
    print(f"  动作维度: {action_dim}")
    print(f"  设备数量: {num_agents}")
    print(f"  模型保存路径: {model_dir}")

    # 创建共享经验回放缓冲区
    buffer_size = config['maddpg']['buffer_size']
    shared_buffer = ReplayBuffer(buffer_size)
    print(f"  共享缓冲区大小: {buffer_size}")

    agents = []
    for i in range(num_agents):
        agent = MADDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            num_agents=num_agents,
            agent_idx=i,
            lr_actor=config['maddpg']['lr_actor'],
            lr_critic=config['maddpg']['lr_critic'],
            gamma=config['maddpg']['gamma'],
            tau=config['maddpg']['tau'],
            buffer_size=config['maddpg']['buffer_size'],
            batch_size=config['maddpg']['batch_size']
        )
        agents.append(agent)

    # 创建绘图器和指标追踪器
    plotter = Plotter(plot_dir)
    metrics_tracker = MetricsTracker()

    # 训练参数
    max_episodes = config['maddpg']['max_episodes']
    max_steps = config['maddpg']['max_steps']
    train_frequency = config['maddpg']['train_frequency']
    
    # 读取训练策略参数
    save_frequency = config['training']['save_frequency']
    log_frequency = config['training']['log_frequency']
    warm_up_episodes = config['training']['warm_up_episodes']
    
    global_step_count = 0  # 全局步数计数器
    
    print(f"🔧 [MADDPG] 训练策略:")
    print(f"  训练轮数: {max_episodes}")
    print(f"  每轮最大步数: {max_steps}")
    print(f"  训练频率: 每{train_frequency}步训练一次")
    print(f"  批次大小: {config['maddpg']['batch_size']}")
    print(f"  预热轮数: {warm_up_episodes}")

    all_actions = []
    training_losses = []  # 记录训练损失
    convergence_rewards = []  # 收敛监控

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_latencies = []  # 🆕 收集每步的延迟
        episode_energies = []   # 🆕 收集每步的能耗
        
        # 判断是否在预热阶段
        is_warm_up = episode < warm_up_episodes
        
        for step in range(max_steps):
            global_step_count += 1
            
            # 选择动作（使用正确的Agent状态提取）
            actions = []
            for i, agent in enumerate(agents):
                agent_state = env.extract_agent_state(state, i)
                # 优化探索策略：预热期和训练早期增加探索
                if is_warm_up:
                    add_noise = True  # 预热期始终探索
                else:
                    add_noise = episode < max_episodes * 0.7  # 训练期前70%探索
                action = agent.select_action(agent_state, add_noise=add_noise, llm_advice=None)
                actions.append(action)
            
            # 将actions转换为numpy数组以满足环境要求
            actions = np.array(actions)
            all_actions.append(actions)
            
            # 执行动作
            next_state, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # 🆕 从info中提取延迟和能耗，过滤零值
            if info:
                step_latencies = info.get('total_latencies', [])
                step_energies = info.get('total_energies', [])
                
                # 过滤掉零值，只保留有效的任务处理数据
                valid_latencies = [lat for lat in step_latencies if lat > 0]
                valid_energies = [eng for eng in step_energies if eng > 0]
                
                if valid_latencies:
                    episode_latencies.extend(valid_latencies)
                if valid_energies:
                    episode_energies.extend(valid_energies)
            
            # 存储经验到共享缓冲区（所有Agent的经验混合存储）
            for i in range(num_agents):
                agent_state = env.extract_agent_state(state, i)
                agent_next_state = env.extract_agent_state(next_state, i)
                
                # 存储到共享缓冲区，使用标准5元组格式
                shared_buffer.add(
                    state=agent_state,
                    action=actions[i],
                    reward=rewards[i],
                    next_state=agent_next_state,
                    done=done,
                    llm_action=None  # 纯MADDPG无LLM专家动作
                )
            
            # 训练条件：非预热期 + 达到训练频率 + 缓冲区充足
            should_train = (not is_warm_up and 
                          global_step_count % train_frequency == 0 and 
                          len(shared_buffer) > config['maddpg']['batch_size'])
            
            if should_train:
                # 从共享缓冲区采样经验
                states, actions_batch, rewards_batch, next_states, dones, _ = shared_buffer.sample(config['maddpg']['batch_size'])
                
                # 所有Agent使用相同的采样经验进行训练
                step_losses = []
                for agent in agents:
                    agent_losses = agent.train(agents, shared_buffer)
                    step_losses.append(agent_losses)
                
                # 记录平均训练损失
                if step_losses and all(losses for losses in step_losses):
                    avg_critic_loss = np.mean([losses.get('critic_loss', 0) for losses in step_losses])
                    avg_actor_loss = np.mean([losses.get('actor_loss', 0) for losses in step_losses])
                    training_losses.append({
                        'step': global_step_count,
                        'episode': episode,
                        'critic_loss': avg_critic_loss,
                        'actor_loss': avg_actor_loss
                    })
                
                # 优化日志输出频率
                if (global_step_count // train_frequency) % 20 == 0:  # 每1000步打印一次训练信息
                    print(f"  Step {global_step_count}: 共享缓冲区大小={len(shared_buffer)}, "
                          f"Critic损失={avg_critic_loss:.4f}, Actor损失={avg_actor_loss:.4f}")
            
            state = next_state
            episode_reward += sum(rewards)
            
            if done:
                break
        
        # Episode 结束，统计指标
        # 使用实际任务完成率而不是固定值
        completion_stats = env.get_task_completion_rate()
        episode_completion_rate = completion_stats.get('on_time_completion_rate', 0.0)
        
        # 计算平均延迟和能耗
        avg_latency = np.mean(episode_latencies) if episode_latencies else 0.0
        avg_energy = np.mean(episode_energies) if episode_energies else 0.0
        
        # 🆕 正确传入延迟和能耗指标
        metrics_tracker.add_episode(episode_reward, avg_latency, avg_energy, False)
        convergence_rewards.append(episode_reward)
        
        # 优化进度打印
        if (episode + 1) % log_frequency == 0:
            avg_metrics = metrics_tracker.get_average_metrics(last_n=log_frequency)
            status = "预热阶段" if is_warm_up else "训练阶段"
            print(f"[MADDPG] Episode {episode + 1}/{max_episodes} ({status})")
            print(f"  平均奖励: {avg_metrics['avg_reward']:.2f}")
            print(f"  缓冲区大小: {len(shared_buffer)}")
        
        # 定期保存模型
        if (episode + 1) % save_frequency == 0:
            for i, agent in enumerate(agents):
                # 保存为分离格式 (actor和critic分别保存)
                actor_path = path_manager.get_model_file_path("maddpg", f"actor_agent_{i}_episode_{episode+1}.pth")
                critic_path = path_manager.get_model_file_path("maddpg", f"critic_agent_{i}_episode_{episode+1}.pth")
                torch.save(agent.actor.state_dict(), actor_path)
                torch.save(agent.critic.state_dict(), critic_path)
            print(f"  ✅ 模型已保存至 Episode {episode + 1}")

    # 训练完成，保存最终模型
    print("\n🎯 [MADDPG] 训练完成，保存最终模型...")
    for i, agent in enumerate(agents):
        # 保存最终模型 (分离格式)
        actor_path = path_manager.get_model_file_path("maddpg", f"actor_agent_{i}_final.pth")
        critic_path = path_manager.get_model_file_path("maddpg", f"critic_agent_{i}_final.pth")
        torch.save(agent.actor.state_dict(), actor_path)
        torch.save(agent.critic.state_dict(), critic_path)
        print(f"  ✅ Agent {i}: {actor_path}")
        print(f"  ✅ Agent {i}: {critic_path}")

    # 保存训练指标到CSV
    print("\n📊 保存训练指标...")
    
    # 计算所有episode的平均完成率作为代表值
    # 这里简化处理，在实际环境中应该单独记录每个episode的完成率
    final_completion_stats = env.get_task_completion_rate()
    avg_completion_rate = final_completion_stats.get('on_time_completion_rate', 0.0)
    
    try:
        csv_file = save_training_metrics_csv(
            episode_rewards=metrics_tracker.episode_rewards,
            episode_latencies=metrics_tracker.episode_delays,
            episode_energies=metrics_tracker.episode_energy,
            episode_completion_rates=[avg_completion_rate] * len(metrics_tracker.episode_rewards),
            algorithm_name="MADDPG",
            save_dir=data_dir
        )
        print(f"  ✅ CSV文件: {csv_file}")
    except Exception as e:
        print(f"  ❌ CSV保存失败: {e}")

    # 保存详细训练数据到JSON
    try:
        training_data = {
            'algorithm': 'MADDPG',
            'config': config['maddpg'],
            'training_losses': training_losses,
            'convergence_rewards': convergence_rewards,
            'final_metrics': metrics_tracker.get_average_metrics(),
            'total_steps': global_step_count
        }
        json_file = path_manager.get_data_file_path("json", "maddpg_training_stats.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(training_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"  ✅ JSON文件: {json_file}")
    except Exception as e:
        print(f"  ❌ JSON保存失败: {e}")

    # 绘制训练曲线
    try:
        plotter.plot_rewards(metrics_tracker.episode_rewards)
        if all_actions:
            plotter.plot_action_distribution(np.array(all_actions).reshape(-1, action_dim))
        print(f"  ✅ 图表保存至: {plot_dir}")
    except Exception as e:
        print(f"  ❌ 图表保存失败: {e}")

    print(f"\n🎉 [MADDPG] 训练完成!")
    print(f"📁 结果保存在: {path_manager.get_experiment_dir()}")
    
    # 返回训练结果
    return {
        'episode_rewards': metrics_tracker.episode_rewards,
        'episode_latencies': metrics_tracker.episode_delays,
        'episode_energies': metrics_tracker.episode_energy,
        'episode_completion_rates': [avg_completion_rate] * len(metrics_tracker.episode_rewards),
        'training_losses': training_losses,
        'global_step_count': global_step_count,
        'algorithm': 'MADDPG',
        'model_paths': {
            'actor': [path_manager.get_model_file_path("maddpg", f"actor_agent_{i}_final.pth") for i in range(num_agents)],
            'critic': [path_manager.get_model_file_path("maddpg", f"critic_agent_{i}_final.pth") for i in range(num_agents)]
        }
    }

if __name__ == "__main__":
    train_maddpg()
