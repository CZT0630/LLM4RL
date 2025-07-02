import numpy as np
import torch
import os
import random
import gymnasium as gym
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from algos.maddpg_agent import MADDPGAgent
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

def train_maddpg(config=None):
    # 加载配置
    if config is None:
        config = load_config()
    set_seed(config.get('seed', 42))

    # 创建保存目录
    save_dir = config.get('save_dir', 'results_maddpg')
    os.makedirs(save_dir, exist_ok=True)

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
    plotter = Plotter(save_dir)
    metrics_tracker = MetricsTracker()

    # 训练参数
    max_episodes = config['maddpg']['max_episodes']
    max_steps = config['maddpg']['max_steps']

    all_actions = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0
        for step in range(max_steps):
            # 选择动作（使用正确的Agent状态提取）
            actions = []
            for i, agent in enumerate(agents):
                agent_state = env.extract_agent_state(state, i)
                action = agent.select_action(agent_state, llm_advice=None)
                actions.append(action)
            
            all_actions.append(actions)
            
            # 执行动作
            next_state, rewards, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            
            # 存储经验（使用正确的状态提取）
            for i, agent in enumerate(agents):
                agent_state = env.extract_agent_state(state, i)
                agent_next_state = env.extract_agent_state(next_state, i)
                agent.replay_buffer.add(agent_state, actions, rewards, agent_next_state, done)
            
            # 训练智能体
            if len(agents[0].replay_buffer) > config['maddpg']['batch_size']:
                experiences = agents[0].replay_buffer.sample(config['maddpg']['batch_size'])
                for agent in agents:
                    agent.train(experiences, agents)
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
            print(f"[MADDPG] Episode {episode + 1}/{max_episodes}")
            print(f"  平均奖励: {avg_metrics['avg_reward']:.2f}")
        if (episode + 1) % 100 == 0:
            for i, agent in enumerate(agents):
                torch.save(agent.actor.state_dict(), f"{save_dir}/actor_agent_{i}_episode_{episode + 1}.pth")
                torch.save(agent.critic.state_dict(), f"{save_dir}/critic_agent_{i}_episode_{episode + 1}.pth")
            plotter.plot_rewards(metrics_tracker.episode_rewards)
            if all_actions:
                plotter.plot_action_distribution(np.array(all_actions).reshape(-1, 2))
    # 保存最终模型
    for i, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(), f"{save_dir}/actor_agent_{i}_final.pth")
        torch.save(agent.critic.state_dict(), f"{save_dir}/critic_agent_{i}_final.pth")
    plotter.plot_rewards(metrics_tracker.episode_rewards)
    if all_actions:
        plotter.plot_action_distribution(np.array(all_actions).reshape(-1, 2))
    print("[MADDPG] 训练完成!")
    return metrics_tracker.episode_rewards
