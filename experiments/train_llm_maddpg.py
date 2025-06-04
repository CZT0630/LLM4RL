# experiments/train_llm_maddpg.py
import numpy as np
import torch
import os
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from llm_assistant.llm_client import LLMClient
from llm_assistant.response_parser import ResponseParser
from maddpg.maddpg_agent import MADDPGAgent
from utils.plotting import Plotter
from utils.metrics import MetricsTracker
from utils.config import load_config


def train_llm_maddpg(config=None):
    # 加载配置
    if config is None:
        config = load_config()

    # 创建保存目录
    save_dir = config.get('save_dir', 'results')
    os.makedirs(save_dir, exist_ok=True)

    # 创建环境
    env = CloudEdgeDeviceEnv(config['environment'])

    # 创建LLM客户端
    llm_client = LLMClient(
        api_key=config['llm']['api_key'],
        model_name=config['llm']['model_name']
    )

    # 创建MADDPG智能体
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[1] + 1  # 目标节点数量
    num_agents = env.num_devices

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
    llm_query_freq = config['llm']['query_frequency']

    # 设备、边缘和云端详细信息
    device_info = [{"cpu": device.cpu_capacity, "memory": device.memory_capacity} for device in env.devices]
    edge_info = [{"cpu": edge.cpu_capacity, "memory": edge.memory_capacity} for edge in env.edge_servers]
    cloud_info = [{"cpu": cloud.cpu_capacity, "memory": cloud.memory_capacity} for cloud in env.cloud_servers]

    # 训练循环
    all_actions = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        episode_delay = 0
        episode_energy = 0

        # 每llm_query_freq个episode咨询一次LLM
        if episode % llm_query_freq == 0:
            print(f"Episode {episode}: 咨询LLM获取卸载策略建议...")
            strategies = llm_client.get_unload_strategy(state, device_info, edge_info, cloud_info)

            # 解析LLM响应
            llm_advice = ResponseParser.parse_unload_strategy(
                strategies,
                env.num_devices,
                env.num_edges,
                env.num_clouds
            )
        else:
            llm_advice = None

        for step in range(max_steps):
            # 选择动作
            actions = []
            for i, agent in enumerate(agents):
                # 为每个智能体提供全局状态和LLM指导
                if llm_advice:
                    agent_llm_advice = [a for a in llm_advice if a["task_id"] == i]
                else:
                    agent_llm_advice = None

                agent_action = agent.select_action(state, agent_llm_advice)
                actions.append(agent_action)

            all_actions.append(actions)

            # 执行动作
            next_state, rewards, done, _ = env.step(actions)

            # 存储经验
            for i, agent in enumerate(agents):
                agent.replay_buffer.add(
                    state, actions, rewards, next_state, done
                )

            # 训练智能体
            if len(agents[0].replay_buffer) > config['maddpg']['batch_size']:
                experiences = agents[0].replay_buffer.sample(config['maddpg']['batch_size'])
                for agent in agents:
                    agent.train(experiences, agents)

            state = next_state
            episode_reward += sum(rewards)

            if done:
                break

        # 记录指标
        metrics_tracker.add_episode(episode_reward, episode_delay, episode_energy, llm_advice)

        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_metrics = metrics_tracker.get_average_metrics(last_n=10)
            print(f"Episode {episode + 1}/{max_episodes}")
            print(f"  平均奖励: {avg_metrics['avg_reward']:.2f}")
            print(f"  LLM使用率: {avg_metrics['llm_usage_ratio']:.2f}")

        # 定期保存模型和绘制图表
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

    # 绘制最终图表
    plotter.plot_rewards(metrics_tracker.episode_rewards)
    if all_actions:
        plotter.plot_action_distribution(np.array(all_actions).reshape(-1, 2))

    print("训练完成!")
    return agents, metrics_tracker