# # experiments/train_llm_maddpg.py
# import numpy as np
# import torch
# import os
# import random
# import gymnasium as gym
# from typing import Union
# from environment.cloud_edge_env import CloudEdgeDeviceEnv
# from llm_assistant.llm_client import LLMClient
# from llm_assistant.response_parser import ResponseParser
# from algos.maddpg_agent import MADDPGAgent
# from utils.plotting import Plotter
# from utils.metrics import MetricsTracker
# from utils.config import load_config


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def train_llm_maddpg(config=None):
#     # 加载配置
#     if config is None:
#         config = load_config()
#     set_seed(config.get('seed', 42))

#     # 创建保存目录
#     save_dir = config.get('save_dir', 'results')
#     os.makedirs(save_dir, exist_ok=True)

#     # 创建环境
#     env = CloudEdgeDeviceEnv(config['environment'])
#     if env is None:
#         raise ValueError("Failed to create the environment. Check the configuration.")

#     # 确认 action_space 是 Box 类型
#     action_space = env.action_space
#     if action_space is None or not isinstance(action_space, gym.spaces.Box):
#         raise ValueError("The action space is not a valid Box space.")
#     print(env.observation_space)
#     print(action_space)

#     # 创建LLM客户端
#     llm_client = LLMClient(
#         api_key=config['llm'].get('api_key', ''),
#         model_name=config['llm']['model_name'],
#         server_url=config['llm'].get('server_url', 'http://10.200.1.35:8888/v1/completions'),
#         timeout_connect=config['llm'].get('timeout_connect', 120),
#         timeout_read=config['llm'].get('timeout_read', 300),
#         use_mock=config['llm'].get('use_mock_when_unavailable', True)
#     )

#     # 创建MADDPG智能体
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.shape[0]
#     max_action = env.action_space.high[1] + 1  # 目标节点数量
#     num_agents = env.num_devices

#     agents = []
#     for i in range(num_agents):
#         agent = MADDPGAgent(
#             state_dim=state_dim,
#             action_dim=action_dim,
#             max_action=max_action,
#             num_agents=num_agents,
#             agent_idx=i,
#             lr_actor=float(config['maddpg']['lr_actor']),
#             lr_critic=float(config['maddpg']['lr_critic']),
#             gamma=float(config['maddpg']['gamma']),
#             tau=float(config['maddpg']['tau']),
#             buffer_size=config['maddpg']['buffer_size'],
#             batch_size=config['maddpg']['batch_size'],
#             distillation_alpha=float(config['maddpg'].get('distillation_alpha', 0.5)),
#             distillation_temp=float(config['maddpg'].get('distillation_temp', 1.0)),
#             distillation_loss_type=config['maddpg'].get('distillation_loss_type', 'mse')
#         )
#         agents.append(agent)

#     # 创建绘图器和指标追踪器
#     plotter = Plotter(save_dir)
#     metrics_tracker = MetricsTracker()

#     # 训练参数
#     max_episodes = config['maddpg']['max_episodes']
#     max_steps = config['maddpg']['max_steps']
#     llm_query_freq = config['llm']['query_frequency']

#     # 设备、边缘和云端详细信息
#     device_info = [{"cpu": device.cpu_capacity, "memory": device.memory_capacity} for device in env.devices]
#     edge_info = [{"cpu": edge.cpu_capacity, "memory": edge.memory_capacity} for edge in env.edge_servers]
#     cloud_info = [{"cpu": cloud.cpu_capacity, "memory": cloud.memory_capacity} for cloud in env.cloud_servers]

#     # 存储LLM专家经验的缓冲区
#     llm_expert_buffer = {
#         'states': [],
#         'actions': []
#     }

#     # 训练循环
#     all_actions = []

#     for episode in range(max_episodes):
#         state, _ = env.reset()
#         episode_reward = 0
#         episode_delay = 0
#         episode_energy = 0
#         episode_llm_actions = []  # 记录本episode中LLM的建议动作

#         # 每llm_query_freq个episode咨询一次LLM
#         if episode % llm_query_freq == 0:
#             print(f"Episode {episode}: 咨询LLM获取卸载策略建议...")
#             strategies = llm_client.get_unload_strategy(state, device_info, edge_info, cloud_info)

#             # 解析LLM响应
#             llm_advice = ResponseParser.parse_unload_strategy(
#                 strategies,
#                 env.num_devices,
#                 env.num_edges,
#                 env.num_clouds
#             )
            
#             # 将LLM建议转换为动作格式并存储
#             llm_actions = []
#             for i in range(num_agents):
#                 agent_llm_advice = next((a for a in llm_advice if a["task_id"] == i), None)
#                 if agent_llm_advice:
#                     llm_actions.append([
#                         agent_llm_advice.get("offload_ratio", 0.0),
#                         agent_llm_advice.get("target_node", 0.0)
#                     ])
#                 else:
#                     llm_actions.append([0.0, 0.0])
            
#             # 记录LLM专家建议
#             llm_expert_buffer['states'].append(state)
#             llm_expert_buffer['actions'].append(llm_actions)
#             episode_llm_actions = llm_actions
#         else:
#             llm_advice = None
#             episode_llm_actions = []

#         for step in range(max_steps):
#             # 选择动作
#             actions = []
#             for i, agent in enumerate(agents):
#                 # 为每个智能体提供全局状态和LLM指导
#                 if llm_advice:
#                     agent_llm_advice = next((a for a in llm_advice if a["task_id"] == i), None)
#                     if agent_llm_advice:
#                         # 构建合适的LLM建议张量
#                         advice_tensor = torch.tensor([
#                             [
#                                 agent_llm_advice.get("offload_ratio", 0.0),  # 卸载比例
#                                 agent_llm_advice.get("target_node", 0.0)     # 目标节点
#                             ]
#                         ], dtype=torch.float32)
#                         agent_action = agent.select_action(state, advice_tensor)
#                     else:
#                         # 如果没有针对该智能体的建议
#                         agent_action = agent.select_action(state, None)
#                 else:
#                     # 没有LLM建议
#                     agent_action = agent.select_action(state, None)
                
#                 actions.append(agent_action)

#             all_actions.append(actions)

#             # 执行动作
#             next_state, rewards, terminated, truncated, info = env.step(actions)
#             done = terminated or truncated

#             # 存储经验
#             for i, agent in enumerate(agents):
#                 agent.replay_buffer.add(
#                     state, actions, rewards, next_state, done
#                 )

#             # 训练智能体
#             if len(agents[0].replay_buffer) > config['maddpg']['batch_size']:
#                 experiences = agents[0].replay_buffer.sample(config['maddpg']['batch_size'])
                
#                 # 准备LLM专家经验 (如果有)
#                 llm_experiences = None
#                 if len(llm_expert_buffer['states']) > 0:
#                     llm_experiences = (
#                         llm_expert_buffer['states'], 
#                         llm_expert_buffer['actions']
#                     )
                
#                 # 训练所有智能体
#                 train_metrics = {}
#                 for agent in agents:
#                     agent_metrics = agent.train(experiences, agents, llm_experiences)
#                     # 合并每个智能体的训练指标
#                     for k, v in agent_metrics.items():
#                         if k not in train_metrics:
#                             train_metrics[k] = []
#                         train_metrics[k].append(v)
                
#                 # 计算平均指标
#                 avg_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
#                 if (episode + 1) % 10 == 0 and step == 0:
#                     print(f"  训练指标: Actor损失={avg_metrics['actor_loss']:.4f}, "
#                           f"Critic损失={avg_metrics['critic_loss']:.4f}, "
#                           f"蒸馏损失={avg_metrics['distillation_loss']:.4f}")
                
#                 # 记录训练指标
#                 metrics_tracker.add_training_metrics(
#                     critic_loss=avg_metrics['critic_loss'],
#                     actor_loss=avg_metrics['actor_loss'],
#                     distillation_loss=avg_metrics['distillation_loss']
#                 )

#             state = next_state
#             episode_reward += sum(rewards)

#             if done:
#                 break

#         # 记录指标
#         metrics_tracker.add_episode(episode_reward, episode_delay, episode_energy, llm_advice is not None)

#         # 打印进度
#         if (episode + 1) % 10 == 0:
#             avg_metrics = metrics_tracker.get_average_metrics(last_n=10)
#             print(f"Episode {episode + 1}/{max_episodes}")
#             print(f"  平均奖励: {avg_metrics['avg_reward']:.2f}")
#             print(f"  LLM使用率: {avg_metrics['llm_usage_ratio']:.2f}")

#         # 定期保存模型和绘制图表
#         if (episode + 1) % 100 == 0:
#             for i, agent in enumerate(agents):
#                 torch.save(agent.actor.state_dict(), f"{save_dir}/actor_agent_{i}_episode_{episode + 1}.pth")
#                 torch.save(agent.critic.state_dict(), f"{save_dir}/critic_agent_{i}_episode_{episode + 1}.pth")

#             # 绘制训练图表
#             plotter.plot_rewards(metrics_tracker.episode_rewards)
#             plotter.plot_training_losses(metrics_tracker)
#             plotter.plot_metrics(metrics_tracker)
#             if all_actions:
#                 plotter.plot_action_distribution(np.array(all_actions).reshape(-1, 2))

#     # 保存最终模型
#     for i, agent in enumerate(agents):
#         torch.save(agent.actor.state_dict(), f"{save_dir}/actor_agent_{i}_final.pth")
#         torch.save(agent.critic.state_dict(), f"{save_dir}/critic_agent_{i}_final.pth")

#     # 绘制最终图表
#     plotter.plot_rewards(metrics_tracker.episode_rewards)
#     plotter.plot_training_losses(metrics_tracker)
#     plotter.plot_metrics(metrics_tracker)
#     if all_actions:
#         plotter.plot_action_distribution(np.array(all_actions).reshape(-1, 2))

#     print("训练完成!")
#     return agents, metrics_tracker