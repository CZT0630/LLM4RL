# experiments/test_llm_maddpg.py
import numpy as np
import torch
import random
import os
import gymnasium as gym
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from llm_assistant.llm_client import LLMClient
from llm_assistant.response_parser import ResponseParser
from algos.maddpg_agent import MADDPGAgent
from utils.config import load_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_llm_maddpg(model_path, config=None):
    # 加载配置
    if config is None:
        config = load_config()
    set_seed(config.get('seed', 42))

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
            agent_idx=i
        )

        # 加载预训练模型
        actor_path = f"{model_path}/actor_agent_{i}_final.pth"
        critic_path = f"{model_path}/critic_agent_{i}_final.pth"

        if os.path.exists(actor_path):
            agent.actor.load_state_dict(torch.load(actor_path))
            agent.actor.eval()

        if os.path.exists(critic_path):
            agent.critic.load_state_dict(torch.load(critic_path))
            agent.critic.eval()

        agents.append(agent)

    # 设备、边缘和云端详细信息
    device_info = [{"cpu": device.cpu_capacity, "memory": device.memory_capacity} for device in env.devices]
    edge_info = [{"cpu": edge.cpu_capacity, "memory": edge.memory_capacity} for edge in env.edge_servers]
    cloud_info = [{"cpu": cloud.cpu_capacity, "memory": cloud.memory_capacity} for cloud in env.cloud_servers]

    # 测试参数
    num_episodes = config['testing']['num_episodes']
    max_steps = config['maddpg']['max_steps']

    # 记录所有episode的指标
    all_episode_energy = []
    all_episode_delay = []
    all_episode_util = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_delay = 0
        episode_util = 0
        step_count = 0

        print(f"\n测试 Episode {episode + 1}/{num_episodes}")

        for step in range(max_steps):
            # 选择动作
            actions = []
            
            # 每步都咨询LLM获取建议
            strategies = llm_client.get_unload_strategy(state, device_info, edge_info, cloud_info)
            llm_advice = ResponseParser.parse_unload_strategy(
                strategies,
                env.num_devices,
                env.num_edges,
                env.num_clouds
            )
            
            for i, agent in enumerate(agents):
                # 为每个智能体提供全局状态和LLM指导
                if llm_advice:
                    agent_llm_advice = next((a for a in llm_advice if a["task_id"] == i), None)
                    if agent_llm_advice:
                        # 构建合适的LLM建议张量
                        advice_tensor = torch.tensor([
                            [
                                agent_llm_advice.get("offload_ratio", 0.0),  # 卸载比例
                                agent_llm_advice.get("target_node", 0.0)     # 目标节点
                            ]
                        ], dtype=torch.float32)
                        agent_action = agent.select_action(state, advice_tensor, add_noise=False)
                    else:
                        # 如果没有针对该智能体的建议
                        agent_action = agent.select_action(state, None, add_noise=False)
                else:
                    # 没有LLM建议
                    agent_action = agent.select_action(state, None, add_noise=False)
                
                actions.append(agent_action)

            # 执行动作
            next_state, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # 打印详细信息
            if step % 10 == 0:
                print(f"步骤 {step}:")
                for i, (action, reward) in enumerate(zip(actions, rewards)):
                    offload_ratio = action[0]
                    target_node = int(action[1])

                    target_name = "本地"
                    if target_node >= 1 and target_node <= env.num_edges:
                        target_name = f"边缘服务器 {target_node - 1}"
                    elif target_node > env.num_edges:
                        target_name = "云端"

                    print(f"  设备 {i}: 卸载比例={offload_ratio:.2f}, 目标={target_name}, 奖励={reward:.2f}")

            # 累加能耗、时延、资源利用率
            episode_energy += sum(info['energies'])
            episode_delay += sum(info['delays'])
            episode_util += sum(info['utilizations'])
            step_count += 1

            state = next_state
            episode_reward += sum(rewards)

            if done:
                print(f"Episode {episode + 1} 完成，总奖励: {episode_reward:.2f}")
                break

        avg_energy = episode_energy / num_agents
        avg_delay = episode_delay / num_agents
        avg_util = episode_util / (num_agents * step_count) if step_count > 0 else 0
        all_episode_energy.append(avg_energy)
        all_episode_delay.append(avg_delay)
        all_episode_util.append(avg_util)
        print(f"Episode {episode + 1} 平均能耗: {avg_energy:.4f}, 平均资源利用率: {avg_util:.4f}, 平均任务时延: {avg_delay:.4f}")

    print("测试完成!")
    return all_episode_energy, all_episode_util, all_episode_delay