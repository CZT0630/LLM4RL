# experiments/test_llm_maddpg.py
import numpy as np
import torch
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from llm_assistant.llm_client import LLMClient
from llm_assistant.response_parser import ResponseParser
from algos.maddpg_agent import MADDPGAgent
from utils.config import load_config


def test_llm_maddpg(model_path, config=None):
    # 加载配置
    if config is None:
        config = load_config()

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
    num_episodes = 10
    max_steps = config['maddpg']['max_steps']

    # 测试循环
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        print(f"\n测试 Episode {episode + 1}/{num_episodes}")

        for step in range(max_steps):
            # 选择动作（不添加探索噪声）
            actions = []
            for i, agent in enumerate(agents):
                # 每步都咨询LLM获取建议
                strategies = llm_client.get_unload_strategy(state, device_info, edge_info, cloud_info)
                llm_advice = ResponseParser.parse_unload_strategy(
                    strategies,
                    env.num_devices,
                    env.num_edges,
                    env.num_clouds
                )

                agent_llm_advice = [a for a in llm_advice if a["task_id"] == i] if llm_advice else None
                agent_action = agent.select_action(state, agent_llm_advice, add_noise=False)
                actions.append(agent_action)

            # 执行动作
            next_state, rewards, done, info = env.step(actions)

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

            state = next_state
            episode_reward += sum(rewards)

            if done:
                print(f"Episode {episode + 1} 完成，总奖励: {episode_reward:.2f}")
                break

        print(f"测试 Episode {episode + 1} 完成，总奖励: {episode_reward:.2f}")

    print("测试完成!")