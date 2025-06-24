import numpy as np
import torch
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from llm_assistant.llm_client import LLMClient
from llm_assistant.response_parser import ResponseParser
from utils.config import load_config

def test_llm(model_path, config=None):
    if config is None:
        config = load_config()
    env = CloudEdgeDeviceEnv(config['environment'])
    llm_client = LLMClient(
        api_key=config['llm']['api_key'],
        model_name=config['llm']['model_name']
    )
    num_episodes = config['testing']['num_episodes']
    max_steps = config['maddpg']['max_steps']
    num_agents = env.num_devices
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            device_info = [{"cpu": d.cpu_capacity, "memory": d.memory_capacity} for d in env.devices]
            edge_info = [{"cpu": e.cpu_capacity, "memory": e.memory_capacity} for e in env.edge_servers]
            cloud_info = [{"cpu": c.cpu_capacity, "memory": c.memory_capacity} for c in env.cloud_servers]
            strategies = llm_client.get_unload_strategy(state, device_info, edge_info, cloud_info)
            llm_advice = ResponseParser.parse_unload_strategy(
                strategies,
                env.num_devices,
                env.num_edges,
                env.num_clouds
            )
            actions = []
            for i in range(num_agents):
                agent_llm_advice = [a for a in llm_advice if a["task_id"] == i] if llm_advice else None
                if agent_llm_advice and len(agent_llm_advice) > 0:
                    offload_ratio = agent_llm_advice[0].get("offload_ratio", 0.0)
                    target_node = agent_llm_advice[0].get("target_node", 0.0)
                    actions.append([offload_ratio, target_node])
                else:
                    actions.append([0.0, 0.0])
            actions = np.array(actions)
            next_state, rewards, done, _ = env.step(actions)
            state = next_state
            episode_reward += sum(rewards)
            if done:
                break
        print(f"[LLM] 测试 Episode {episode + 1} 完成，总奖励: {episode_reward:.2f}")
    print("[LLM] 测试完成!")