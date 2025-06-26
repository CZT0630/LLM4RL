import numpy as np
import torch
import gymnasium as gym
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from llm_assistant.llm_client import LLMClient
from llm_assistant.response_parser import ResponseParser
from utils.config import load_config

def test_llm(model_path, config=None):
    if config is None:
        config = load_config()
    env = CloudEdgeDeviceEnv(config['environment'])
    llm_client = LLMClient(
        api_key=config['llm'].get('api_key', ''),
        model_name=config['llm']['model_name'],
        server_url=config['llm'].get('server_url', 'http://10.200.1.35:8888/v1/completions'),
        timeout_connect=config['llm'].get('timeout_connect', 120),
        timeout_read=config['llm'].get('timeout_read', 300),
        use_mock=config['llm'].get('use_mock_when_unavailable', True)
    )
    num_episodes = config['testing']['num_episodes']
    max_steps = config['maddpg']['max_steps']
    num_agents = env.num_devices
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
        for step in range(max_steps):
            device_info = [{"cpu": d.cpu_capacity, "memory": d.memory_capacity} for d in env.devices]
            edge_info = [{"cpu": e.cpu_capacity, "memory": e.memory_capacity} for e in env.edge_servers]
            cloud_info = [{"cpu": c.cpu_capacity, "memory": c.memory_capacity} for c in env.cloud_servers]
            # 获取LLM策略并执行
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
            next_state, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            episode_energy += sum(info['energies'])
            episode_delay += sum(info['delays'])
            episode_util += sum(info['utilizations'])
            step_count += 1
            state = next_state
            episode_reward += sum(rewards)
            if done:
                break
        avg_energy = episode_energy / num_agents
        avg_delay = episode_delay / num_agents
        avg_util = episode_util / (num_agents * step_count) if step_count > 0 else 0
        all_episode_energy.append(avg_energy)
        all_episode_delay.append(avg_delay)
        all_episode_util.append(avg_util)
        print(f"[LLM] Episode {episode + 1} 平均能耗: {avg_energy:.4f}, 平均资源利用率: {avg_util:.4f}, 平均任务时延: {avg_delay:.4f}")
    print("[LLM] 测试完成!")
    return all_episode_energy, all_episode_util, all_episode_delay