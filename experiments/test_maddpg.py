import numpy as np
import torch
import os
import gymnasium as gym
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from algos.maddpg_agent import MADDPGAgent
from utils.config import load_config

def test_maddpg(model_path, config=None):
    if config is None:
        config = load_config()
    
    env = CloudEdgeDeviceEnv(config['environment'])
    
    # 使用正确的单个Agent状态维度
    state_dim = env.get_agent_state_dim()  # 20维：3(自己UE) + 10(所有ES) + 1(CS) + 6(自己任务)
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[1] + 1
    num_agents = env.num_devices
    
    print(f"🔧 [MADDPG测试] Agent配置信息:")
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
            agent_idx=i
        )
        actor_path = f"{model_path}/actor_agent_{i}_final.pth"
        critic_path = f"{model_path}/critic_agent_{i}_final.pth"
        if os.path.exists(actor_path):
            agent.actor.load_state_dict(torch.load(actor_path))
            agent.actor.eval()
        if os.path.exists(critic_path):
            agent.critic.load_state_dict(torch.load(critic_path))
            agent.critic.eval()
        agents.append(agent)
        
    num_episodes = config['testing']['num_episodes']
    max_steps = config['maddpg']['max_steps']
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
            # 使用正确的Agent状态提取
            actions = []
            for i, agent in enumerate(agents):
                agent_state = env.extract_agent_state(state, i)
                action = agent.select_action(agent_state, llm_advice=None, add_noise=False)
                actions.append(action)
            
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
        print(f"[MADDPG] Episode {episode + 1} 平均能耗: {avg_energy:.4f}, 平均资源利用率: {avg_util:.4f}, 平均任务时延: {avg_delay:.4f}")
        
    print("[MADDPG] 测试完成!")
    return all_episode_energy, all_episode_util, all_episode_delay