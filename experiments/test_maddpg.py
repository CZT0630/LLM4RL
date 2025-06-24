import numpy as np
import torch
import os
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from algos.maddpg_agent import MADDPGAgent
from utils.config import load_config

def test_maddpg(model_path, config=None):
    if config is None:
        config = load_config()
    env = CloudEdgeDeviceEnv(config['environment'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[1] + 1
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
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            actions = [agent.select_action(state, llm_advice=None, add_noise=False) for agent in agents]
            next_state, rewards, done, _ = env.step(actions)
            state = next_state
            episode_reward += sum(rewards)
            if done:
                break
        print(f"[MADDPG] 测试 Episode {episode + 1} 完成，总奖励: {episode_reward:.2f}")
    print("[MADDPG] 测试完成!")