# maddpg/replay_buffer.py
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, llm_action=None):
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: Agent执行的动作
            reward: 奖励
            next_state: 下一状态
            done: 是否终止
            llm_action: LLM专家动作（可选）
        """
        experience = {
            'state': state,
            'action': action, 
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'llm_action': llm_action if llm_action is not None else []
        }
        self.buffer.append(experience)

    def sample(self, batch_size):
        """采样经验"""
        sampled_experiences = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones, llm_actions = [], [], [], [], [], []

        for experience in sampled_experiences:
            states.append(experience['state'])
            actions.append(experience['action'])
            rewards.append(experience['reward'])
            next_states.append(experience['next_state'])
            dones.append(experience['done'])
            llm_actions.append(experience['llm_action'])

        return (states, actions, rewards, next_states, dones, llm_actions)

    def __len__(self):
        return len(self.buffer)