# maddpg/maddpg_agent.py
import torch
import numpy as np
from .actor_critic import Actor, Critic
from .replay_buffer import ReplayBuffer
from .noise import OUNoise


class MADDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, num_agents, agent_idx,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005,
                 buffer_size=int(1e6), batch_size=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.num_agents = num_agents
        self.agent_idx = agent_idx
        self.gamma = gamma
        self.tau = tau

        # 创建Actor和Critic网络
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic = Critic(state_dim, action_dim, num_agents)
        self.critic_target = Critic(state_dim, action_dim, num_agents)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size

        # 探索噪声
        self.noise = OUNoise(action_dim)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

    def select_action(self, state, llm_advice=None, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        # 处理LLM建议
        llm_input = self._process_llm_advice(llm_advice)
        llm_input = torch.FloatTensor(llm_input.reshape(1, -1)).to(self.device)

        # 获取动作
        action = self.actor(state, llm_input).cpu().data.numpy().flatten()

        # 添加探索噪声
        if add_noise:
            noise = self.noise.sample()
            action = (action + noise).clip(0, self.max_action - 1)

        return action

    def train(self, experiences, all_agents):
        # 从经验中提取数据
        states, actions, rewards, next_states, dones = experiences

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # 提取当前智能体的奖励和终止标志
        rewards = rewards[:, self.agent_idx].unsqueeze(1)
        dones = dones[:, self.agent_idx].unsqueeze(1)

        # 获取所有智能体的下一状态动作
        next_actions = []
        for i, agent in enumerate(all_agents):
            # 处理LLM建议（假设没有LLM建议用于下一状态）
            llm_input = torch.zeros(next_states.size(0), 4).to(self.device)
            next_action = agent.actor_target(next_states[:, i * self.state_dim:(i + 1) * self.state_dim], llm_input)
            next_actions.append(next_action)
        next_actions = torch.cat(next_actions, dim=1)

        # 训练Critic网络
        self.critic_optimizer.zero_grad()

        # 计算目标Q值
        target_q = self.critic_target(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * target_q

        # 计算当前Q值
        current_q = self.critic(states, actions)

        # 计算Critic损失
        critic_loss = torch.nn.MSELoss()(current_q, target_q)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # 训练Actor网络
        self.actor_optimizer.zero_grad()

        # 获取当前智能体的预测动作
        llm_input = torch.zeros(states.size(0), 4).to(self.device)
        actor_actions = self.actor(states[:, self.agent_idx * self.state_dim:(self.agent_idx + 1) * self.state_dim],
                                   llm_input)

        # 复制其他智能体的动作
        actions_pred = actions.clone()
        actions_pred[:, self.agent_idx * self.action_dim:(self.agent_idx + 1) * self.action_dim] = actor_actions

        # 计算Actor损失
        actor_loss = -self.critic(states, actions_pred).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        # 软更新目标网络
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

    def _soft_update(self, local_model, target_model):
        """软更新目标网络参数"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def _process_llm_advice(self, llm_advice):
        """处理LLM建议，转换为网络输入格式"""
        if llm_advice is None or not llm_advice:
            # 默认建议：不卸载
            return np.array([0.0, 0.0, 0.0, 0.0])

        # 找到当前智能体的建议
        agent_advice = next((a for a in llm_advice if a["task_id"] == self.agent_idx), None)
        if not agent_advice:
            return np.array([0.0, 0.0, 0.0, 0.0])

        # 编码卸载比例
        unload_ratio = agent_advice["unload_ratio"]

        # 编码目标节点
        target_idx = agent_advice["target_idx"]
        target_encoding = np.zeros(3)  # 本地、边缘、云端
        if target_idx == 0:  # 本地
            target_encoding[0] = 1
        elif 1 <= target_idx <= self.num_edges:  # 边缘
            target_encoding[1] = 1
        else:  # 云端
            target_encoding[2] = 1

        # 组合成LLM输入向量
        return np.concatenate([[unload_ratio], target_encoding])