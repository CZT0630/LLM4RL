# maddpg/maddpg_agent.py
import torch
import torch.nn.functional as F
import numpy as np
from .maddpg_actor_critic import Actor, Critic
from .replay_buffer import ReplayBuffer
from .noise import OUNoise


class MADDPGAgent:
    def __init__(self, state_dim, action_dim, num_agents, agent_idx, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Agent配置
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_idx = agent_idx
        
        # 超参数
        self.lr_actor = config.get('lr_actor', 0.001)
        self.lr_critic = config.get('lr_critic', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.01)
        self.batch_size = config.get('batch_size', 64)
        self.llm_distill_weight = config.get('llm_distill_weight', 0.1)  # LLM知识蒸馏权重
        
        # 网络初始化
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim * num_agents, action_dim * num_agents).to(self.device)
        self.critic_target = Critic(state_dim * num_agents, action_dim * num_agents).to(self.device)
        
        # 复制网络参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # 噪声
        self.noise = OUNoise(action_dim)
        
        # 经验回放
        self.memory = ReplayBuffer(config.get('buffer_size', 100000))

    def select_action(self, state, add_noise=True):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
            
        if add_noise:
            action += self.noise.sample()
            
        # 确保动作在有效范围内
        action = np.clip(action, 0.0, 1.0)
        
        # 处理边缘服务器ID（最后一个维度）
        if len(action) >= 4:
            action[-1] = np.clip(action[-1] * 5, 0, 4)  # 5个边缘服务器，索引0-4
            
        return action

    def train(self, replay_buffer=None):
        """从缓冲区训练Agent"""
        if replay_buffer is None:
            # 使用自己的缓冲区
            if len(self.memory) < self.batch_size:
                return
            buffer_to_use = self.memory
        else:
            # 使用传入的共享缓冲区
            if len(replay_buffer) < self.batch_size:
                return
            buffer_to_use = replay_buffer
        
        # 采样经验
        states, actions, rewards, next_states, dones, llm_actions = buffer_to_use.sample(self.batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # 获取当前Agent的信息
        agent_rewards = rewards[:, self.agent_idx].unsqueeze(1)
        agent_dones = dones[:, self.agent_idx].unsqueeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            # 获取下一状态的动作
            next_actions = []
            for i in range(self.num_agents):
                if i == self.agent_idx:
                    next_action = self.actor_target(next_states[:, i])
                else:
                    # 这里应该是其他Agent的目标Actor网络，简化为当前Actor
                    next_action = self.actor_target(next_states[:, i])
                next_actions.append(next_action)
            
            next_actions = torch.cat(next_actions, dim=1)
            next_q_values = self.critic_target(next_states.view(self.batch_size, -1), next_actions)
            target_q_values = agent_rewards + self.gamma * next_q_values * (~agent_dones)
        
        # 当前Q值
        current_actions = actions.view(self.batch_size, -1)
        current_q_values = self.critic(states.view(self.batch_size, -1), current_actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor损失（策略梯度 + LLM知识蒸馏）
        predicted_actions = []
        for i in range(self.num_agents):
            if i == self.agent_idx:
                predicted_action = self.actor(states[:, i])
            else:
                # 使用当前动作作为其他Agent的动作
                predicted_action = actions[:, i]
            predicted_actions.append(predicted_action)
        
        predicted_actions_cat = torch.cat(predicted_actions, dim=1)
        actor_loss = -self.critic(states.view(self.batch_size, -1), predicted_actions_cat).mean()
        
        # LLM知识蒸馏损失
        if len(llm_actions) > 0 and any(len(llm_action) > 0 for llm_action in llm_actions):
            llm_distill_loss = self._compute_llm_distillation_loss(
                states[:, self.agent_idx], predicted_actions[self.agent_idx], llm_actions
            )
            actor_loss += self.llm_distill_weight * llm_distill_loss
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)

    def _compute_llm_distillation_loss(self, states, predicted_actions, llm_actions):
        """计算LLM知识蒸馏损失"""
        distill_loss = 0.0
        valid_samples = 0
        
        for i, llm_action in enumerate(llm_actions):
            if isinstance(llm_action, list) and len(llm_action) > self.agent_idx:
                # 获取该Agent对应的LLM专家动作
                llm_agent_action = llm_action[self.agent_idx]
                if llm_agent_action is not None and len(llm_agent_action) > 0:
                    llm_tensor = torch.FloatTensor(llm_agent_action).to(self.device)
                    # 确保维度匹配
                    if llm_tensor.shape[-1] == predicted_actions.shape[-1]:
                        distill_loss += F.mse_loss(predicted_actions[i], llm_tensor)
                        valid_samples += 1
        
        if valid_samples > 0:
            return distill_loss / valid_samples
        else:
            return torch.tensor(0.0, device=self.device)

    def _soft_update(self, target, source, tau):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])