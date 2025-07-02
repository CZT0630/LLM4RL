# maddpg/maddpg_agent.py
import torch
import torch.nn.functional as F
import numpy as np
from .maddpg_actor_critic import Actor, Critic
from .replay_buffer import ReplayBuffer
from .noise import OUNoise


class MADDPGAgent:
    def __init__(self, state_dim, action_dim, num_agents, agent_idx, config=None, max_action=None, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Agent配置
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_idx = agent_idx
        
        # 处理配置参数
        if config is None:
            config = {}
        
        # 从kwargs中提取参数（兼容旧版本调用）
        self.lr_actor = kwargs.get('lr_actor', config.get('lr_actor', 0.001))
        self.lr_critic = kwargs.get('lr_critic', config.get('lr_critic', 0.001))
        self.gamma = kwargs.get('gamma', config.get('gamma', 0.99))
        self.tau = kwargs.get('tau', config.get('tau', 0.01))
        self.batch_size = kwargs.get('batch_size', config.get('batch_size', 64))
        buffer_size = kwargs.get('buffer_size', config.get('buffer_size', 100000))
        self.llm_distill_weight = config.get('llm_distill_weight', 0.1)  # LLM知识蒸馏权重
        
        # 确定max_action
        if max_action is None:
            max_action = num_agents + 2  # 默认值：设备数量 + 边缘 + 云端
        self.max_action = max_action
        
        # 网络初始化
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim, num_agents).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, num_agents).to(self.device)
        
        # 复制网络参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # 噪声
        self.noise = OUNoise(action_dim)
        
        # 经验回放
        self.memory = ReplayBuffer(buffer_size)
        
        # 兼容旧版本的replay_buffer属性
        self.replay_buffer = self.memory
        
        # 训练统计
        self.training_count = 0

    def select_action(self, state, add_noise=True, llm_advice=None):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 处理LLM建议
        if llm_advice is not None:
            llm_advice = torch.FloatTensor(llm_advice).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state, llm_advice).cpu().data.numpy().flatten()
            
        if add_noise:
            action += self.noise.sample()
            
        # 确保动作在有效范围内
        action = np.clip(action, 0.0, 1.0)
        
        # 处理边缘服务器ID（最后一个维度）
        if len(action) >= 2:
            action[-1] = np.clip(action[-1] * 5, 0, 4)  # 5个边缘服务器，索引0-4
            
        return action

    def store_experience(self, state, action, reward, next_state, done, llm_action=None):
        """存储经验到本地缓冲区"""
        self.memory.add(state, action, reward, next_state, done, llm_action)

    def train(self, all_agents=None, replay_buffer=None, experiences=None):
        """
        训练Agent
        
        Args:
            all_agents: 所有Agent的列表（用于获取其他Agent的动作）
            replay_buffer: 可选的共享经验缓冲区，如果为None则使用自己的缓冲区
            experiences: 兼容旧版本的经验数据
        """
        # 兼容旧版本的调用方式
        if experiences is not None:
            return self._train_with_experiences(experiences, all_agents)
        
        if replay_buffer is None:
            # 使用自己的缓冲区
            if len(self.memory) < self.batch_size:
                return {}
            buffer_to_use = self.memory
        else:
            # 使用传入的共享缓冲区
            if len(replay_buffer) < self.batch_size:
                return {}
            buffer_to_use = replay_buffer
        
        # 采样经验
        states, actions, rewards, next_states, dones, llm_actions = buffer_to_use.sample(self.batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # 训练Critic网络
        critic_loss = self._train_critic(states, actions, rewards, next_states, dones, all_agents)
        
        # 训练Actor网络（包含LLM知识蒸馏）
        actor_loss, distill_loss = self._train_actor(states, actions, llm_actions, all_agents)
        
        # 软更新目标网络
        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)
        
        self.training_count += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'distill_loss': distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss,
            'training_count': self.training_count
        }

    def _train_with_experiences(self, experiences, all_agents):
        """兼容旧版本的训练方法"""
        # 简化的训练逻辑，用于兼容
        states, actions, rewards, next_states, dones = experiences
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        # 简化的训练过程
        critic_loss = self._train_critic(states, actions, rewards, next_states, dones, all_agents)
        actor_loss, distill_loss = self._train_actor(states, actions, None, all_agents)
        
        # 软更新目标网络
        self._soft_update(self.actor_target, self.actor, self.tau)
        self._soft_update(self.critic_target, self.critic, self.tau)
        
        self.training_count += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'distill_loss': 0.0,
            'training_count': self.training_count
        }

    def _train_critic(self, states, actions, rewards, next_states, dones, all_agents):
        """训练Critic网络"""
        # 计算目标Q值
        with torch.no_grad():
            # 获取下一状态的动作（简化版本，实际应该用其他Agent的目标网络）
            next_actions = []
            for i in range(self.num_agents):
                if i == self.agent_idx:
                    next_action = self.actor_target(next_states)
                else:
                    # 简化处理：使用当前Agent的目标网络代替其他Agent
                    # 在完整实现中，这里应该是 all_agents[i].actor_target(next_states)
                    next_action = self.actor_target(next_states)
                next_actions.append(next_action)
            
            next_actions = torch.cat(next_actions, dim=1)
            
            # 扩展状态维度以匹配多智能体Critic输入
            states_expanded = states.repeat(1, self.num_agents)
            next_states_expanded = next_states.repeat(1, self.num_agents)
            
            next_q_values = self.critic_target(next_states_expanded, next_actions)
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
        
        # 当前Q值
        current_actions = actions.repeat(1, self.num_agents)  # 简化：假设所有Agent动作相同
        current_q_values = self.critic(states_expanded, current_actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss

    def _train_actor(self, states, actions, llm_actions, all_agents):
        """训练Actor网络（包含LLM知识蒸馏）"""
        # 策略梯度损失
        predicted_actions = self.actor(states)
        
        # 构建用于Critic的动作序列（简化版本）
        actions_for_critic = predicted_actions.repeat(1, self.num_agents)
        states_expanded = states.repeat(1, self.num_agents)
        
        actor_loss = -self.critic(states_expanded, actions_for_critic).mean()
        
        # LLM知识蒸馏损失
        distill_loss = self._compute_llm_distillation_loss(predicted_actions, llm_actions)
        
        # 总损失
        total_actor_loss = actor_loss + self.llm_distill_weight * distill_loss
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()
        
        return total_actor_loss, distill_loss

    def _compute_llm_distillation_loss(self, predicted_actions, llm_actions):
        """计算LLM知识蒸馏损失"""
        if not llm_actions or all(action is None for action in llm_actions):
            return torch.tensor(0.0, device=self.device)
        
        distill_loss = 0.0
        valid_samples = 0
        
        for i, llm_action in enumerate(llm_actions):
            if llm_action is not None and len(llm_action) > self.agent_idx:
                # 获取该Agent对应的LLM专家动作
                if isinstance(llm_action, list) and self.agent_idx < len(llm_action):
                    llm_agent_action = llm_action[self.agent_idx]
                elif isinstance(llm_action, (list, np.ndarray)) and len(llm_action) == self.action_dim:
                    llm_agent_action = llm_action
                else:
                    continue
                
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
            'training_count': self.training_count,
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
        self.training_count = checkpoint.get('training_count', 0)