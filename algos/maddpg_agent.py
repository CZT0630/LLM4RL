# maddpg/maddpg_agent.py
import torch
import numpy as np
from .maddpg_actor_critic import Actor, Critic
from .replay_buffer import ReplayBuffer
from .noise import OUNoise


class MADDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, num_agents, agent_idx,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005,
                 buffer_size=int(1e6), batch_size=100, 
                 distillation_alpha=0.5, distillation_temp=1.0, distillation_loss_type="mse"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.num_agents = num_agents
        self.agent_idx = agent_idx
        self.gamma = gamma
        self.tau = tau
        
        # 蒸馏学习参数
        self.distillation_alpha = distillation_alpha  # 蒸馏损失权重
        self.distillation_temp = distillation_temp    # 蒸馏温度
        self.distillation_loss_type = distillation_loss_type  # 蒸馏损失类型

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
        """根据状态和可选的LLM建议选择动作"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            if llm_advice is not None:
                # llm_advice格式: tensor([[offload_ratio, target_node]])
                # 确保llm_advice是张量形式
                if not isinstance(llm_advice, torch.Tensor):
                    llm_advice = torch.FloatTensor([[0.0, 0.0]]).to(self.device)
                # 使用LLM建议作为输入
                action = self.actor(state, llm_advice.to(self.device)).cpu().data.numpy().flatten()
            else:
                # 无LLM建议时使用零张量
                zero_advice = torch.zeros((1, 2)).to(self.device)  # [offload_ratio, target_node]
                action = self.actor(state, zero_advice).cpu().data.numpy().flatten()
        
        self.actor.train()
        
        if add_noise:
            # 添加探索噪声
            action = action + self.noise.sample()
            
            # 约束动作范围
            action[0] = np.clip(action[0], 0.0, 1.0)  # 卸载比例 [0,1]
            action[1] = np.clip(action[1], 0.0, self.max_action - 1)  # 目标节点
        
        return action

    def train(self, experiences, all_agents, llm_experiences=None):
        """训练智能体，包含知识蒸馏
        
        Args:
            experiences: 从经验回放中采样的经验
            all_agents: 所有智能体的列表
            llm_experiences: LLM专家建议的经验 (states, llm_actions)
        """
        # 从经验中提取数据
        states, actions, rewards, next_states, dones = experiences

        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # 提取当前智能体的奖励和终止标志
        agent_rewards = rewards[:, self.agent_idx].unsqueeze(1)
        agent_dones = dones[:, self.agent_idx].unsqueeze(1)

        # 获取所有智能体的下一状态动作
        next_actions = []
        for i, agent in enumerate(all_agents):
            # 处理LLM建议（假设没有LLM建议用于下一状态）
            zero_advice = torch.zeros(next_states.size(0), 2).to(self.device)
            next_action = agent.actor_target(next_states[:, i * self.state_dim:(i + 1) * self.state_dim], zero_advice)
            next_actions.append(next_action)
        next_actions = torch.cat(next_actions, dim=1)

        # 训练Critic网络
        self.critic_optimizer.zero_grad()

        # 计算目标Q值
        target_q = self.critic_target(next_states, next_actions)
        target_q = agent_rewards + (1 - agent_dones) * self.gamma * target_q

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
        zero_advice = torch.zeros(states.size(0), 2).to(self.device)
        actor_actions = self.actor(states[:, self.agent_idx * self.state_dim:(self.agent_idx + 1) * self.state_dim],
                                   zero_advice)

        # 复制其他智能体的动作
        actions_pred = actions.clone()
        actions_pred[:, self.agent_idx * self.action_dim:(self.agent_idx + 1) * self.action_dim] = actor_actions

        # 计算策略梯度损失
        policy_loss = -self.critic(states, actions_pred).mean()
        
        # 如果有LLM专家建议，加入蒸馏损失
        distillation_loss = 0.0
        if llm_experiences is not None and len(llm_experiences) > 0:
            llm_states, llm_actions = llm_experiences
            llm_states = torch.FloatTensor(llm_states).to(self.device)
            llm_actions = torch.FloatTensor(llm_actions).to(self.device)
            
            # 获取当前agent相关的状态和LLM建议
            agent_states = llm_states[:, self.agent_idx * self.state_dim:(self.agent_idx + 1) * self.state_dim]
            agent_llm_actions = llm_actions[:, self.agent_idx * self.action_dim:(self.agent_idx + 1) * self.action_dim]
            
            # 使用Actor网络预测动作
            zero_advice = torch.zeros(agent_states.size(0), 2).to(self.device)
            pred_actions = self.actor(agent_states, zero_advice)
            
            # 计算蒸馏损失
            if self.distillation_loss_type == "mse":
                # 使用均方误差作为蒸馏损失
                distillation_loss = torch.nn.MSELoss()(pred_actions, agent_llm_actions)
            elif self.distillation_loss_type == "kl":
                # 使用KL散度作为蒸馏损失 (需要将动作转换为概率分布)
                pred_logits = pred_actions / self.distillation_temp
                target_logits = agent_llm_actions / self.distillation_temp
                log_softmax_pred = torch.nn.functional.log_softmax(pred_logits, dim=1)
                softmax_target = torch.nn.functional.softmax(target_logits, dim=1)
                distillation_loss = torch.nn.functional.kl_div(log_softmax_pred, softmax_target, reduction='batchmean')
            else:
                # 默认使用均方误差
                distillation_loss = torch.nn.MSELoss()(pred_actions, agent_llm_actions)
        
        # 总损失 = 策略梯度损失 + 蒸馏损失权重 * 蒸馏损失
        actor_loss = policy_loss + self.distillation_alpha * distillation_loss
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        # 软更新目标网络
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'policy_loss': policy_loss.item(),
            'distillation_loss': distillation_loss.item() if isinstance(distillation_loss, torch.Tensor) else 0.0
        }

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