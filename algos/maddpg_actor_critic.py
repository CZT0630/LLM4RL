# maddpg/actor_critic.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, llm_input_dim=2):
        super(Actor, self).__init__()

        # 状态编码器 - 处理设备、任务和环境状态
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # LLM建议编码器 - 处理LLM提供的卸载比例和目标节点
        self.llm_encoder = nn.Sequential(
            nn.Linear(llm_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 合并层 - 融合状态表示和LLM建议
        self.combined = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.max_action = max_action

    def forward(self, state, llm_advice=None):
        # 处理状态输入
        state_repr = self.state_encoder(state)

        # 处理LLM建议
        if llm_advice is None:
            # 如果没有LLM建议，使用默认值
            llm_advice = torch.zeros(state.size(0), 2).to(state.device)

        llm_repr = self.llm_encoder(llm_advice)

        # 合并表示
        combined = torch.cat([state_repr, llm_repr], dim=1)

        # 生成动作
        action = self.combined(combined)
        action[:, 0] = torch.sigmoid(action[:, 0])  # 卸载比例限制在[0,1]
        action[:, 1] = torch.tanh(action[:, 1]) * (self.max_action - 1)  # 目标节点索引
        action[:, 1] = (action[:, 1] + self.max_action - 1) / 2  # 映射到[0, max_action-1]

        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(Critic, self).__init__()

        self.layer1 = nn.Linear(state_dim * num_agents + action_dim * num_agents, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, state, action):
        # 连接所有智能体的状态和动作
        sa = torch.cat([state, action], 1)

        q = F.relu(self.layer1(sa))
        q = F.relu(self.layer2(q))
        q = self.layer3(q)

        return q