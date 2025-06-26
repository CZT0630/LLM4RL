# utils/metrics.py
class MetricsTracker:
    """跟踪并记录训练过程中的指标"""
    def __init__(self):
        self.episode_rewards = []
        self.episode_delays = []
        self.episode_energy = []
        self.llm_used = []
        self.distillation_losses = []  # 蒸馏损失
        self.policy_similarity = []    # LLM与MADDPG行为相似度
        self.critic_losses = []        # Critic损失
        self.actor_losses = []         # Actor损失

    def add_episode(self, reward, delay, energy, llm_used=False):
        """添加单个episode的指标"""
        self.episode_rewards.append(reward)
        self.episode_delays.append(delay)
        self.episode_energy.append(energy)
        self.llm_used.append(1 if llm_used else 0)

    def add_training_metrics(self, critic_loss, actor_loss, distillation_loss, similarity=None):
        """添加训练过程的指标"""
        self.critic_losses.append(critic_loss)
        self.actor_losses.append(actor_loss)
        self.distillation_losses.append(distillation_loss)
        if similarity is not None:
            self.policy_similarity.append(similarity)

    def get_average_metrics(self, last_n=None):
        """获取平均指标
        
        Args:
            last_n: 只计算最后n个episode的平均值，如果为None则计算所有
        """
        if last_n is not None:
            rewards = self.episode_rewards[-last_n:]
            delays = self.episode_delays[-last_n:]
            energy = self.episode_energy[-last_n:]
            llm_used = self.llm_used[-last_n:]
            distillation = self.distillation_losses[-last_n:] if self.distillation_losses else []
            similarity = self.policy_similarity[-last_n:] if self.policy_similarity else []
            critic = self.critic_losses[-last_n:] if self.critic_losses else []
            actor = self.actor_losses[-last_n:] if self.actor_losses else []
        else:
            rewards = self.episode_rewards
            delays = self.episode_delays
            energy = self.episode_energy
            llm_used = self.llm_used
            distillation = self.distillation_losses
            similarity = self.policy_similarity
            critic = self.critic_losses
            actor = self.actor_losses

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        avg_delay = sum(delays) / len(delays) if delays else 0
        avg_energy = sum(energy) / len(energy) if energy else 0
        llm_usage_ratio = sum(llm_used) / len(llm_used) if llm_used else 0
        avg_distillation = sum(distillation) / len(distillation) if distillation else 0
        avg_similarity = sum(similarity) / len(similarity) if similarity else 0
        avg_critic = sum(critic) / len(critic) if critic else 0
        avg_actor = sum(actor) / len(actor) if actor else 0

        return {
            'avg_reward': avg_reward,
            'avg_delay': avg_delay,
            'avg_energy': avg_energy,
            'llm_usage_ratio': llm_usage_ratio,
            'avg_distillation_loss': avg_distillation,
            'avg_policy_similarity': avg_similarity,
            'avg_critic_loss': avg_critic,
            'avg_actor_loss': avg_actor
        }