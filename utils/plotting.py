# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os


class Plotter:
    """绘制训练和评估的指标"""
    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_rewards(self, rewards):
        """绘制奖励曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'rewards.png'))
        plt.close()

    def plot_action_distribution(self, actions):
        """绘制动作分布"""
        plt.figure(figsize=(12, 5))
        
        # 卸载比例分布
        plt.subplot(1, 2, 1)
        plt.hist(actions[:, 0], bins=20, alpha=0.7)
        plt.xlabel('Offload Ratio')
        plt.ylabel('Frequency')
        plt.title('Offload Ratio Distribution')
        plt.grid(True)
        
        # 目标节点分布
        plt.subplot(1, 2, 2)
        plt.hist(actions[:, 1], bins=10, alpha=0.7)
        plt.xlabel('Target Node')
        plt.ylabel('Frequency')
        plt.title('Target Node Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'action_distribution.png'))
        plt.close()
    
    def plot_training_losses(self, metrics_tracker):
        """绘制训练损失曲线"""
        if not metrics_tracker.critic_losses or not metrics_tracker.actor_losses:
            return
            
        plt.figure(figsize=(12, 10))
        
        # Critic损失
        plt.subplot(2, 2, 1)
        plt.plot(metrics_tracker.critic_losses)
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.title('Critic Loss')
        plt.grid(True)
        
        # Actor损失
        plt.subplot(2, 2, 2)
        plt.plot(metrics_tracker.actor_losses)
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.title('Actor Loss')
        plt.grid(True)
        
        # 蒸馏损失
        plt.subplot(2, 2, 3)
        plt.plot(metrics_tracker.distillation_losses)
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.title('Distillation Loss')
        plt.grid(True)
        
        # 行为相似度（如果有）
        if metrics_tracker.policy_similarity:
            plt.subplot(2, 2, 4)
            plt.plot(metrics_tracker.policy_similarity)
            plt.xlabel('Update Steps')
            plt.ylabel('Similarity')
            plt.title('Policy Similarity with LLM')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_losses.png'))
        plt.close()
        
    def plot_metrics(self, metrics_tracker):
        """绘制综合指标"""
        plt.figure(figsize=(15, 10))
        
        # 奖励
        plt.subplot(2, 2, 1)
        plt.plot(metrics_tracker.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.grid(True)
        
        # 延迟
        plt.subplot(2, 2, 2)
        plt.plot(metrics_tracker.episode_delays)
        plt.xlabel('Episode')
        plt.ylabel('Delay')
        plt.title('Task Delay')
        plt.grid(True)
        
        # 能耗
        plt.subplot(2, 2, 3)
        plt.plot(metrics_tracker.episode_energy)
        plt.xlabel('Episode')
        plt.ylabel('Energy')
        plt.title('Energy Consumption')
        plt.grid(True)
        
        # LLM使用率
        plt.subplot(2, 2, 4)
        window_size = 10
        llm_usage_moving_avg = []
        for i in range(len(metrics_tracker.llm_used)):
            if i < window_size:
                llm_usage_moving_avg.append(np.mean(metrics_tracker.llm_used[:i+1]))
            else:
                llm_usage_moving_avg.append(np.mean(metrics_tracker.llm_used[i-window_size+1:i+1]))
        
        plt.plot(llm_usage_moving_avg)
        plt.xlabel('Episode')
        plt.ylabel('Usage Ratio')
        plt.title('LLM Usage Ratio (Moving Avg)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'metrics.png'))
        plt.close()