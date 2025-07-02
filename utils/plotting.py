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


def plot_training_curves(episode_rewards, episode_latencies, episode_energies, 
                        episode_completion_rates=None, training_losses=None, save_dir="results"):
    """绘制训练曲线 - 兼容函数"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LLM+MADDPG 训练曲线', fontsize=16, fontweight='bold')
    
    # 1. 奖励曲线
    if episode_rewards:
        axes[0, 0].plot(episode_rewards, 'b-', linewidth=2)
        axes[0, 0].set_title('Episode 奖励', fontsize=12)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('总奖励')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加移动平均线
        if len(episode_rewards) > 10:
            window_size = min(50, len(episode_rewards) // 10)
            moving_avg = []
            for i in range(len(episode_rewards)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(episode_rewards[start_idx:i+1]))
            axes[0, 0].plot(moving_avg, 'r--', linewidth=1, alpha=0.8, label=f'移动平均({window_size})')
            axes[0, 0].legend()
    
    # 2. 时延曲线
    if episode_latencies:
        axes[0, 1].plot(episode_latencies, 'g-', linewidth=2)
        axes[0, 1].set_title('平均时延', fontsize=12)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('时延 (s)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 能耗曲线
    if episode_energies:
        axes[1, 0].plot(episode_energies, 'orange', linewidth=2)
        axes[1, 0].set_title('平均能耗', fontsize=12)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('能耗 (J)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 完成率或训练损失
    if episode_completion_rates:
        axes[1, 1].plot(episode_completion_rates, 'purple', linewidth=2)
        axes[1, 1].set_title('任务完成率', fontsize=12)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('完成率')
        axes[1, 1].grid(True, alpha=0.3)
    elif training_losses:
        axes[1, 1].plot(training_losses, 'm-', linewidth=2)
        axes[1, 1].set_title('训练损失', fontsize=12)
        axes[1, 1].set_xlabel('更新步数')
        axes[1, 1].set_ylabel('损失')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, '暂无数据', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('训练指标', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存至: {save_path}")
    return save_path