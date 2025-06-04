# utils/plotting.py
import matplotlib.pyplot as plt
import numpy as np
import os


class Plotter:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_rewards(self, rewards, title="Training Rewards"):
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'rewards.png'))
        plt.close()

    def plot_llm_usage(self, llm_usage, title="LLM Guidance Usage"):
        plt.figure(figsize=(10, 6))
        plt.plot(llm_usage)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('LLM Influence')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'llm_usage.png'))
        plt.close()

    def plot_action_distribution(self, actions, title="Action Distribution"):
        # 假设actions是一个二维数组，第一列是卸载比例，第二列是目标节点
        plt.figure(figsize=(12, 5))

        # 卸载比例分布
        plt.subplot(1, 2, 1)
        plt.hist(actions[:, 0], bins=20, range=(0, 1))
        plt.title('Offloading Ratio Distribution')
        plt.xlabel('Offloading Ratio')
        plt.ylabel('Frequency')

        # 目标节点分布
        plt.subplot(1, 2, 2)
        plt.hist(actions[:, 1], bins=5, range=(0, 4))  # 假设5个目标节点
        plt.title('Target Node Distribution')
        plt.xlabel('Target Node Index')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'action_distribution.png'))
        plt.close()