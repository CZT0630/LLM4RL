# utils/metrics.py
class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_rewards = []
        self.episode_delays = []
        self.episode_energies = []
        self.llm_usage = []

    def add_episode(self, reward, delay, energy, llm_advice):
        self.episode_rewards.append(reward)
        self.episode_delays.append(delay)
        self.episode_energies.append(energy)
        self.llm_usage.append(1 if llm_advice else 0)

    def get_average_metrics(self, last_n=None):
        if last_n is None:
            last_n = len(self.episode_rewards)

        if last_n == 0:
            return {
                "avg_reward": 0,
                "avg_delay": 0,
                "avg_energy": 0,
                "llm_usage_ratio": 0
            }

        return {
            "avg_reward": sum(self.episode_rewards[-last_n:]) / last_n,
            "avg_delay": sum(self.episode_delays[-last_n:]) / last_n,
            "avg_energy": sum(self.episode_energies[-last_n:]) / last_n,
            "llm_usage_ratio": sum(self.llm_usage[-last_n:]) / last_n
        }