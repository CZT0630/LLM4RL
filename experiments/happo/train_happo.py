import numpy as np
import os
import random
import torch
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from algos.happo_agent import HAPPOAgent
from algos.common.trajectory_buffer import TrajectoryBuffer
from utils.plotting import Plotter
from utils.metrics import MetricsTracker
from utils.config import load_config
from utils.path_manager import get_path_manager
from utils.csv_saver import save_training_metrics_csv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_happo(config=None):
    if config is None:
        config = load_config()
    set_seed(config.get('seed', 42))

    path_manager = get_path_manager()
    model_dir = path_manager.get_model_path("happo")
    data_dir = path_manager.get_data_path("csv")
    plot_dir = path_manager.get_plot_path()
    log_dir = path_manager.get_log_path()

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = CloudEdgeDeviceEnv(config)

    state_dim = env.get_agent_state_dim()
    global_state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    num_agents = env.num_devices
    num_edges = env.num_edges

    happo_cfg = config.get('happo', {
        'lr_actor': 3e-4,
        'lr_critic': 1e-3,
        'gamma': 0.99,
        'lam': 0.95,
        'clip_range': 0.2,
        'kl_coeff': 0.5,
        'entropy_coeff': 0.01,
        'value_coeff': 0.5,
        'update_epochs': 4,
        'batch_size': 64,
        'max_episodes': 200,
        'max_steps': 200,
    })

    agents = [
        HAPPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            agent_idx=i,
            config=happo_cfg,
        )
        for i in range(num_agents)
    ]

    buffer = TrajectoryBuffer(num_agents)
    plotter = Plotter(plot_dir)
    metrics_tracker = MetricsTracker()

    max_episodes = happo_cfg.get('max_episodes', 200)
    max_steps = happo_cfg.get('max_steps', 200)

    all_actions = []
    training_losses = []
    episode_completion_rates = []

    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        step_means = []

        for step in range(max_steps):
            actions_env = []
            pending = []
            for i, agent in enumerate(agents):
                agent_state = env.extract_agent_state(state, i)
                action_raw, info = agent.select_action(agent_state, state, deterministic=False)
                edge_val = action_raw[-1]
                edge_id = int(np.clip(np.floor(edge_val * num_edges), 0, num_edges - 1))
                action_env = np.array([action_raw[0], action_raw[1], action_raw[2], edge_id], dtype=np.float32)
                actions_env.append(action_env)
                pending.append((i, agent_state, state, action_raw, info))

            actions = np.array(actions_env, dtype=np.float32)
            all_actions.append(actions)
            next_state, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            for i, agent_state, gstate, action_raw, info_map in pending:
                buffer.add(
                    i,
                    agent_state,
                    gstate,
                    action_raw,
                    rewards[i],
                    float(done),
                    info_map['log_prob'],
                    info_map['value'],
                    info_map['alpha'],
                    info_map['beta'],
                )

            state = next_state
            if info and 'has_task_list' in info:
                valid_rewards = [r for r, has_task in zip(rewards, info['has_task_list']) if has_task]
                if valid_rewards:
                    step_means.append(np.mean(valid_rewards))
            else:
                valid_rewards = [r for r in rewards if r > 0]
                if valid_rewards:
                    step_means.append(np.mean(valid_rewards))
                else:
                    step_means.append(np.mean(rewards))

            if done:
                break

        buffer.compute_advantages(gamma=happo_cfg['gamma'], lam=happo_cfg['lam'])
        step_losses = []
        for i, agent in enumerate(agents):
            batch = buffer.get_batch(i)
            if len(batch['states']) == 0:
                continue
            losses = agent.update(batch)
            step_losses.append(losses)
        buffer.clear()

        if step_losses:
            avg_policy_loss = float(np.mean([l['policy_loss'] for l in step_losses]))
            avg_value_loss = float(np.mean([l['value_loss'] for l in step_losses]))
            avg_kl = float(np.mean([l['kl'] for l in step_losses]))
            avg_entropy = float(np.mean([l['entropy'] for l in step_losses]))
            training_losses.append({
                'episode': episode,
                'policy_loss': avg_policy_loss,
                'value_loss': avg_value_loss,
                'kl': avg_kl,
                'entropy': avg_entropy,
            })

        if info and 'task_completion_stats' in info:
            episode_completion_rates.append(info['task_completion_stats'])

        if step_means:
            metrics_tracker.record_episode_reward(np.mean(step_means))

    if training_losses:
        save_training_metrics_csv(training_losses, os.path.join(data_dir, 'happo_training_metrics.csv'))
    plotter.plot_training_curves(training_losses, algo_name='HAPPO')


if __name__ == "__main__":
    train_happo()
