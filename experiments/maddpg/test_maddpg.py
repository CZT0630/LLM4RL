import numpy as np
import torch
import os
import gymnasium as gym
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from algos.maddpg.maddpg_agent import MADDPGAgent
from utils.config import load_config
from utils.path_manager import get_path_manager
from utils.csv_saver import save_test_results_csv

def test_maddpg(model_path=None, config=None):
    if config is None:
        config = load_config()
    
    # 使用路径管理器
    path_manager = get_path_manager()
    
    # 如果没有指定模型路径，自动从路径管理器获取
    if model_path is None:
        model_path = path_manager.get_model_path("maddpg")
    
    print(f"🔧 [MADDPG测试] 配置信息:")
    print(f"  模型路径: {model_path}")
    print(f"  测试结果保存路径: {path_manager.get_test_results_path()}")
    
    env = CloudEdgeDeviceEnv(config['environment'])
    
    # 使用正确的单个Agent状态维度
    state_dim = env.get_agent_state_dim()  # 20维：3(自己UE) + 10(所有ES) + 1(CS) + 6(自己任务)
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[1] + 1
    num_agents = env.num_devices
    
    print(f"  单个Agent状态维度: {state_dim}")
    print(f"  全局状态维度: {env.observation_space.shape[0]}")
    print(f"  动作维度: {action_dim}")
    print(f"  设备数量: {num_agents}")
    
    agents = []
    model_loaded_count = 0
    for i in range(num_agents):
        agent = MADDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            num_agents=num_agents,
            agent_idx=i
        )
        
        # 尝试多种可能的模型文件名格式
        possible_actor_paths = [
            f"{model_path}/actor_agent_{i}_final.pth",
            f"{model_path}/actor_agent_{i}.pth",
            os.path.join(model_path, f"actor_agent_{i}_final.pth"),
            os.path.join(model_path, f"actor_agent_{i}.pth")
        ]
        
        possible_critic_paths = [
            f"{model_path}/critic_agent_{i}_final.pth",
            f"{model_path}/critic_agent_{i}.pth", 
            os.path.join(model_path, f"critic_agent_{i}_final.pth"),
            os.path.join(model_path, f"critic_agent_{i}.pth")
        ]
        
        # 加载Actor模型
        actor_loaded = False
        for actor_path in possible_actor_paths:
            if os.path.exists(actor_path):
                try:
                    agent.actor.load_state_dict(torch.load(actor_path))
                    agent.actor.eval()
                    print(f"  ✅ 加载Agent {i} Actor: {actor_path}")
                    actor_loaded = True
                    break
                except Exception as e:
                    print(f"  ❌ 加载Actor失败 {actor_path}: {e}")
        
        # 加载Critic模型
        critic_loaded = False
        for critic_path in possible_critic_paths:
            if os.path.exists(critic_path):
                try:
                    agent.critic.load_state_dict(torch.load(critic_path))
                    agent.critic.eval()
                    print(f"  ✅ 加载Agent {i} Critic: {critic_path}")
                    critic_loaded = True
                    break
                except Exception as e:
                    print(f"  ❌ 加载Critic失败 {critic_path}: {e}")
        
        if actor_loaded and critic_loaded:
            model_loaded_count += 1
        else:
            print(f"  ⚠️  Agent {i} 模型加载不完整")
            
        agents.append(agent)
    
    if model_loaded_count == 0:
        print("❌ 没有成功加载任何模型，无法进行测试")
        return [], [], []
    
    print(f"✅ 成功加载 {model_loaded_count}/{num_agents} 个Agent的模型")
        
    num_episodes = config['testing']['num_episodes']
    max_steps = config['maddpg']['max_steps']
    all_episode_energy = []
    all_episode_delay = []
    all_episode_util = []
    
    print(f"\n🧪 开始测试，共{num_episodes}轮...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_delay = 0
        episode_util = 0
        step_count = 0
        
        for step in range(max_steps):
            # 使用正确的Agent状态提取
            actions = []
            for i, agent in enumerate(agents):
                agent_state = env.extract_agent_state(state, i)
                action = agent.select_action(agent_state, llm_advice=None, add_noise=False)
                actions.append(action)
            
            next_state, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            episode_energy += sum(info['energies'])
            episode_delay += sum(info['delays'])
            episode_util += sum(info['utilizations'])
            step_count += 1
            state = next_state
            episode_reward += sum(rewards)
            
            if done:
                break
                
        avg_energy = episode_energy / num_agents
        avg_delay = episode_delay / num_agents
        avg_util = episode_util / (num_agents * step_count) if step_count > 0 else 0
        all_episode_energy.append(avg_energy)
        all_episode_delay.append(avg_delay)
        all_episode_util.append(avg_util)
        
        if (episode + 1) % 10 == 0:
            print(f"[MADDPG] Episode {episode + 1}/{num_episodes} - "
                  f"能耗: {avg_energy:.4f}, 利用率: {avg_util:.4f}, 时延: {avg_delay:.4f}")
    
    # 计算最终统计结果
    final_energy = np.mean(all_episode_energy)
    final_util = np.mean(all_episode_util)
    final_delay = np.mean(all_episode_delay)
    
    print(f"\n📊 [MADDPG] 测试完成!")
    print(f"  平均能耗: {final_energy:.4f}")
    print(f"  平均资源利用率: {final_util:.4f}")
    print(f"  平均任务时延: {final_delay:.4f}")
    
    # 保存测试结果
    try:
        test_results = {
            'algorithm': 'MADDPG',
            'num_episodes': num_episodes,
            'avg_energy': final_energy,
            'avg_utilization': final_util,
            'avg_delay': final_delay,
            'energy_std': np.std(all_episode_energy),
            'utilization_std': np.std(all_episode_util),
            'delay_std': np.std(all_episode_delay),
            'all_episode_energy': all_episode_energy,
            'all_episode_utilization': all_episode_util,
            'all_episode_delay': all_episode_delay
        }
        
        # 保存到测试结果目录
        import json
        result_file = path_manager.get_test_results_file_path("maddpg_test_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"  ✅ 测试结果保存至: {result_file}")
        
        # 保存CSV格式的测试结果
        csv_file = save_test_results_csv(
            test_results=test_results,
            algorithm_name="MADDPG",
            save_dir=path_manager.get_test_results_path()
        )
        print(f"  ✅ CSV结果保存至: {csv_file}")
        
    except Exception as e:
        print(f"  ❌ 保存测试结果失败: {e}")
        
    return all_episode_energy, all_episode_util, all_episode_delay

if __name__ == "__main__":
    test_maddpg()