# experiments/test_llm_maddpg.py
import numpy as np
import torch
import random
import os
import gymnasium as gym
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from algos.maddpg_agent import MADDPGAgent
from utils.config import load_config
from utils.path_manager import get_path_manager
from utils.csv_saver import save_test_results_csv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_llm_maddpg(model_path=None, config=None):
    """
    测试经过LLM+MADDPG训练的Agent在没有LLM指导下的表现
    这相当于测试Agent通过知识蒸馏学到的决策能力
    """
    # 加载配置
    if config is None:
        config = load_config()
    set_seed(config.get('seed', 42))

    # 使用路径管理器
    path_manager = get_path_manager()
    
    # 如果没有指定模型路径，自动从路径管理器获取
    if model_path is None:
        model_path = path_manager.get_model_path("llm_maddpg")

    print(f"🔧 [LLM+MADDPG测试] 配置信息:")
    print(f"  模型路径: {model_path}")
    print(f"  测试结果保存路径: {path_manager.get_test_results_path()}")
    print(f"  🧠 注意: 使用LLM+MADDPG训练的模型，测试时不提供LLM指导")

    # 创建环境（不需要LLM客户端）
    env = CloudEdgeDeviceEnv(config['environment'])

    # 创建MADDPG智能体
    # 使用正确的单个Agent状态维度  
    state_dim = env.get_agent_state_dim()  # 20维：3(自己UE) + 10(所有ES) + 1(CS) + 6(自己任务)
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[1] + 1  # 目标节点数量
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

        # 尝试多种可能的模型文件格式
        possible_model_paths = [
            # LLM+MADDPG完整模型格式 (推荐)
            f"{model_path}/agent_{i}_final.pth",
            f"{model_path}/agent_{i}.pth",
            os.path.join(model_path, f"agent_{i}_final.pth"),
            os.path.join(model_path, f"agent_{i}.pth"),
            # 分离格式 (备用)
            f"{model_path}/actor_agent_{i}_final.pth",
            f"{model_path}/actor_agent_{i}.pth",
            os.path.join(model_path, f"actor_agent_{i}_final.pth"),
            os.path.join(model_path, f"actor_agent_{i}.pth")
        ]

        model_loaded = False
        
        # 首先尝试加载完整模型格式
        for model_path_full in possible_model_paths[:4]:
            if os.path.exists(model_path_full):
                try:
                    if hasattr(agent, 'load_model'):
                        agent.load_model(model_path_full)
                        print(f"  ✅ 加载Agent {i} 完整模型: {model_path_full}")
                        model_loaded = True
                        break
                except Exception as e:
                    print(f"  ❌ 加载完整模型失败 {model_path_full}: {e}")
        
        # 如果完整模型加载失败，尝试分离格式
        if not model_loaded:
            actor_path = f"{model_path}/actor_agent_{i}_final.pth"
            critic_path = f"{model_path}/critic_agent_{i}_final.pth"
            
            if os.path.exists(actor_path):
                try:
                    agent.actor.load_state_dict(torch.load(actor_path))
                    agent.actor.eval()
                    print(f"  ✅ 加载Agent {i} Actor (分离格式): {actor_path}")
                    model_loaded = True
                except Exception as e:
                    print(f"  ❌ 加载Actor失败: {e}")

            if os.path.exists(critic_path):
                try:
                    agent.critic.load_state_dict(torch.load(critic_path))
                    agent.critic.eval()
                    print(f"  ✅ 加载Agent {i} Critic (分离格式): {critic_path}")
                except Exception as e:
                    print(f"  ❌ 加载Critic失败: {e}")

        if model_loaded:
            model_loaded_count += 1
        else:
            print(f"  ⚠️  Agent {i} 模型加载失败")

        agents.append(agent)

    if model_loaded_count == 0:
        print("❌ 没有成功加载任何模型，无法进行测试")
        return [], [], []
    
    print(f"✅ 成功加载 {model_loaded_count}/{num_agents} 个Agent的模型")

    # 测试参数
    num_episodes = config['testing']['num_episodes']
    max_steps = config['maddpg']['max_steps']

    # 记录所有episode的指标
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

        if (episode + 1) % 20 == 0:
            print(f"\n[LLM+MADDPG纯Agent模式] 测试 Episode {episode + 1}/{num_episodes}")

        for step in range(max_steps):
            # 选择动作 - 关键修改：不使用LLM指导
            actions = []
            
            for i, agent in enumerate(agents):
                # 使用正确的Agent状态提取
                agent_state = env.extract_agent_state(state, i)
                
                # 🔥 关键修改：llm_advice=None，测试Agent的知识蒸馏学习效果
                agent_action = agent.select_action(agent_state, add_noise=False, llm_advice=None)
                actions.append(agent_action)

            # 执行动作
            next_state, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            # 详细信息输出（仅部分episode）
            if (episode + 1) % 50 == 0 and step % 20 == 0:
                print(f"  步骤 {step} (纯Agent决策):")
                for i, (action, reward) in enumerate(zip(actions, rewards)):
                    if len(action) >= 4:  # 新格式 [α1, α2, α3, edge_id]
                        alpha1, alpha2, alpha3, edge_id = action[:4]
                        print(f"    设备 {i}: 本地={alpha1:.2f}, 边缘={alpha2:.2f}, 云端={alpha3:.2f}, 目标边缘={int(edge_id)}, 奖励={reward:.2f}")
                    else:  # 旧格式 [offload_ratio, target_node]
                        offload_ratio = action[0]
                        target_node = int(action[1])
                        target_name = "本地"
                        if target_node >= 1 and target_node <= env.num_edges:
                            target_name = f"边缘服务器 {target_node - 1}"
                        elif target_node > env.num_edges:
                            target_name = "云端"
                        print(f"    设备 {i}: 卸载比例={offload_ratio:.2f}, 目标={target_name}, 奖励={reward:.2f}")

            # 累加能耗、时延、资源利用率
            episode_energy += sum(info['energies'])
            episode_delay += sum(info['delays'])
            episode_util += sum(info['utilizations'])
            step_count += 1

            state = next_state
            episode_reward += sum(rewards)

            if done:
                if (episode + 1) % 20 == 0:
                    print(f"  Episode {episode + 1} 完成，总奖励: {episode_reward:.2f}")
                break

        avg_energy = episode_energy / num_agents
        avg_delay = episode_delay / num_agents
        avg_util = episode_util / (num_agents * step_count) if step_count > 0 else 0
        all_episode_energy.append(avg_energy)
        all_episode_delay.append(avg_delay)
        all_episode_util.append(avg_util)
        
        if (episode + 1) % 10 == 0:
            print(f"[LLM+MADDPG] Episode {episode + 1}/{num_episodes} - "
                  f"能耗: {avg_energy:.4f}, 利用率: {avg_util:.4f}, 时延: {avg_delay:.4f}")

    # 计算最终统计结果
    final_energy = np.mean(all_episode_energy)
    final_util = np.mean(all_episode_util)
    final_delay = np.mean(all_episode_delay)

    print(f"\n📊 [LLM+MADDPG纯Agent模式] 测试完成!")
    print(f"  平均能耗: {final_energy:.4f}")
    print(f"  平均资源利用率: {final_util:.4f}")
    print(f"  平均任务时延: {final_delay:.4f}")
    print("📋 结果说明: 此结果展示了Agent通过知识蒸馏学到的决策能力")
    
    # 保存测试结果
    try:
        test_results = {
            'algorithm': 'LLM+MADDPG (Pure Agent)',
            'description': 'LLM+MADDPG训练的Agent在无LLM指导下的测试结果',
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
        result_file = path_manager.get_test_results_file_path("llm_maddpg_pure_agent_test_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"  ✅ 测试结果保存至: {result_file}")
        
        # 保存CSV格式的测试结果
        csv_file = save_test_results_csv(
            test_results=test_results,
            algorithm_name="LLM_MADDPG_Pure_Agent",
            save_dir=path_manager.get_test_results_path()
        )
        print(f"  ✅ CSV结果保存至: {csv_file}")
        
    except Exception as e:
        print(f"  ❌ 保存测试结果失败: {e}")
        
    return all_episode_energy, all_episode_util, all_episode_delay

if __name__ == "__main__":
    test_llm_maddpg()