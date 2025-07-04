import numpy as np
import torch
import gymnasium as gym
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from llm_assistant.llm_client import LLMClient
from llm_assistant.response_parser import ResponseParser
from utils.config import load_config
from utils.path_manager import get_path_manager
from utils.csv_saver import save_test_results_csv

def test_llm(model_path=None, config=None):
    if config is None:
        config = load_config()
    
    # 使用路径管理器
    path_manager = get_path_manager()
    
    print(f"🔧 [纯LLM测试] 配置信息:")
    print(f"  测试结果保存路径: {path_manager.get_test_results_path()}")
    print(f"  🧠 注意: 纯LLM测试直接使用LLM进行决策")
    
    env = CloudEdgeDeviceEnv(config['environment'])
    
    llm_client = LLMClient(
        config=config,
        use_mock=config['llm'].get('use_mock_when_unavailable', True)
    )
    
    num_episodes = config['testing']['num_episodes']
    max_steps = config['maddpg']['max_steps']
    num_agents = env.num_devices
    all_episode_energy = []
    all_episode_delay = []
    all_episode_util = []
    
    print(f"  设备数量: {num_agents}")
    print(f"  测试轮数: {num_episodes}")
    print(f"  每轮最大步数: {max_steps}")
    
    print(f"\n🧪 开始测试，共{num_episodes}轮...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_energy = 0
        episode_delay = 0
        episode_util = 0
        step_count = 0
        
        for step in range(max_steps):
            device_info = [{"cpu": d.cpu_frequency} for d in env.devices]
            edge_info = [{"cpu": e.cpu_frequency} for e in env.edge_servers]
            cloud_info = [{"cpu": c.cpu_frequency} for c in env.cloud_servers]
            
            # 获取格式1的LLM策略
            llm_strategies = llm_client.get_unload_strategy(state, device_info, edge_info, cloud_info)
            
            # 解析LLM响应（格式1）
            parsed_strategies = ResponseParser.parse_unload_strategy(
                llm_strategies,
                env.num_devices,
                env.num_edges,
                env.num_clouds
            )
            
            # 将格式1策略转换为环境可接受的动作格式 [α1, α2, α3, edge_id]
            action_list = []
            for device_idx in range(env.num_devices):
                device_strategy = next(
                    (s for s in parsed_strategies if s["device_id"] == device_idx), 
                    {
                        "device_id": device_idx,
                        "local_ratio": 1.0,
                        "edge_ratio": 0.0,
                        "cloud_ratio": 0.0,
                        "target_edge": 0
                    }
                )
                
                # 转换为环境动作格式
                action = [
                    device_strategy["local_ratio"],   # α1: 本地执行比例
                    device_strategy["edge_ratio"],    # α2: 边缘执行比例  
                    device_strategy["cloud_ratio"],   # α3: 云端执行比例
                    device_strategy["target_edge"]    # edge_id: 目标边缘服务器
                ]
                action_list.append(action)

            actions = np.array(action_list)

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
            print(f"[纯LLM] Episode {episode + 1}/{num_episodes} - "
                  f"能耗: {avg_energy:.4f}, 利用率: {avg_util:.4f}, 时延: {avg_delay:.4f}")

    # 计算最终统计结果
    final_energy = np.mean(all_episode_energy)
    final_util = np.mean(all_episode_util)
    final_delay = np.mean(all_episode_delay)

    print(f"\n📊 [纯LLM] 测试完成!")
    print(f"  平均能耗: {final_energy:.4f}")
    print(f"  平均资源利用率: {final_util:.4f}")
    print(f"  平均任务时延: {final_delay:.4f}")
    
    # 保存测试结果
    try:
        test_results = {
            'algorithm': 'LLM',
            'description': '纯LLM算法的测试结果',
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
        result_file = path_manager.get_test_results_file_path("llm_test_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"  ✅ 测试结果保存至: {result_file}")
        
        # 保存CSV格式的测试结果
        csv_file = save_test_results_csv(
            test_results=test_results,
            algorithm_name="LLM",
            save_dir=path_manager.get_test_results_path()
        )
        print(f"  ✅ CSV结果保存至: {csv_file}")
        
    except Exception as e:
        print(f"  ❌ 保存测试结果失败: {e}")
        
    return all_episode_energy, all_episode_util, all_episode_delay

if __name__ == "__main__":
    test_llm() 