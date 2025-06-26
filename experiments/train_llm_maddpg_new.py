"""
新版本LLM+MADDPG训练脚本
实现：每step生成新任务 -> LLM咨询 -> Agent动作生成 -> 动作执行 -> 存储经验 -> Episode结束后统一训练
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from tqdm import tqdm
import logging
from datetime import datetime

from environment.cloud_edge_env import CloudEdgeDeviceEnv
from algos.maddpg_agent import MADDPGAgent
from algos.replay_buffer import ReplayBuffer
from llm_assistant.llm_client import LLMClient
from llm_assistant.prompt_builder import PromptBuilder
from llm_assistant.response_parser import ResponseParser
from utils.plotting import plot_training_curves
from utils.metrics import calculate_episode_metrics


def setup_logging():
    """设置日志"""
    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/train_llm_maddpg_new_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_agents(env, config):
    """创建所有Agent"""
    agents = []
    state_dim = env.observation_space.shape[0] // env.num_devices  # 每个Agent的状态维度
    action_dim = 4  # [α1, α2, α3, edge_id]
    
    for i in range(env.num_devices):
        agent = MADDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=env.num_devices,
            agent_idx=i,
            config=config['training']
        )
        agents.append(agent)
        
    return agents


def consult_llm_for_all_devices(env, llm_client, prompt_builder, response_parser, logger):
    """为所有设备咨询LLM获取专家动作"""
    try:
        # 获取系统状态信息
        device_info = env.get_device_info()
        edge_info = env.get_edge_info()
        cloud_info = env.get_cloud_info()
        
        # 获取当前任务信息
        task_info = []
        for i, task in enumerate(env.current_tasks):
            task_info.append({
                "task_id": i,
                "task_type": task.task_type,
                "data_size": task.data_size_mb,
                "cpu_cycles": task.cpu_cycles / 1e9,  # 转换为Gcycles便于理解
                "deadline": task.deadline
            })
        
        # 构建LLM提示
        prompt = prompt_builder.build_decision_prompt(
            device_info=device_info,
            edge_info=edge_info,
            cloud_info=cloud_info,
            task_info=task_info
        )
        
        # 咨询LLM
        response = llm_client.get_completion(prompt)
        
        # 解析LLM响应
        llm_actions = response_parser.parse_offloading_decisions(response)
        
        # 转换为标准格式 [α1, α2, α3, edge_id]
        formatted_actions = []
        for i in range(env.num_devices):
            if i < len(llm_actions):
                action = llm_actions[i]
                # 将三元分割决策转换为标准格式
                alpha1 = action.get('local_ratio', 0.5)
                alpha2 = action.get('edge_ratio', 0.3)
                alpha3 = action.get('cloud_ratio', 0.2)
                edge_id = action.get('edge_server_id', 0)
                
                formatted_actions.append([alpha1, alpha2, alpha3, edge_id])
            else:
                # 默认动作：50%本地，30%边缘，20%云端，边缘服务器0
                formatted_actions.append([0.5, 0.3, 0.2, 0])
                
        logger.debug(f"LLM专家动作: {formatted_actions}")
        return formatted_actions
        
    except Exception as e:
        logger.warning(f"LLM咨询失败: {e}")
        # 返回默认动作
        default_actions = [[0.5, 0.3, 0.2, 0] for _ in range(env.num_devices)]
        return default_actions


def train_llm_maddpg(config_path):
    """主训练函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("开始新版本LLM+MADDPG训练")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建环境
    logger.info("创建云边端三层架构环境...")
    env = CloudEdgeDeviceEnv(config)
    
    # 创建Agent
    logger.info("创建MADDPG智能体...")
    agents = create_agents(env, config)
    
    # 创建共享经验回放缓冲区
    buffer_size = config['training'].get('buffer_size', 100000)
    shared_buffer = ReplayBuffer(buffer_size)
    
    # 创建LLM组件
    logger.info("初始化LLM助手...")
    llm_client = LLMClient(config.get('llm_config', {}))
    prompt_builder = PromptBuilder()
    response_parser = ResponseParser()
    
    # 训练参数
    num_episodes = config['training'].get('episodes', 1000)
    max_steps_per_episode = config['training'].get('max_steps_per_episode', 100)
    train_start_episode = config['training'].get('train_start_episode', 10)
    llm_consult_frequency = config['training'].get('llm_consult_frequency', 1)  # 每几个step咨询一次LLM
    
    # 记录指标
    episode_rewards = []
    episode_latencies = []
    episode_energies = []
    
    logger.info(f"开始训练，总Episodes: {num_episodes}")
    
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        logger.info(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        
        # 重置环境
        state, _ = env.reset()
        episode_reward = 0
        episode_step_count = 0
        
        # Episode内的所有经验（在Episode结束后存储到共享缓冲区）
        episode_experiences = []
        
        # Episode循环
        for step in range(max_steps_per_episode):
            logger.debug(f"  Step {step + 1}/{max_steps_per_episode}")
            
            # 1. 每个step都有新任务（环境已自动生成）
            current_state = state.copy()
            
            # 2. 每个step都咨询LLM获取专家动作
            if step % llm_consult_frequency == 0:
                llm_expert_actions = consult_llm_for_all_devices(
                    env, llm_client, prompt_builder, response_parser, logger
                )
            
            # 3. 每个Agent根据状态生成动作
            agent_actions = []
            for i, agent in enumerate(agents):
                # 提取当前Agent的状态
                agent_state = current_state[i * (len(current_state) // env.num_devices):
                                         (i + 1) * (len(current_state) // env.num_devices)]
                action = agent.select_action(agent_state, add_noise=(episode < num_episodes * 0.8))
                agent_actions.append(action)
            
            agent_actions = np.array(agent_actions)
            
            # 4. 执行动作
            next_state, rewards, terminated, truncated, info = env.step(
                agent_actions, llm_actions=llm_expert_actions
            )
            
            # 5. 存储经验到Episode缓冲区
            episode_experiences.append({
                'state': current_state,
                'action': agent_actions,
                'reward': rewards,
                'next_state': next_state,
                'done': terminated or truncated,
                'llm_action': llm_expert_actions
            })
            
            # 更新状态和奖励
            state = next_state
            episode_reward += np.mean(rewards)
            episode_step_count += 1
            
            # 检查终止条件
            if terminated or truncated:
                logger.debug(f"    Episode终止: terminated={terminated}, truncated={truncated}")
                break
        
        # 6. Episode结束后，将所有经验添加到共享缓冲区
        for exp in episode_experiences:
            # 为每个Agent添加经验
            for i in range(env.num_devices):
                agent_state = exp['state'][i * (len(exp['state']) // env.num_devices):
                                        (i + 1) * (len(exp['state']) // env.num_devices)]
                agent_next_state = exp['next_state'][i * (len(exp['next_state']) // env.num_devices):
                                                   (i + 1) * (len(exp['next_state']) // env.num_devices)]
                
                shared_buffer.add(
                    state=agent_state,
                    action=exp['action'][i],
                    reward=exp['reward'][i],
                    next_state=agent_next_state,
                    done=exp['done'],
                    llm_action=exp['llm_action']
                )
        
        # 7. Episode结束后统一训练所有Agent
        if episode >= train_start_episode and len(shared_buffer) >= config['training'].get('batch_size', 64):
            logger.debug(f"  训练所有Agent (缓冲区大小: {len(shared_buffer)})")
            for agent in agents:
                agent.train(replay_buffer=shared_buffer)
        
        # 记录Episode指标
        episode_rewards.append(episode_reward)
        
        # 计算Episode指标
        if info:
            avg_latency = np.mean(info.get('total_latencies', [0]))
            avg_energy = np.mean(info.get('total_energies', [0]))
            episode_latencies.append(avg_latency)
            episode_energies.append(avg_energy)
        
        # 定期打印进度
        if (episode + 1) % 50 == 0:
            recent_reward = np.mean(episode_rewards[-50:])
            recent_latency = np.mean(episode_latencies[-50:]) if episode_latencies else 0
            recent_energy = np.mean(episode_energies[-50:]) if episode_energies else 0
            
            logger.info(f"Episode {episode + 1}: "
                       f"平均奖励={recent_reward:.3f}, "
                       f"平均时延={recent_latency:.3f}s, "
                       f"平均能耗={recent_energy:.3f}J")
        
        # 定期保存模型
        if (episode + 1) % 200 == 0:
            model_dir = "results/models"
            os.makedirs(model_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                model_path = f"{model_dir}/agent_{i}_episode_{episode + 1}.pth"
                agent.save_model(model_path)
            logger.info(f"模型已保存到 {model_dir}")
    
    # 训练结束，保存最终模型
    logger.info("训练完成，保存最终模型...")
    final_model_dir = "results/final_models"
    os.makedirs(final_model_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        model_path = f"{final_model_dir}/agent_{i}_final.pth"
        agent.save_model(model_path)
    
    # 绘制训练曲线
    logger.info("绘制训练曲线...")
    plot_dir = "results/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_training_curves(
        episode_rewards,
        episode_latencies,
        episode_energies,
        save_path=f"{plot_dir}/llm_maddpg_new_training_curves.png"
    )
    
    logger.info(f"训练完成！结果保存在 results/ 目录")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_latencies': episode_latencies,
        'episode_energies': episode_energies
    }


if __name__ == "__main__":
    config_path = "config.yaml"
    
    # 检查配置文件
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    try:
        results = train_llm_maddpg(config_path)
        print("训练成功完成！")
        print(f"最终平均奖励: {np.mean(results['episode_rewards'][-100:]):.3f}")
        print(f"最终平均时延: {np.mean(results['episode_latencies'][-100:]):.3f}s")
        print(f"最终平均能耗: {np.mean(results['episode_energies'][-100:]):.3f}J")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 