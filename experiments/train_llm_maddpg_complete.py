# experiments/train_llm_maddpg_complete.py
"""
完整版LLM+MADDPG训练脚本
实现：每step新任务 -> LLM咨询（交替） -> Agent动作 -> 执行 -> 每20步训练一次
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
import json
from tqdm import tqdm
import logging
from datetime import datetime
import matplotlib.pyplot as plt

from environment.cloud_edge_env import CloudEdgeDeviceEnv
from algos.maddpg_agent import MADDPGAgent
from algos.replay_buffer import ReplayBuffer
from llm_assistant.llm_client import LLMClient
from llm_assistant.prompt_builder import PromptBuilder
from llm_assistant.response_parser import ResponseParser
# from utils.plotting import plot_training_curves
# from utils.metrics import calculate_episode_metrics


def setup_logging():
    """设置日志"""
    log_dir = "results/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/train_complete_{timestamp}.log"
    
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
    # 使用正确的单个Agent状态维度
    state_dim = env.get_agent_state_dim()  # 20维：3(自己UE) + 10(所有ES) + 1(CS) + 6(自己任务)
    action_dim = 4  # [α1, α2, α3, edge_id]
    
    print(f"🔧 Agent配置信息:")
    print(f"  单个Agent状态维度: {state_dim}")
    print(f"  全局状态维度: {env.observation_space.shape[0]}")
    print(f"  动作维度: {action_dim}")
    print(f"  设备数量: {env.num_devices}")
    
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
        task_info = env.get_current_tasks_info()
        
        # 构建LLM提示
        prompt = prompt_builder.build_offloading_strategy_prompt(
            env_state=None,  # 环境状态信息
            device_info=device_info,
            edge_info=edge_info,
            cloud_info=cloud_info,
            tasks_info=task_info
        )
        
        # 咨询LLM
        response = llm_client.query(prompt)
        
        # 解析LLM响应
        llm_actions = response_parser.parse_unload_strategy(response, env.num_devices, env.num_edges, env.num_clouds)
        
        # 转换为标准格式 [α1, α2, α3, edge_id]
        formatted_actions = []
        for i in range(env.num_devices):
            if i < len(llm_actions):
                action = llm_actions[i]
                # 将三元分割决策转换为标准格式
                alpha1 = action.get('local_ratio', 0.0)
                alpha2 = action.get('edge_ratio', 0.0)
                alpha3 = action.get('cloud_ratio', 1.0)
                edge_id = action.get('edge_server_id', 0)
                
                formatted_actions.append([alpha1, alpha2, alpha3, edge_id])
            else:
                # 默认动作：全部在云端执行
                formatted_actions.append([0.0, 0.0, 1, 0])
                
        return formatted_actions
        
    except Exception as e:
        logger.warning(f"LLM咨询失败: {e}")
        # 返回默认动作
        default_actions = [[0, 0, 1, 0] for _ in range(env.num_devices)]
        return default_actions


def train_agents_from_buffer(agents, shared_buffer, logger, step_count):
    """从共享缓冲区训练所有Agent"""
    if len(shared_buffer) < 64:  # 最小批量要求
        logger.debug(f"缓冲区样本不足，跳过训练 (当前: {len(shared_buffer)})")
        return {}
    
    training_stats = {}
    
    for i, agent in enumerate(agents):
        try:
            stats = agent.train(all_agents=agents, replay_buffer=shared_buffer)
            training_stats[f'agent_{i}'] = stats
            
            if stats:
                logger.debug(f"Agent{i} 训练完成: "
                           f"critic_loss={stats.get('critic_loss', 0):.4f}, "
                           f"actor_loss={stats.get('actor_loss', 0):.4f}, "
                           f"distill_loss={stats.get('distill_loss', 0):.4f}")
                           
        except Exception as e:
            logger.error(f"Agent{i} 训练失败: {e}")
            
    logger.info(f"Step {step_count}: 完成所有Agent训练")
    return training_stats


def train_llm_maddpg_complete(config_path):
    """主训练函数 - 完整的训练流程"""
    # 设置日志
    logger = setup_logging()
    logger.info("开始完整版LLM+MADDPG训练")
    
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
    try:
        llm_client = LLMClient(
            api_key=config['llm'].get('api_key', ''),
            model_name=config['llm']['model_name'],
            server_url=config['llm'].get('base_url', 'http://10.200.1.35:8888/v1/completions'),
            timeout_connect=config['llm'].get('timeout', 120),
            timeout_read=config['llm'].get('read_timeout', 300),
            use_mock=config['llm'].get('use_mock_when_unavailable', False),
            config=config  # 传递完整配置以支持max_tokens等参数
        )
        prompt_builder = PromptBuilder()
        response_parser = ResponseParser()
        llm_available = True
    except Exception as e:
        logger.warning(f"LLM初始化失败，将跳过LLM咨询: {e}")
        llm_available = False
    
    # 训练参数
    num_episodes = config['training'].get('episodes', 1000)
    max_steps_per_episode = config['training'].get('max_steps_per_episode', 100)
    
    # 训练策略参数
    train_frequency = 20  # 每20步训练一次
    llm_episode_interval = 2  # 每2个Episode使用一次LLM（交替）
    
    # 记录指标
    episode_rewards = []
    episode_latencies = []
    episode_energies = []
    episode_completion_rates = []
    training_losses = []
    
    # 全局step计数器
    global_step_count = 0
    
    logger.info(f"开始训练，总Episodes: {num_episodes}")
    logger.info(f"训练策略: 每{train_frequency}步训练一次, 每{llm_episode_interval}个Episode使用LLM")
    
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        logger.info(f"\n{'='*80}")
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        logger.info(f"{'='*80}")
        
        # 判断是否在当前Episode使用LLM
        use_llm_this_episode = (episode % llm_episode_interval == 0) and llm_available
        logger.info(f"Episode {episode + 1}: {'使用LLM指导' if use_llm_this_episode else '纯MADDPG训练'}")
        
        # 重置环境
        state, _ = env.reset()
        episode_reward = 0
        episode_step_count = 0
        
        # Episode循环
        for step in range(max_steps_per_episode):
            global_step_count += 1
            episode_step_count += 1
            
            logger.debug(f"\nEpisode {episode + 1}, Step {step + 1}")

            # 当前step的状态
            current_state = state.copy()
            
            # 1. LLM咨询（如果当前Episode使用LLM）
            llm_expert_actions = None
            if use_llm_this_episode:
                llm_expert_actions = consult_llm_for_all_devices(
                    env, llm_client, prompt_builder, response_parser, logger
                )
                logger.debug(f"获取LLM专家动作: {len(llm_expert_actions)}个设备")
            
            # 2. Agent动作生成
            agent_actions = []
            print(f"\n📋 MADDPG智能体策略生成:")
            print(f"{'='*80}")
            
            for i, agent in enumerate(agents):
                # 使用正确的Agent状态提取方法
                agent_state = env.extract_agent_state(current_state, i)
                
                # 在训练早期增加探索
                add_noise = episode < num_episodes * 0.8
                action = agent.select_action(agent_state, add_noise=add_noise)
                agent_actions.append(action)
                
                # 详细输出MADDPG策略
                alpha1, alpha2, alpha3 = action[:3]
                edge_id = int(action[-1]) if len(action) >= 4 else 0
                
                # 归一化分割比例
                total = alpha1 + alpha2 + alpha3
                if total > 0:
                    alpha1_norm, alpha2_norm, alpha3_norm = alpha1/total, alpha2/total, alpha3/total
                else:
                    alpha1_norm, alpha2_norm, alpha3_norm = 1.0, 0.0, 0.0
                
                print(f"  🤖 Agent{i} (Device{i}) MADDPG策略:")
                print(f"    Agent状态维度: {len(agent_state)} (正确提取)")
                print(f"    原始动作: [α1={alpha1:.3f}, α2={alpha2:.3f}, α3={alpha3:.3f}, edge={action[-1]:.3f}]")
                print(f"    归一化分割: [本地:{alpha1_norm:.3f}, 边缘:{alpha2_norm:.3f}, 云端:{alpha3_norm:.3f}]")
                print(f"    目标边缘服务器: Edge{edge_id}")
                print(f"    探索模式: {'开启' if add_noise else '关闭'}")
                
                # # 策略分析
                # if alpha1_norm > 0.5:
                #     strategy = "本地优先策略"
                # elif alpha2_norm > 0.5:
                #     strategy = "边缘卸载策略"
                # elif alpha3_norm > 0.5:
                #     strategy = "云端卸载策略"
                # else:
                #     strategy = "混合卸载策略"
                # print(f"    策略类型: {strategy}")
                # print()

            agent_actions = np.array(agent_actions)
            
            # print(f"📊 MADDPG整体策略统计:")
            # print(f"  总设备数: {len(agent_actions)}")
            # print(f"  平均本地比例: {np.mean(agent_actions[:, 0]):.3f}")
            # print(f"  平均边缘比例: {np.mean(agent_actions[:, 1]):.3f}")
            # print(f"  平均云端比例: {np.mean(agent_actions[:, 2]):.3f}")
            # print(f"  最常选择的边缘服务器: Edge{int(np.round(np.mean(agent_actions[:, -1])))}")
            # print()
            
            # 3. 执行动作、环境交互
            next_state, rewards, terminated, truncated, info = env.step(
                agent_actions, llm_actions=llm_expert_actions
            )
            
            # 4. 存储经验到共享缓冲区
            for i in range(env.num_devices):
                # 使用正确的Agent状态提取方法
                agent_state = env.extract_agent_state(current_state, i)
                agent_next_state = env.extract_agent_state(next_state, i)
                
                shared_buffer.add(
                    state=agent_state,
                    action=agent_actions[i],
                    reward=rewards[i],
                    next_state=agent_next_state,
                    done=terminated or truncated,
                    llm_action=llm_expert_actions if use_llm_this_episode else None
                )
            
            # 5. 每20步训练一次
            if global_step_count % train_frequency == 0:
                logger.info(f"\n--- 第{global_step_count}步: 开始训练所有Agent ---")
                train_stats = train_agents_from_buffer(agents, shared_buffer, logger, global_step_count)
                training_losses.append(train_stats)
            
            # 更新状态和奖励
            state = next_state
            episode_reward += np.mean(rewards)
            
            # 检查终止条件
            if terminated or truncated:
                logger.info(f"Episode {episode + 1} 在第{step + 1}步终止")
                break
        
        # Episode结束，记录指标
        episode_rewards.append(episode_reward)
        
        # 从info中提取指标
        if info:
            avg_latency = np.mean(info.get('total_latencies', [0]))
            avg_energy = np.mean(info.get('total_energies', [0]))
            
            episode_latencies.append(avg_latency)
            episode_energies.append(avg_energy)
            
            # 计算任务完成率
            completion_stats = info.get('task_completion_stats', {})
            completion_rate = completion_stats.get('on_time_completion_rate', 0.0)
            overall_completion_rate = completion_stats.get('overall_completion_rate', 0.0)
            timeout_rate = completion_stats.get('timeout_rate', 0.0)
            
            episode_completion_rates.append(completion_rate)
            
            # 如果有截止时间违反信息，记录详细信息
            violations = info.get('deadline_violations', [])
            if violations:
                logger.info(f"Episode {episode + 1} 截止时间违反:")
                for v in violations[-3:]:  # 只显示最近3个违反
                    logger.info(f"  任务{v['task_id']}({v['task_type']}): 截止{v['deadline']:.1f}s, 实际{v['actual_time']:.1f}s, 超时{v['overtime']:.1f}s")
        
        # 定期打印进度
        if (episode + 1) % 10 == 0:
            recent_reward = np.mean(episode_rewards[-10:])
            recent_latency = np.mean(episode_latencies[-10:]) if episode_latencies else 0
            recent_energy = np.mean(episode_energies[-10:]) if episode_energies else 0
            recent_completion = np.mean(episode_completion_rates[-10:]) if episode_completion_rates else 0
            
            logger.info(f"\nEpisode {episode + 1} 阶段性总结:")
            logger.info(f"  最近10轮平均奖励: {recent_reward:.3f}")
            logger.info(f"  最近10轮平均时延: {recent_latency:.3f}s")
            logger.info(f"  最近10轮平均能耗: {recent_energy:.3f}J")
            logger.info(f"  最近10轮按时完成率: {recent_completion:.3f}")
            
            # 显示详细的任务完成统计
            if info and 'task_completion_stats' in info:
                comp_stats = info['task_completion_stats']
                logger.info(f"  详细完成率统计:")
                logger.info(f"    总完成率: {comp_stats.get('overall_completion_rate', 0):.3f}")
                logger.info(f"    按时完成率: {comp_stats.get('on_time_completion_rate', 0):.3f}")
                logger.info(f"    超时完成率: {comp_stats.get('timeout_rate', 0):.3f}")
                logger.info(f"    失败率: {comp_stats.get('failure_rate', 0):.3f}")
                logger.info(f"    平均超时时间: {comp_stats.get('avg_overtime', 0):.2f}s")
            
            logger.info(f"  全局步数: {global_step_count}")
            logger.info(f"  缓冲区大小: {len(shared_buffer)}")
        
        # 定期保存模型
        if (episode + 1) % 100 == 0:
            model_dir = "results/models"
            os.makedirs(model_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                model_path = f"{model_dir}/complete_agent_{i}_episode_{episode + 1}.pth"
                agent.save_model(model_path)
            logger.info(f"Episode {episode + 1}: 模型已保存")
    
    # 训练结束，保存最终模型
    logger.info("训练完成，保存最终模型...")
    final_model_dir = "results/final_models"
    os.makedirs(final_model_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        model_path = f"{final_model_dir}/complete_agent_{i}_final.pth"
        agent.save_model(model_path)
    
    # 保存训练统计数据
    logger.info("保存训练统计数据...")
    stats_dir = "results/stats"
    os.makedirs(stats_dir, exist_ok=True)
    
    training_data = {
        'episode_rewards': episode_rewards,
        'episode_latencies': episode_latencies,
        'episode_energies': episode_energies,
        'episode_completion_rates': episode_completion_rates,
        'training_losses': training_losses,
        'global_step_count': global_step_count,
        'config': config
    }
    
    with open(f"{stats_dir}/training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # 绘制训练曲线
    logger.info("绘制训练曲线...")
    plot_dir = "results/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 创建综合训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # 时延曲线
    if episode_latencies:
        axes[0, 1].plot(episode_latencies)
        axes[0, 1].set_title('Episode Average Latency')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Latency (s)')
        axes[0, 1].grid(True)
    
    # 能耗曲线
    if episode_energies:
        axes[1, 0].plot(episode_energies)
        axes[1, 0].set_title('Episode Average Energy')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Energy (J)')
        axes[1, 0].grid(True)
    
    # 任务完成率曲线
    if episode_completion_rates:
        axes[1, 1].plot(episode_completion_rates)
        axes[1, 1].set_title('Task Completion Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Completion Rate')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/complete_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"训练完成！")
    logger.info(f"总Episodes: {num_episodes}")
    logger.info(f"总Steps: {global_step_count}")
    logger.info(f"最终平均奖励: {np.mean(episode_rewards[-50:]):.3f}")
    logger.info(f"最终平均时延: {np.mean(episode_latencies[-50:]):.3f}s")
    logger.info(f"最终平均能耗: {np.mean(episode_energies[-50:]):.3f}J")
    logger.info(f"最终任务完成率: {np.mean(episode_completion_rates[-50:]):.3f}")
    logger.info(f"结果保存在 results/ 目录")
    
    return training_data


if __name__ == "__main__":
    config_path = "config.yaml"
    
    # 检查配置文件
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    try:
        results = train_llm_maddpg_complete(config_path)
        print("\n" + "="*60)
        print("🎉 训练成功完成！")
        print("="*60)
        print(f"最终结果:")
        print(f"  平均奖励: {np.mean(results['episode_rewards'][-50:]):.3f}")
        print(f"  平均时延: {np.mean(results['episode_latencies'][-50:]):.3f}s")
        print(f"  平均能耗: {np.mean(results['episode_energies'][-50:]):.3f}J")
        print(f"  任务完成率: {np.mean(results['episode_completion_rates'][-50:]):.3f}")
        print("="*60)
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 