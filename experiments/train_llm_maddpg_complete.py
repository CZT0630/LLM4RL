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
import random

from environment.cloud_edge_env import CloudEdgeDeviceEnv
from algos.maddpg_agent import MADDPGAgent
from algos.replay_buffer import ReplayBuffer
from llm_assistant.llm_client import LLMClient
from llm_assistant.prompt_builder import PromptBuilder
from llm_assistant.response_parser import ResponseParser
from utils.config import load_config
from utils.path_manager import get_path_manager
from utils.csv_saver import save_training_metrics_csv
from utils.metrics import MetricsTracker
# from utils.plotting import plot_training_curves
# from utils.metrics import calculate_episode_metrics


def setup_logging(path_manager):
    """设置日志"""
    log_dir = path_manager.get_log_path()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = path_manager.get_log_file_path(f"train_llm_maddpg_{timestamp}.log")
    
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
    
    # 🆕 读取退火策略配置
    llm_config = config.get('llm_maddpg', {})
    use_annealing = llm_config.get('use_annealing', False)
    
    if use_annealing:
        print(f"🔥 退火策略配置:")
        print(f"  启用状态: {use_annealing}")
        print(f"  初始权重: {llm_config.get('initial_llm_distill_weight', 0.8)}")
        print(f"  恒定权重: {llm_config.get('constant_llm_distill_weight', 0.15)}")
        print(f"  最终权重: {llm_config.get('final_llm_distill_weight', 0.0)}")
        print(f"  阶段1结束: {llm_config.get('stage1_end_episode', 300)} episodes")
        print(f"  阶段2结束: {llm_config.get('stage2_end_episode', 700)} episodes")
    else:
        print("ℹ️  退火策略未启用，使用固定蒸馏权重")
    
    for i in range(env.num_devices):
        # 构建Agent配置，包含退火策略和训练参数
        agent_config = config['training'].copy()
        agent_config.update(llm_config)  # 添加LLM配置包括退火策略
        
        agent = MADDPGAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=env.num_devices,
            agent_idx=i,
            config=agent_config
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
            stats = agent.train(agents, shared_buffer)
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
    # 加载配置
    config = load_config(config_path)
    
    # 使用路径管理器
    path_manager = get_path_manager()
    
    # 设置日志
    logger = setup_logging(path_manager)
    logger.info("开始完整版LLM+MADDPG训练")
    
    # 创建保存目录 - 使用正确的算法名称
    model_dir = path_manager.get_model_path("llm_maddpg")
    data_dir = path_manager.get_data_path("csv")
    json_dir = path_manager.get_data_path("json")
    plot_dir = path_manager.get_plot_path()
    log_dir = path_manager.get_log_path()
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    logger.info(f"🔧 [LLM+MADDPG] 路径配置:")
    logger.info(f"  模型保存路径: {model_dir}")
    logger.info(f"  数据保存路径: {data_dir}")
    logger.info(f"  实验目录: {path_manager.get_experiment_dir()}")
    
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
            config=config,
            use_mock=config['llm'].get('use_mock_when_unavailable', False)
        )
        prompt_builder = PromptBuilder()
        response_parser = ResponseParser()
        llm_available = True
    except Exception as e:
        logger.warning(f"LLM初始化失败，将跳过LLM咨询: {e}")
        llm_available = False
    
    # 🆕 从配置文件读取训练参数
    llm_config = config.get('llm_maddpg', {})
    
    # 训练参数
    num_episodes = llm_config.get('max_episodes', config['training'].get('episodes', 1000))
    max_steps_per_episode = llm_config.get('max_steps', config['training'].get('max_steps_per_episode', 200))
    
    # 🆕 训练策略参数（从配置文件读取）
    train_frequency = llm_config.get('train_frequency', 50)  # 每12步训练一次
    llm_episode_interval = llm_config.get('llm_episode_interval', 2)  # 每2个Episode使用一次LLM
    llm_distill_weight = llm_config.get('llm_distill_weight', 0.1)  # LLM知识蒸馏权重
    exploration_episodes = llm_config.get('exploration_episodes', int(num_episodes * 0.9))  # 探索轮数
    
    # 🆕 读取训练策略参数
    save_frequency = config['training']['save_frequency']
    log_frequency = config['training']['log_frequency']
    warm_up_episodes = config['training']['warm_up_episodes']
    
    # 记录指标 - 使用MetricsTracker类保持与其他算法一致
    metrics_tracker = MetricsTracker()
    training_losses = []
    
    # 全局step计数器
    global_step_count = 0
    
    logger.info(f"🔧 [LLM+MADDPG] 训练策略配置:")
    logger.info(f"  训练轮数: {num_episodes}")
    logger.info(f"  每轮最大步数: {max_steps_per_episode}")
    logger.info(f"  训练频率: 每{train_frequency}步训练一次")
    logger.info(f"  LLM指导间隔: 每{llm_episode_interval}个Episode使用LLM")
    logger.info(f"  预热轮数: {warm_up_episodes}")
    logger.info(f"  探索轮数: {exploration_episodes}")
    logger.info(f"  知识蒸馏权重: {llm_distill_weight}")
    
    for episode in tqdm(range(num_episodes), desc="训练进度"):
        logger.info(f"\n{'='*80}")
        logger.info(f"Episode {episode + 1}/{num_episodes}")
        logger.info(f"{'='*80}")
        
        # 🆕 更新所有Agent的LLM蒸馏权重（退火策略）
        for agent in agents:
            if hasattr(agent, 'update_llm_distill_weight'):
                old_weight = agent.llm_distill_weight
                new_weight = agent.update_llm_distill_weight(episode)
                
                # 显示权重变化（仅在变化时显示）
                if abs(old_weight - new_weight) > 0.001:
                    stage_name, stage_desc = agent.get_current_annealing_stage(episode)
                    logger.info(f"🔥 退火策略更新: {stage_name}")
                    logger.info(f"    {stage_desc}")
                    logger.info(f"    权重变化: {old_weight:.3f} → {new_weight:.3f}")
        
        # 🆕 显示当前退火阶段（第一个Agent的状态代表所有Agent）
        if hasattr(agents[0], 'get_current_annealing_stage'):
            stage_name, stage_desc = agents[0].get_current_annealing_stage(episode)
            current_weight = agents[0].llm_distill_weight
            logger.info(f"📊 当前蒸馏状态: {stage_desc} (当前权重: {current_weight:.3f})")
        
        # 🆕 判断当前训练阶段
        is_warm_up = episode < warm_up_episodes
        is_exploration = episode < exploration_episodes
        
        # 判断是否在当前Episode使用LLM
        use_llm_this_episode = (episode % llm_episode_interval == 0) and llm_available and not is_warm_up
        
        stage = "预热阶段" if is_warm_up else ("探索阶段" if is_exploration else "收敛阶段")
        llm_status = "使用LLM指导" if use_llm_this_episode else "纯MADDPG训练"
        logger.info(f"Episode {episode + 1}: {stage} - {llm_status}")
        
        # 重置环境
        state, _ = env.reset()
        step_means = []  # 新增：收集每个step所有智能体reward的均值
        episode_latencies = []  # 🆕 收集每步的延迟
        episode_energies = []   # 🆕 收集每步的能耗
        
        # Episode循环
        for step in range(max_steps_per_episode):
            global_step_count += 1
            
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
                
                # 🆕 优化探索策略
                if is_warm_up:
                    add_noise = True  # 预热期始终探索
                elif is_exploration:
                    add_noise = True  # 探索期始终探索
                else:
                    add_noise = episode < num_episodes * 0.9  # 收敛期适度探索
                    
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

            agent_actions = np.array(agent_actions)
            
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
            
            # 🆕 训练条件：非预热期 + 达到训练频率 + 缓冲区充足
            should_train = (not is_warm_up and 
                          global_step_count % train_frequency == 0 and
                          len(shared_buffer) > config['maddpg']['batch_size'])
            
            if should_train:
                logger.info(f"\n--- 第{global_step_count}步: 开始训练所有Agent ---")
                train_stats = train_agents_from_buffer(agents, shared_buffer, logger, global_step_count)
                training_losses.append(train_stats)
            
            # 更新状态和奖励
            state = next_state
            # 记录本step所有智能体reward的均值
            step_means.append(np.mean(rewards))
            
            # 🆕 从info中提取延迟和能耗，过滤零值
            if info:
                step_latencies = info.get('total_latencies', [])
                step_energies = info.get('total_energies', [])
                
                # 过滤掉零值，只保留有效的任务处理数据
                valid_latencies = [lat for lat in step_latencies if lat > 0]
                valid_energies = [eng for eng in step_energies if eng > 0]
                
                if valid_latencies:
                    episode_latencies.extend(valid_latencies)
                if valid_energies:
                    episode_energies.extend(valid_energies)
            
            # 检查终止条件
            if terminated or truncated:
                logger.info(f"Episode {episode + 1} 在第{step + 1}步终止")
                break
        
        # 统一episode reward计算方式
        episode_reward = np.mean(step_means) if step_means else 0.0
        
        # Episode结束，记录指标 - 使用MetricsTracker保持与其他算法一致
        # 使用实际任务完成率而不是固定值
        completion_stats = env.get_task_completion_rate()
        episode_completion_rate = completion_stats.get('on_time_completion_rate', 0.0)
        
        # 计算平均延迟和能耗
        avg_latency = np.mean(episode_latencies) if episode_latencies else 0.0
        avg_energy = np.mean(episode_energies) if episode_energies else 0.0
        
        # 使用MetricsTracker记录episode指标
        metrics_tracker.add_episode(episode_reward, avg_latency, avg_energy, use_llm_this_episode)
        
        # 🆕 优化进度打印 - 使用MetricsTracker获取统计数据
        if (episode + 1) % log_frequency == 0:
            avg_metrics = metrics_tracker.get_average_metrics(last_n=log_frequency)
            
            logger.info(f"\nEpisode {episode + 1} 阶段性总结 ({stage}):")
            logger.info(f"  最近{log_frequency}轮平均奖励: {avg_metrics['avg_reward']:.3f}")
            logger.info(f"  最近{log_frequency}轮平均时延: {avg_metrics['avg_delay']:.3f}s")
            logger.info(f"  最近{log_frequency}轮平均能耗: {avg_metrics['avg_energy']:.3f}J")
            logger.info(f"  LLM使用比例: {avg_metrics['llm_usage_ratio']:.3f}")
            
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
            
            # 🆕 收敛检测 - 使用MetricsTracker数据
            if not is_warm_up and len(metrics_tracker.episode_rewards) >= 50:
                recent_rewards = metrics_tracker.episode_rewards[-50:]
                reward_std = np.std(recent_rewards)
                convergence_threshold = config['training']['convergence_threshold']
                if reward_std < convergence_threshold:
                    logger.info(f"  🎯 检测到收敛！奖励标准差: {reward_std:.4f} < {convergence_threshold}")
        
        # 🆕 定期保存模型
        if (episode + 1) % save_frequency == 0:
            for i, agent in enumerate(agents):
                model_path = path_manager.get_model_file_path("llm_maddpg", f"agent_{i}_episode_{episode + 1}.pth")
                agent.save_model(model_path)
            logger.info(f"Episode {episode + 1}: 模型已保存到 {model_dir}")
    
    # 训练结束，保存最终模型
    logger.info("训练完成，保存最终模型...")
    for i, agent in enumerate(agents):
        final_model_path = path_manager.get_model_file_path("llm_maddpg", f"agent_{i}_final.pth")
        agent.save_model(final_model_path)
    
    # 保存训练统计数据
    logger.info("保存训练统计数据...")
    
    # 计算所有episode的平均完成率作为代表值
    final_completion_stats = env.get_task_completion_rate()
    avg_completion_rate = final_completion_stats.get('on_time_completion_rate', 0.0)
    
    training_data = {
        'episode_rewards': metrics_tracker.episode_rewards,
        'episode_latencies': metrics_tracker.episode_delays,
        'episode_energies': metrics_tracker.episode_energy,
        'episode_completion_rates': [avg_completion_rate] * len(metrics_tracker.episode_rewards),
        'training_losses': training_losses,
        'global_step_count': global_step_count,
        'config': config
    }
    
    # 保存JSON格式
    json_file = path_manager.get_data_file_path("json", f"llm_maddpg_training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # 🆕 保存核心指标到CSV表格
    logger.info("保存训练指标到CSV表格...")
    try:
        csv_filepath = save_training_metrics_csv(
            episode_rewards=metrics_tracker.episode_rewards,
            episode_latencies=metrics_tracker.episode_delays,
            episode_energies=metrics_tracker.episode_energy,
            episode_completion_rates=[avg_completion_rate] * len(metrics_tracker.episode_rewards),
            algorithm_name="LLM_MADDPG",
            save_dir=data_dir
        )
        logger.info(f"✅ CSV文件已保存: {csv_filepath}")
    except Exception as e:
        logger.error(f"❌ 保存CSV文件失败: {e}")
    
    # 绘制训练曲线
    logger.info("绘制训练曲线...")
    
    # 创建综合训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 奖励曲线
    axes[0, 0].plot(metrics_tracker.episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # 时延曲线
    if metrics_tracker.episode_delays:
        axes[0, 1].plot(metrics_tracker.episode_delays)
        axes[0, 1].set_title('Episode Average Latency')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Latency (s)')
        axes[0, 1].grid(True)
    
    # 能耗曲线
    if metrics_tracker.episode_energy:
        axes[1, 0].plot(metrics_tracker.episode_energy)
        axes[1, 0].set_title('Episode Average Energy')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Energy (J)')
        axes[1, 0].grid(True)
    
    # LLM使用比例曲线
    if metrics_tracker.llm_used:
        axes[1, 1].plot(metrics_tracker.llm_used)
        axes[1, 1].set_title('LLM Usage Ratio')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('LLM Usage')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_file = path_manager.get_plot_file_path("llm_maddpg_training_curves.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"训练完成！")
    logger.info(f"总Episodes: {num_episodes}")
    logger.info(f"总Steps: {global_step_count}")
    logger.info(f"最终平均奖励: {np.mean(metrics_tracker.episode_rewards[-50:]):.3f}")
    logger.info(f"最终平均时延: {np.mean(metrics_tracker.episode_delays[-50:]):.3f}s")
    logger.info(f"最终平均能耗: {np.mean(metrics_tracker.episode_energy[-50:]):.3f}J")
    logger.info(f"最终LLM使用比例: {np.mean(metrics_tracker.llm_used[-50:]):.3f}")
    logger.info(f"模型保存路径: {model_dir}")
    logger.info(f"数据保存路径: {data_dir}")
    logger.info(f"📁 所有结果保存在: {path_manager.get_experiment_dir()}")
    
    # 返回完整的训练结果
    return {
        'algorithm': 'LLM+MADDPG',
        'episode_rewards': metrics_tracker.episode_rewards,
        'episode_latencies': metrics_tracker.episode_delays,
        'episode_energies': metrics_tracker.episode_energy,
        'episode_completion_rates': [avg_completion_rate] * len(metrics_tracker.episode_rewards),
        'training_losses': training_losses,
        'global_step_count': global_step_count,
        'config': config,
        'model_paths': [path_manager.get_model_file_path("llm_maddpg", f"agent_{i}_final.pth") for i in range(len(agents))]
    }


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