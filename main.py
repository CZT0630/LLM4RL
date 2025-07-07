#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM+MADDPG 云边端计算卸载系统
统一入口文件 - 支持完整的训练、测试、对比功能
"""

import os
import sys
import json
import time
import yaml
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import torch

# 导入训练和测试模块
from experiments.train_llm_maddpg_complete import train_llm_maddpg_complete
from experiments.train_maddpg import train_maddpg
from experiments.train_llm import train_llm
from experiments.test_llm_maddpg import test_llm_maddpg
from experiments.test_maddpg import test_maddpg
from experiments.test_llm import test_llm
from utils.config import load_config
from utils.path_manager import PathManager, create_new_experiment


def setup_gpu(gpu_id=None):
    """设置GPU环境"""
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"✅ GPU设置成功: GPU {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
            return gpu_id
        else:
            print(f"❌ GPU {gpu_id} 不存在，使用默认GPU 0")
            gpu_id = 0
            torch.cuda.set_device(0)
            os.environ['CUDA_VISIBLE_DEVICES'] = "0"
            return gpu_id
    elif torch.cuda.is_available():
        print(f"🔧 使用默认GPU: {torch.cuda.get_device_name(0)}")
        return 0
    else:
        print("⚠️  CUDA不可用，将使用CPU训练")
        return None


def print_gpu_info():
    """打印GPU信息"""
    if torch.cuda.is_available():
        print(f"\n🎮 GPU环境信息:")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 显示当前GPU内存使用情况
        current_gpu = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_gpu) / 1024**3
        memory_cached = torch.cuda.memory_reserved(current_gpu) / 1024**3
        print(f"  当前GPU {current_gpu} 内存: {memory_allocated:.2f}GB / {memory_cached:.2f}GB (已分配/已缓存)")
    else:
        print(f"\n⚠️  GPU不可用:")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  将使用CPU进行训练")


def set_seed(seed):
    """设置随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_banner():
    """打印项目启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                  LLM+MADDPG 云边端计算卸载系统                 ║
║                        统一训练测试平台                       ║
║                                                              ║
║  支持算法：                                                    ║
║  • LLM+MADDPG（完整版）- 每step咨询LLM + 知识蒸馏              ║
║  • 纯MADDPG - 多智能体深度确定性策略梯度                          ║
║  • 纯LLM - 大语言模型直接决策                                  ║
║                                                              ║
║  功能：训练 → 测试 → 性能对比 → 结果可视化                      ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def train_llm_maddpg_algorithm(path_manager, config):
    """训练LLM+MADDPG完整版"""
    print("\n" + "="*80)
    print("🚀 训练 LLM+MADDPG（完整版）")
    print("="*80)
    
    try:
        # 训练
        start_time = time.time()
        results = train_llm_maddpg_complete("config.yaml")
        training_time = time.time() - start_time
        
        # 保存结果摘要
        results['training_time'] = training_time
        results['algorithm'] = 'LLM+MADDPG'
        
        result_file = path_manager.get_algorithm_result_file_path("llm_maddpg", "training_summary.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ LLM+MADDPG训练完成，耗时: {training_time:.2f}秒")
        print(f"💾 结果保存至: {result_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ LLM+MADDPG训练失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def train_pure_algorithms(path_manager, config):
    """训练纯MADDPG和纯LLM算法"""
    results = {}
    
    # 训练纯MADDPG
    print("\n" + "="*80)
    print("🔥 训练 纯MADDPG")
    print("="*80)
    
    try:
        start_time = time.time()
        maddpg_result = train_maddpg(config)
        training_time = time.time() - start_time
        
        results['maddpg'] = {
            'training_time': training_time,
            'algorithm': 'MADDPG',
            'result': maddpg_result
        }
        
        result_file = path_manager.get_algorithm_result_file_path("maddpg", "training_summary.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results['maddpg'], f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 纯MADDPG训练完成，耗时: {training_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 纯MADDPG训练失败: {e}")
        results['maddpg'] = {'error': str(e)}
    
    # 训练纯LLM
    print("\n" + "="*80)
    print("🧠 训练 纯LLM")
    print("="*80)
    
    try:
        start_time = time.time()
        llm_result = train_llm(config)
        training_time = time.time() - start_time
        
        results['llm'] = {
            'training_time': training_time,
            'algorithm': 'LLM',
            'result': llm_result
        }
        
        result_file = path_manager.get_algorithm_result_file_path("llm", "training_summary.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results['llm'], f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 纯LLM训练完成，耗时: {training_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 纯LLM训练失败: {e}")
        results['llm'] = {'error': str(e)}
    
    return results


def run_algorithm_tests(path_manager, config):
    """运行算法测试"""
    print("\n" + "="*80)
    print("🧪 开始算法性能测试")
    print("="*80)
    
    test_results = {}
    
    # 测试LLM+MADDPG (纯Agent模式，不使用LLM指导)
    try:
        print("🔬 测试 LLM+MADDPG (纯Agent模式)...")
        print("  📋 说明: 测试经过LLM+MADDPG训练的Agent在无LLM指导下的表现")
        llm_maddpg_metrics = test_llm_maddpg()
        if isinstance(llm_maddpg_metrics, tuple) and len(llm_maddpg_metrics) == 3:
            energy, util, delay = llm_maddpg_metrics
            test_results['llm_maddpg_pure_agent'] = {
                'energy': np.mean(energy) if energy else 0,
                'utilization': np.mean(util) if util else 0,
                'delay': np.mean(delay) if delay else 0,
                'energy_std': np.std(energy) if energy else 0,
                'utilization_std': np.std(util) if util else 0,
                'delay_std': np.std(delay) if delay else 0,
            }
            print(f"  ✅ 能耗: {test_results['llm_maddpg_pure_agent']['energy']:.4f}, "
                  f"时延: {test_results['llm_maddpg_pure_agent']['delay']:.4f}")
    except Exception as e:
        print(f"  ❌ LLM+MADDPG纯Agent测试失败: {e}")

    # 测试纯MADDPG
    try:
        print("\n🔬 测试 纯MADDPG...")
        maddpg_metrics = test_maddpg()
        if isinstance(maddpg_metrics, tuple) and len(maddpg_metrics) == 3:
            energy, util, delay = maddpg_metrics
            test_results['maddpg'] = {
                'energy': np.mean(energy) if energy else 0,
                'utilization': np.mean(util) if util else 0,
                'delay': np.mean(delay) if delay else 0,
                'energy_std': np.std(energy) if energy else 0,
                'utilization_std': np.std(util) if util else 0,
                'delay_std': np.std(delay) if delay else 0,
            }
            print(f"  ✅ 能耗: {test_results['maddpg']['energy']:.4f}, "
                  f"时延: {test_results['maddpg']['delay']:.4f}")
    except Exception as e:
        print(f"  ❌ 纯MADDPG测试失败: {e}")

    # 测试纯LLM
    try:
        print("\n🔬 测试 纯LLM...")
        llm_metrics = test_llm()
        if isinstance(llm_metrics, tuple) and len(llm_metrics) == 3:
            energy, util, delay = llm_metrics
            test_results['llm'] = {
                'energy': np.mean(energy) if energy else 0,
                'utilization': np.mean(util) if util else 0,
                'delay': np.mean(delay) if delay else 0,
                'energy_std': np.std(energy) if energy else 0,
                'utilization_std': np.std(util) if util else 0,
                'delay_std': np.std(delay) if delay else 0,
            }
            print(f"  ✅ 能耗: {test_results['llm']['energy']:.4f}, "
                  f"时延: {test_results['llm']['delay']:.4f}")
    except Exception as e:
        print(f"  ❌ 纯LLM测试失败: {e}")
    
    return test_results


def generate_comparison_report(training_results, test_results, path_manager):
    """生成算法对比报告"""
    print("\n" + "="*80)
    print("📊 生成算法对比报告")
    print("="*80)
    
    # 合并训练和测试结果
    comparison_data = []
    
    algorithms = ['llm_maddpg', 'maddpg', 'llm']
    algorithm_names = ['LLM+MADDPG', 'MADDPG', 'LLM']
    
    for algo, name in zip(algorithms, algorithm_names):
        row = {'Algorithm': name}
        
        # 添加训练结果
        if algo in training_results and 'result' in training_results[algo]:
            train_data = training_results[algo]['result']
            if isinstance(train_data, dict):
                row['Training_Time'] = training_results[algo].get('training_time', 0)
                row['Final_Reward'] = np.mean(train_data.get('episode_rewards', [0])[-50:]) if train_data.get('episode_rewards') else 0
                row['Total_Episodes'] = len(train_data.get('episode_rewards', []))
        
        # 添加测试结果
        test_key = f"{algo}_pure_agent" if algo == 'llm_maddpg' else algo
        if test_key in test_results:
            test_data = test_results[test_key]
            row['Test_Energy'] = test_data.get('energy', 0)
            row['Test_Utilization'] = test_data.get('utilization', 0)
            row['Test_Delay'] = test_data.get('delay', 0)
            row['Energy_Std'] = test_data.get('energy_std', 0)
            row['Utilization_Std'] = test_data.get('utilization_std', 0)
            row['Delay_Std'] = test_data.get('delay_std', 0)
        
        comparison_data.append(row)
    
    # 创建DataFrame并保存
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # 保存到CSV
        comparison_file = path_manager.get_comparison_file_path("algorithm_comparison.csv")
        df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
        
        # 保存到JSON
        json_file = path_manager.get_comparison_file_path("algorithm_comparison.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 对比报告保存至: {comparison_file}")
        print(f"✅ 详细数据保存至: {json_file}")
        
        # 打印对比结果
        print("\n📋 算法性能对比:")
        print("-" * 80)
        for _, row in df.iterrows():
            print(f"{row['Algorithm']}:")
            print(f"  能耗: {row.get('Test_Energy', 0):.4f} ± {row.get('Energy_Std', 0):.4f}")
            print(f"  利用率: {row.get('Test_Utilization', 0):.4f} ± {row.get('Utilization_Std', 0):.4f}")
            print(f"  时延: {row.get('Test_Delay', 0):.4f} ± {row.get('Delay_Std', 0):.4f}")
            if 'Training_Time' in row:
                print(f"  训练时间: {row.get('Training_Time', 0):.2f}秒")
            print()


def create_comparison_plots(test_results, path_manager):
    """创建对比图表"""
    print("📊 生成对比图表...")
    
    if not test_results:
        print("  ⚠️  没有测试结果，跳过图表生成")
        return
    
    try:
        # 准备数据
        algorithms = []
        energies = []
        utilizations = []
        delays = []
        
        for algo, data in test_results.items():
            if algo == 'llm_maddpg_pure_agent':
                algorithms.append('LLM+MADDPG\n(Pure Agent)')
            elif algo == 'maddpg':
                algorithms.append('MADDPG')
            elif algo == 'llm':
                algorithms.append('LLM')
            else:
                continue
                
            energies.append(data.get('energy', 0))
            utilizations.append(data.get('utilization', 0))
            delays.append(data.get('delay', 0))
        
        if len(algorithms) > 0:
            # 创建对比图表
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 能耗对比
            axes[0].bar(algorithms, energies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0].set_title('平均能耗对比')
            axes[0].set_ylabel('能耗 (J)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # 资源利用率对比
            axes[1].bar(algorithms, utilizations, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[1].set_title('平均资源利用率对比')
            axes[1].set_ylabel('利用率')
            axes[1].tick_params(axis='x', rotation=45)
            
            # 时延对比
            axes[2].bar(algorithms, delays, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[2].set_title('平均任务时延对比')
            axes[2].set_ylabel('时延 (s)')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_file = path_manager.get_plot_file_path("algorithm_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ 对比图表保存至: {plot_file}")
        
    except Exception as e:
        print(f"  ❌ 图表生成失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM+MADDPG云边端计算卸载系统 - 服务器训练版')
    
    # 运行模式
    parser.add_argument('--mode', choices=['all', 'train_only', 'test_only', 'llm_maddpg_only', 
                                          'maddpg_only', 'llm_only', 'test_maddpg_only', 
                                          'test_llm_maddpg_only', 'test_llm_only'], 
                       default='all', help='运行模式')
    
    # GPU设置
    parser.add_argument('--gpu', type=int, default=None, help='指定GPU ID (默认: 自动选择)')
    # 训练参数
    parser.add_argument('--episodes', type=int, default=None, help='训练轮数 (默认: 使用配置文件)')
    # 文件设置
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    # 服务器模式设置
    parser.add_argument('--server-mode', action='store_true', help='服务器模式: 显示详细信息和进度')
    parser.add_argument('--batch-train', action='store_true', help='批量训练: 按顺序训练所有算法')
    # 模型路径
    parser.add_argument('--model-path', type=str, default=None, help='指定测试时加载的模型目录')
    args = parser.parse_args()
    
    # 打印启动横幅
    print_banner()
    
    # 设置GPU环境
    if args.server_mode or args.gpu is not None:
        print_gpu_info()
    
    gpu_id = setup_gpu(args.gpu)
    
    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    config = load_config(args.config)

    # 如果指定了训练轮数，更新配置
    if args.episodes is not None:
        if 'training' not in config:
            config['training'] = {}
        config['training']['episodes'] = args.episodes
        print(f"🔧 训练轮数设置为: {args.episodes}")
    
    # 创建新的实验路径管理器
    path_manager = create_new_experiment()
    
    print(f"\n🗂️  实验配置:")
    print(f"  运行模式: {args.mode}")
    if gpu_id is not None:
        print(f"  使用GPU: {gpu_id}")
    else:
        print(f"  使用设备: CPU")
    print(f"  配置文件: {args.config}")
    print(f"  随机种子: {args.seed}")
    if args.episodes:
        print(f"  训练轮数: {args.episodes}")
    print(f"  实验目录: {path_manager.get_experiment_dir()}")
    print(f"  时间戳: {path_manager.experiment_timestamp}")
    
    # 服务器模式显示详细目录结构
    if args.server_mode:
        print("\n📁 目录结构:")
        dir_info = path_manager.get_directory_info()
        for key, path in dir_info.items():
            if key != 'experiment_timestamp':
                print(f"  {key}: {path}")
    
    training_results = {}
    test_results = {}
    
    try:
        # 批量训练模式
        if args.batch_train or args.mode == 'all':
            print(f"\n{'='*100}")
            print("🚀 服务器批量训练模式")
            print(f"{'='*100}")
            
            # 按顺序训练所有算法
            algorithms = ['maddpg', 'llm_maddpg', 'llm']
            
            print("📋 训练计划:")
            for i, algo in enumerate(algorithms, 1):
                print(f"  {i}. {algo.upper()}")
            print()
            
            for i, algo in enumerate(algorithms, 1):
                print(f"\n{'='*80}")
                print(f"🔥 步骤 {i}/3: 训练 {algo.upper()}")
                print(f"{'='*80}")
                
                if algo == 'maddpg':
                    # 训练纯MADDPG
                    try:
                        start_time = time.time()
                        maddpg_result = train_maddpg(config)
                        training_time = time.time() - start_time
                        
                        training_results['maddpg'] = {
                            'training_time': training_time,
                            'algorithm': 'MADDPG',
                            'result': maddpg_result
                        }
                        
                        result_file = path_manager.get_algorithm_result_file_path("maddpg", "training_summary.json")
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(training_results['maddpg'], f, indent=2, ensure_ascii=False, default=str)
                        
                        print(f"✅ {algo.upper()} 训练完成，耗时: {training_time:.2f}秒")
                        
                        if args.server_mode:
                            print(f"😴 休息10秒后继续下一个算法...")
                            time.sleep(10)
                            
                    except Exception as e:
                        print(f"❌ {algo.upper()} 训练失败: {e}")
                        training_results['maddpg'] = {'error': str(e)}
                
                elif algo == 'llm_maddpg':
                    # 训练LLM+MADDPG
                    try:
                        start_time = time.time()
                        results = train_llm_maddpg_complete("config.yaml")
                        training_time = time.time() - start_time
                        
                        results['training_time'] = training_time
                        results['algorithm'] = 'LLM+MADDPG'
                        
                        result_file = path_manager.get_algorithm_result_file_path("llm_maddpg", "training_summary.json")
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                        
                        training_results['llm_maddpg'] = results
                        print(f"✅ {algo.upper()} 训练完成，耗时: {training_time:.2f}秒")
                        
                        if args.server_mode:
                            print(f"😴 休息10秒后继续下一个算法...")
                            time.sleep(10)
                            
                    except Exception as e:
                        print(f"❌ {algo.upper()} 训练失败: {e}")
                        training_results['llm_maddpg'] = {'error': str(e)}
                
                elif algo == 'llm':
                    # 训练纯LLM
                    try:
                        start_time = time.time()
                        llm_result = train_llm(config)
                        training_time = time.time() - start_time
                        
                        training_results['llm'] = {
                            'training_time': training_time,
                            'algorithm': 'LLM',
                            'result': llm_result
                        }
                        
                        result_file = path_manager.get_algorithm_result_file_path("llm", "training_summary.json")
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(training_results['llm'], f, indent=2, ensure_ascii=False, default=str)
                        
                        print(f"✅ {algo.upper()} 训练完成，耗时: {training_time:.2f}秒")
                        
                    except Exception as e:
                        print(f"❌ {algo.upper()} 训练失败: {e}")
                        training_results['llm'] = {'error': str(e)}
            
            # 批量测试
            if args.mode == 'all':
                print(f"\n{'='*100}")
                print("🧪 服务器批量测试模式")
                print(f"{'='*100}")
                
                test_results = run_algorithm_tests(path_manager, config)
        
        else:
            # 单独训练模式逻辑
            print(f"\n{'='*80}")
            print("🎯 单独算法训练模式")
            print(f"{'='*80}")
            
            if args.mode == 'llm_maddpg_only':
                print("🚀 训练 LLM+MADDPG（完整版）")
                print("="*80)
                try:
                    start_time = time.time()
                    results = train_llm_maddpg_complete("config.yaml")
                    training_time = time.time() - start_time
                    
                    results['training_time'] = training_time
                    results['algorithm'] = 'LLM+MADDPG'
                    
                    result_file = path_manager.get_algorithm_result_file_path("llm_maddpg", "training_summary.json")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
                    
                    training_results['llm_maddpg'] = results
                    print(f"✅ LLM+MADDPG训练完成，耗时: {training_time:.2f}秒")
                    
                except Exception as e:
                    print(f"❌ LLM+MADDPG训练失败: {e}")
                    import traceback
                    traceback.print_exc()
                    training_results['llm_maddpg'] = {'error': str(e)}
            
            elif args.mode == 'maddpg_only':
                print("🔥 训练 纯MADDPG")
                print("="*80)
                try:
                    start_time = time.time()
                    maddpg_result = train_maddpg(config)
                    training_time = time.time() - start_time
                    
                    training_results['maddpg'] = {
                        'training_time': training_time,
                        'algorithm': 'MADDPG',
                        'result': maddpg_result
                    }
                    
                    result_file = path_manager.get_algorithm_result_file_path("maddpg", "training_summary.json")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(training_results['maddpg'], f, indent=2, ensure_ascii=False, default=str)
                    
                    print(f"✅ 纯MADDPG训练完成，耗时: {training_time:.2f}秒")
                    
                except Exception as e:
                    print(f"❌ 纯MADDPG训练失败: {e}")
                    import traceback
                    traceback.print_exc()
                    training_results['maddpg'] = {'error': str(e)}
            
            elif args.mode == 'llm_only':
                print("🧠 训练 纯LLM")
                print("="*80)
                try:
                    start_time = time.time()
                    llm_result = train_llm(config)
                    training_time = time.time() - start_time
                    
                    training_results['llm'] = {
                        'training_time': training_time,
                        'algorithm': 'LLM',
                        'result': llm_result
                    }
                    
                    result_file = path_manager.get_algorithm_result_file_path("llm", "training_summary.json")
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(training_results['llm'], f, indent=2, ensure_ascii=False, default=str)
                    
                    print(f"✅ 纯LLM训练完成，耗时: {training_time:.2f}秒")
                    
                except Exception as e:
                    print(f"❌ 纯LLM训练失败: {e}")
                    import traceback
                    traceback.print_exc()
                    training_results['llm'] = {'error': str(e)}
            
            # 原有的复合模式（all、train_only等）
            elif args.mode in ['all', 'train_only']:
                print(f"🎯 开始训练阶段")
                print(f"{'='*80}")
                
                if args.mode == 'all' or args.mode == 'llm_maddpg_only':
                    # 训练LLM+MADDPG
                    llm_maddpg_result = train_llm_maddpg_algorithm(path_manager, config)
                    if llm_maddpg_result:
                        training_results['llm_maddpg'] = llm_maddpg_result
                
                if args.mode == 'all' or args.mode == 'maddpg_only' or args.mode == 'llm_only':
                    # 训练其他算法
                    pure_results = train_pure_algorithms(path_manager, config)
                    training_results.update(pure_results)
            
            # 测试阶段
            if args.mode in ['all', 'test_only', 'test_maddpg_only', 'test_llm_maddpg_only', 'test_llm_only']:
                print(f"\n{'='*80}")
                print("🎯 开始测试阶段")
                print(f"{'='*80}")
                
                # 处理单独测试特定算法的情况
                if args.mode == 'test_maddpg_only':
                    print("\n🔬 仅测试 纯MADDPG...")
                    try:
                        maddpg_metrics = test_maddpg(model_path=args.model_path)
                        if isinstance(maddpg_metrics, tuple) and len(maddpg_metrics) == 3:
                            energy, util, delay = maddpg_metrics
                            test_results['maddpg'] = {
                                'energy': np.mean(energy) if energy else 0,
                                'utilization': np.mean(util) if util else 0,
                                'delay': np.mean(delay) if delay else 0,
                                'energy_std': np.std(energy) if energy else 0,
                                'utilization_std': np.std(util) if util else 0,
                                'delay_std': np.std(delay) if delay else 0,
                            }
                            print(f"  ✅ 能耗: {test_results['maddpg']['energy']:.4f}, "
                                  f"时延: {test_results['maddpg']['delay']:.4f}")
                    except Exception as e:
                        print(f"  ❌ 纯MADDPG测试失败: {e}")
                elif args.mode == 'test_llm_maddpg_only':
                    print("\n🔬 仅测试 LLM+MADDPG (纯Agent)...")
                    try:
                        llm_maddpg_metrics = test_llm_maddpg(model_path=args.model_path)
                        if isinstance(llm_maddpg_metrics, tuple) and len(llm_maddpg_metrics) == 3:
                            energy, util, delay = llm_maddpg_metrics
                            test_results['llm_maddpg_pure_agent'] = {
                                'energy': np.mean(energy) if energy else 0,
                                'utilization': np.mean(util) if util else 0,
                                'delay': np.mean(delay) if delay else 0,
                                'energy_std': np.std(energy) if energy else 0,
                                'utilization_std': np.std(util) if util else 0,
                                'delay_std': np.std(delay) if delay else 0,
                            }
                            print(f"  ✅ 能耗: {test_results['llm_maddpg_pure_agent']['energy']:.4f}, "
                                  f"时延: {test_results['llm_maddpg_pure_agent']['delay']:.4f}")
                    except Exception as e:
                        print(f"  ❌ LLM+MADDPG纯Agent测试失败: {e}")
                elif args.mode == 'test_llm_only':
                    print("\n🔬 仅测试 纯LLM...")
                    try:
                        llm_metrics = test_llm(model_path=args.model_path)
                        if isinstance(llm_metrics, tuple) and len(llm_metrics) == 3:
                            energy, util, delay = llm_metrics
                            test_results['llm'] = {
                                'energy': np.mean(energy) if energy else 0,
                                'utilization': np.mean(util) if util else 0,
                                'delay': np.mean(delay) if delay else 0,
                                'energy_std': np.std(energy) if energy else 0,
                                'utilization_std': np.std(util) if util else 0,
                                'delay_std': np.std(delay) if delay else 0,
                            }
                            print(f"  ✅ 能耗: {test_results['llm']['energy']:.4f}, "
                                  f"时延: {test_results['llm']['delay']:.4f}")
                    except Exception as e:
                        print(f"  ❌ 纯LLM测试失败: {e}")
                else:
                    # 默认批量测试
                    test_results = run_algorithm_tests(path_manager, config)
        
        # 对比分析
        if args.mode == 'all' and (training_results or test_results):
            print(f"\n{'='*80}")
            print("🎯 开始对比分析")
            print(f"{'='*80}")
            
            generate_comparison_report(training_results, test_results, path_manager)
            create_comparison_plots(test_results, path_manager)
        
        # 保存完整实验记录
        experiment_summary = {
            'experiment_timestamp': path_manager.experiment_timestamp,
            'mode': args.mode,
            'gpu_id': gpu_id,
            'episodes': args.episodes,
            'config_file': args.config,
            'seed': args.seed,
            'server_mode': args.server_mode,
            'batch_train': args.batch_train,
            'training_results': training_results,
            'test_results': test_results,
            'directory_structure': path_manager.get_directory_info()
        }
        
        summary_file = path_manager.get_experiment_dir() + "/experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n🎉 实验完成！")
        print(f"📁 所有结果保存在: {path_manager.get_experiment_dir()}")
        print(f"📄 实验摘要: {summary_file}")
        
        if args.server_mode:
            print(f"\n📊 最终统计:")
            success_count = len([r for r in training_results.values() if 'error' not in r])
            total_count = len(training_results)
            print(f"  训练成功: {success_count}/{total_count}")
            
            if test_results:
                print(f"  测试完成: {len(test_results)} 个算法")
            
            if gpu_id is not None:
                # 显示最终GPU内存使用
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                memory_cached = torch.cuda.memory_reserved(gpu_id) / 1024**3
                print(f"  GPU {gpu_id} 最终内存: {memory_allocated:.2f}GB / {memory_cached:.2f}GB")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断实验")
        print(f"📁 已生成的结果保存在: {path_manager.get_experiment_dir()}")
    except Exception as e:
        print(f"\n❌ 实验执行失败: {e}")
        if args.server_mode:
            import traceback
            traceback.print_exc()
        print(f"📁 部分结果可能保存在: {path_manager.get_experiment_dir()}")


if __name__ == "__main__":
    main()