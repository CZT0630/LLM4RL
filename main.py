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
║                  LLM+MADDPG 云边端计算卸载系统                  ║
║                        统一训练测试平台                         ║
║                                                              ║
║  支持算法：                                                    ║
║  • LLM+MADDPG（完整版）- 每step咨询LLM + 知识蒸馏                 ║
║  • 纯MADDPG - 多智能体深度确定性策略梯度                          ║
║  • 纯LLM - 大语言模型直接决策                                    ║
║                                                              ║
║  功能：训练 → 测试 → 性能对比 → 结果可视化                         ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def create_result_directories(base_dir="results"):
    """创建结果保存目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"{base_dir}/experiment_{timestamp}"
    
    directories = [
        experiment_dir,
        f"{experiment_dir}/llm_maddpg",
        f"{experiment_dir}/pure_maddpg", 
        f"{experiment_dir}/pure_llm",
        f"{experiment_dir}/comparison",
        f"{experiment_dir}/logs",
        f"{experiment_dir}/plots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return experiment_dir


def train_llm_maddpg_complete(config, experiment_dir):
    """训练LLM+MADDPG完整版"""
    print("\n" + "="*80)
    print("🚀 训练 LLM+MADDPG（完整版）")
    print("="*80)
    
    try:
        from experiments.train_llm_maddpg_complete import train_llm_maddpg_complete
        
        # 训练
        start_time = time.time()
        results = train_llm_maddpg_complete("config.yaml")
        training_time = time.time() - start_time
        
        # 保存结果
        results['training_time'] = training_time
        results['algorithm'] = 'LLM+MADDPG'
        
        result_file = f"{experiment_dir}/llm_maddpg/training_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ LLM+MADDPG训练完成，耗时: {training_time:.2f}秒")
        print(f"💾 结果保存至: {result_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ LLM+MADDPG训练失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def train_pure_algorithms(config, experiment_dir):
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
        
        results['pure_maddpg'] = {
            'training_time': training_time,
            'algorithm': '纯MADDPG',
            'result': maddpg_result
        }
        
        result_file = f"{experiment_dir}/pure_maddpg/training_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results['pure_maddpg'], f, indent=2, ensure_ascii=False)
        
        print(f"✅ 纯MADDPG训练完成，耗时: {training_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 纯MADDPG训练失败: {e}")
        results['pure_maddpg'] = {'error': str(e)}
    
    # 训练纯LLM
    print("\n" + "="*80)
    print("🧠 训练 纯LLM")
    print("="*80)
    
    try:
        start_time = time.time()
        llm_result = train_llm(config)
        training_time = time.time() - start_time
        
        results['pure_llm'] = {
            'training_time': training_time,
            'algorithm': '纯LLM',
            'result': llm_result
        }
        
        result_file = f"{experiment_dir}/pure_llm/training_results.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results['pure_llm'], f, indent=2, ensure_ascii=False)
        
        print(f"✅ 纯LLM训练完成，耗时: {training_time:.2f}秒")
        
    except Exception as e:
        print(f"❌ 纯LLM训练失败: {e}")
        results['pure_llm'] = {'error': str(e)}
    
    return results


def run_algorithm_tests(config, experiment_dir):
    """运行算法测试"""
    print("\n" + "="*80)
    print("🧪 算法性能测试")
    print("="*80)
    
    test_results = {}
    
    # 测试LLM+MADDPG
    try:
        print("🔬 测试 LLM+MADDPG...")
        llm_maddpg_metrics = test_llm_maddpg(f"{experiment_dir}/llm_maddpg", config)
        if isinstance(llm_maddpg_metrics, tuple) and len(llm_maddpg_metrics) == 3:
            energy, util, delay = llm_maddpg_metrics
            test_results['llm_maddpg'] = {
                'energy': np.mean(energy) if energy else 0,
                'utilization': np.mean(util) if util else 0,
                'delay': np.mean(delay) if delay else 0,
                'energy_std': np.std(energy) if energy else 0,
                'utilization_std': np.std(util) if util else 0,
                'delay_std': np.std(delay) if delay else 0,
            }
            print(f"  ✅ 能耗: {test_results['llm_maddpg']['energy']:.4f}, 时延: {test_results['llm_maddpg']['delay']:.4f}")
    except Exception as e:
        print(f"  ❌ LLM+MADDPG测试失败: {e}")
        test_results['llm_maddpg'] = {'error': str(e)}
    
    # 测试纯MADDPG
    try:
        print("🔬 测试 纯MADDPG...")
        maddpg_metrics = test_maddpg(f"{experiment_dir}/pure_maddpg", config)
        if isinstance(maddpg_metrics, tuple) and len(maddpg_metrics) == 3:
            energy, util, delay = maddpg_metrics
            test_results['pure_maddpg'] = {
                'energy': np.mean(energy) if energy else 0,
                'utilization': np.mean(util) if util else 0,
                'delay': np.mean(delay) if delay else 0,
                'energy_std': np.std(energy) if energy else 0,
                'utilization_std': np.std(util) if util else 0,
                'delay_std': np.std(delay) if delay else 0,
            }
            print(f"  ✅ 能耗: {test_results['pure_maddpg']['energy']:.4f}, 时延: {test_results['pure_maddpg']['delay']:.4f}")
    except Exception as e:
        print(f"  ❌ 纯MADDPG测试失败: {e}")
        test_results['pure_maddpg'] = {'error': str(e)}
    
    # 测试纯LLM
    try:
        print("🔬 测试 纯LLM...")
        llm_metrics = test_llm(f"{experiment_dir}/pure_llm", config)
        if isinstance(llm_metrics, tuple) and len(llm_metrics) == 3:
            energy, util, delay = llm_metrics
            test_results['pure_llm'] = {
                'energy': np.mean(energy) if energy else 0,
                'utilization': np.mean(util) if util else 0,
                'delay': np.mean(delay) if delay else 0,
                'energy_std': np.std(energy) if energy else 0,
                'utilization_std': np.std(util) if util else 0,
                'delay_std': np.std(delay) if delay else 0,
            }
            print(f"  ✅ 能耗: {test_results['pure_llm']['energy']:.4f}, 时延: {test_results['pure_llm']['delay']:.4f}")
    except Exception as e:
        print(f"  ❌ 纯LLM测试失败: {e}")
        test_results['pure_llm'] = {'error': str(e)}
    
    # 保存测试结果
    test_file = f"{experiment_dir}/comparison/test_results.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    return test_results


def generate_comparison_report(training_results, test_results, experiment_dir):
    """生成详细的对比报告"""
    print("\n" + "="*80)
    print("📊 生成算法对比报告")
    print("="*80)
    
    # 整合所有结果
    comparison_data = []
    
    # LLM+MADDPG结果
    if training_results and test_results.get('llm_maddpg'):
        llm_maddpg_data = {
            'algorithm': 'LLM+MADDPG',
            'training_time': training_results.get('training_time', 0),
            'avg_reward': np.mean(training_results.get('episode_rewards', [0])[-50:]),
            'avg_latency': np.mean(training_results.get('episode_latencies', [0])[-50:]),
            'avg_energy': np.mean(training_results.get('episode_energies', [0])[-50:]),
            'completion_rate': np.mean(training_results.get('episode_completion_rates', [0])[-50:]),
            'test_energy': test_results['llm_maddpg'].get('energy', 0),
            'test_delay': test_results['llm_maddpg'].get('delay', 0),
            'test_utilization': test_results['llm_maddpg'].get('utilization', 0),
        }
        comparison_data.append(llm_maddpg_data)
    
    # 纯MADDPG和纯LLM结果（简化处理）
    for algo_name in ['pure_maddpg', 'pure_llm']:
        if test_results.get(algo_name) and 'error' not in test_results[algo_name]:
            algo_data = {
                'algorithm': '纯MADDPG' if algo_name == 'pure_maddpg' else '纯LLM',
                'test_energy': test_results[algo_name].get('energy', 0),
                'test_delay': test_results[algo_name].get('delay', 0),
                'test_utilization': test_results[algo_name].get('utilization', 0),
            }
            comparison_data.append(algo_data)
    
    if not comparison_data:
        print("❌ 没有有效的对比数据")
        return
    
    # 创建DataFrame并保存
    df = pd.DataFrame(comparison_data)
    csv_path = f"{experiment_dir}/comparison/comparison_results.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 打印对比表格
    print("\n📈 算法性能对比结果:")
    print("="*120)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # 生成可视化图表
    create_comparison_plots(df, experiment_dir)
    
    # 保存完整报告
    report = {
        'comparison_data': comparison_data,
        'training_results': training_results,
        'test_results': test_results,
        'experiment_time': datetime.now().isoformat(),
    }
    
    report_file = f"{experiment_dir}/comparison/full_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 完整报告保存至: {report_file}")
    print(f"📊 对比表格保存至: {csv_path}")


def create_comparison_plots(df, experiment_dir):
    """创建对比可视化图表"""
    print("📊 生成对比图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLM+MADDPG vs 纯MADDPG vs 纯LLM 性能对比', fontsize=16, fontweight='bold')
    
    algorithms = df['algorithm'].tolist()
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # 1. 测试能耗对比
    if 'test_energy' in df.columns:
        energies = df['test_energy'].tolist()
        axes[0, 0].bar(algorithms, energies, color=colors[:len(algorithms)])
        axes[0, 0].set_title('测试能耗对比')
        axes[0, 0].set_ylabel('能耗 (J)')
        axes[0, 0].tick_params(axis='x', rotation=15)
    
    # 2. 测试时延对比
    if 'test_delay' in df.columns:
        delays = df['test_delay'].tolist()
        axes[0, 1].bar(algorithms, delays, color=colors[:len(algorithms)])
        axes[0, 1].set_title('测试时延对比')
        axes[0, 1].set_ylabel('时延 (s)')
        axes[0, 1].tick_params(axis='x', rotation=15)
    
    # 3. 资源利用率对比
    if 'test_utilization' in df.columns:
        utilizations = df['test_utilization'].tolist()
        axes[1, 0].bar(algorithms, utilizations, color=colors[:len(algorithms)])
        axes[1, 0].set_title('资源利用率对比')
        axes[1, 0].set_ylabel('利用率')
        axes[1, 0].tick_params(axis='x', rotation=15)
    
    # 4. 训练奖励对比（如果有数据）
    if 'avg_reward' in df.columns:
        rewards = df['avg_reward'].tolist()
        # 过滤掉NaN值
        valid_rewards = [(alg, rew) for alg, rew in zip(algorithms, rewards) if not np.isnan(rew)]
        if valid_rewards:
            valid_algs, valid_rews = zip(*valid_rewards)
            axes[1, 1].bar(valid_algs, valid_rews, color=colors[:len(valid_algs)])
            axes[1, 1].set_title('训练平均奖励对比')
            axes[1, 1].set_ylabel('平均奖励')
            axes[1, 1].tick_params(axis='x', rotation=15)
        else:
            axes[1, 1].text(0.5, 0.5, '无训练奖励数据', ha='center', va='center', transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].text(0.5, 0.5, '无训练奖励数据', ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plot_path = f"{experiment_dir}/plots/comparison_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 对比图表保存至: {plot_path}")


def main():
    """主函数 - 统一入口"""
    parser = argparse.ArgumentParser(description='LLM+MADDPG 云边端计算卸载系统')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['all', 'train', 'test', 'compare', 'llm_maddpg_only', 'maddpg_only', 'llm_only'],
                       help='运行模式')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（减少训练轮数）')
    parser.add_argument('--no-plots', action='store_true', help='跳过图表生成')
    
    args = parser.parse_args()
    
    # 打印启动横幅
    print_banner()
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 运行模式: {args.mode}")
    
    # 加载配置
    try:
        config = load_config(args.config)
        print(f"✅ 配置文件加载成功: {args.config}")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        sys.exit(1)
    
    # 快速测试模式
    if args.quick:
        print("⚡ 快速测试模式")
        config['training']['episodes'] = 50
        config['training']['max_steps_per_episode'] = 20
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 创建实验目录
    experiment_dir = create_result_directories()
    print(f"📁 实验结果目录: {experiment_dir}")
    
    # 根据模式执行不同功能
    training_results = {}
    test_results = {}
    
    if args.mode in ['all', 'train', 'llm_maddpg_only']:
        # 训练LLM+MADDPG完整版
        training_results = train_llm_maddpg_complete(config, experiment_dir)
    
    if args.mode in ['all', 'train']:
        # 训练其他算法
        other_results = train_pure_algorithms(config, experiment_dir)
        training_results.update(other_results)
    
    if args.mode in ['maddpg_only']:
        print("\n🔥 仅训练纯MADDPG")
        try:
            start_time = time.time()
            result = train_maddpg(config)
            training_time = time.time() - start_time
            print(f"✅ 纯MADDPG训练完成，耗时: {training_time:.2f}秒")
        except Exception as e:
            print(f"❌ 纯MADDPG训练失败: {e}")
    
    if args.mode in ['llm_only']:
        print("\n🧠 仅训练纯LLM")
        try:
            start_time = time.time()
            result = train_llm(config)
            training_time = time.time() - start_time
            print(f"✅ 纯LLM训练完成，耗时: {training_time:.2f}秒")
        except Exception as e:
            print(f"❌ 纯LLM训练失败: {e}")
    
    if args.mode in ['all', 'test', 'compare']:
        # 运行测试
        test_results = run_algorithm_tests(config, experiment_dir)
    
    if args.mode in ['all', 'compare'] and (training_results or test_results):
        # 生成对比报告
        if not args.no_plots:
            generate_comparison_report(training_results, test_results, experiment_dir)
        else:
            print("⏭️ 跳过图表生成")
    
    # 打印最终总结
    print("\n" + "🎉" * 20)
    print("✅ 实验完成！")
    print("🎉" * 20)
    print(f"\n📁 所有结果保存在: {experiment_dir}")
    print(f"📊 对比报告: {experiment_dir}/comparison/")
    print(f"📈 可视化图表: {experiment_dir}/plots/")
    
    # 简要显示结果
    if test_results:
        print("\n📈 简要性能对比:")
        print("-" * 60)
        for algo, metrics in test_results.items():
            if 'error' not in metrics:
                algo_name = {'llm_maddpg': 'LLM+MADDPG', 'pure_maddpg': '纯MADDPG', 'pure_llm': '纯LLM'}.get(algo, algo)
                print(f"{algo_name:12} | 能耗: {metrics.get('energy', 0):.4f} | 时延: {metrics.get('delay', 0):.4f}")


if __name__ == "__main__":
    main()