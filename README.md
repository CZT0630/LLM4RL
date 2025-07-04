# LLM+MADDPG 云边端计算卸载系统

## 🌟 项目概述

本项目实现了基于大语言模型(LLM)和多智能体深度确定性策略梯度(MADDPG)的云边端三层架构计算卸载决策系统。通过LLM专家知识指导和知识蒸馏技术，提升MADDPG智能体的决策能力。

### 核心特性

- **LLM+MADDPG混合架构**: 结合LLM的推理能力和MADDPG的学习能力
- **知识蒸馏技术**: Agent通过知识蒸馏学习LLM的决策模式
- **纯Agent模式**: 训练后的Agent可在无LLM指导下独立决策
- **云边端三层架构**: 支持10个端设备、5个边缘服务器、1个云服务器的异构环境
- **三元分割决策**: 支持任务在本地、边缘、云端的灵活分配
- **统一路径管理**: 标准化的目录结构和文件保存系统

## 🏗️ 系统架构

### 训练阶段
```
LLM专家指导 → MADDPG决策 → 环境交互 → 经验存储 → 网络训练(含知识蒸馏)
```

### 测试阶段
```
纯Agent决策 → 环境交互 → 性能评估
```

**注意**: 测试阶段不再需要LLM指导，Agent通过知识蒸馏已经内化了LLM的决策能力。

## 📁 项目结构

```
LLM4RL/
├── algos/                      # 算法实现
│   ├── maddpg_agent.py        # MADDPG智能体(支持知识蒸馏)
│   ├── maddpg_actor_critic.py # Actor-Critic网络
│   ├── replay_buffer.py       # 经验回放缓冲区
│   └── noise.py              # 噪声模型
├── environment/               # 环境模型
│   ├── cloud_edge_env.py     # 云边端环境
│   ├── device_models.py      # 设备模型
│   └── task_generator.py     # 任务生成器
├── llm_assistant/            # LLM助手模块
│   ├── llm_client.py         # LLM客户端
│   ├── prompt_builder.py     # 提示词构建
│   └── response_parser.py    # 响应解析
├── experiments/              # 训练测试脚本
│   ├── train_llm_maddpg_complete.py  # LLM+MADDPG训练
│   ├── test_llm_maddpg.py           # 纯Agent模式测试
│   ├── train_maddpg.py              # 纯MADDPG训练
│   ├── test_maddpg.py               # 纯MADDPG测试
│   ├── train_llm.py                 # 纯LLM训练
│   └── test_llm.py                  # 纯LLM测试
├── utils/                    # 工具函数
│   ├── path_manager.py       # 统一路径管理器
│   ├── csv_saver.py          # CSV数据保存
│   ├── config.py            # 配置管理
│   ├── metrics.py           # 性能指标
│   └── plotting.py          # 图表绘制
├── results/                  # 结果目录(自动生成)
│   └── experiment_{timestamp}/
│       ├── maddpg/models/           # 纯MADDPG模型
│       │   ├── actor_agent_0_final.pth
│       │   ├── critic_agent_0_final.pth
│       │   └── ...
├── llm_maddpg/                 # LLM+MADDPG算法结果
│   ├── models/                 # 模型文件
│   │   ├── agent_0_final.pth
│   │   ├── agent_1_final.pth
│   │   └── ...
├── llm/                        # 纯LLM算法结果
│   └── models/                 # 模型文件(如有)
├── data/                       # 数据文件
│   ├── csv/                    # CSV格式训练指标
│   │   ├── MADDPG_training_metrics_*.csv
│   │   ├── LLM_MADDPG_training_metrics_*.csv
│   │   └── LLM_training_metrics_*.csv
│   └── json/                   # JSON格式详细数据
├── plots/                      # 图表文件
├── logs/                       # 日志文件
├── test_results/               # 测试结果
└── comparison/                 # 算法对比结果
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install torch numpy gymnasium matplotlib pyyaml pandas requests

# 配置LLM服务器(可选，仅训练时需要)
# 编辑 config.yaml 中的 llm 配置项
```

### 2. 使用统一入口(推荐)

```bash
# 完整训练和测试流程
python main.py --mode all

# 仅训练所有算法
python main.py --mode train_only

# 仅测试所有算法
python main.py --mode test_only

# 训练特定算法
python main.py --mode llm_maddpg_only
python main.py --mode maddpg_only
python main.py --mode llm_only
```

### 3. 🎮 服务器GPU训练（新增）

#### 环境检查
```bash
# 检查GPU环境
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

#### 指定GPU训练
```bash
# 使用GPU 0进行完整训练+测试
python main.py --gpu 0 --mode all --server-mode

# 使用GPU 1，自定义训练轮数
python main.py --gpu 1 --episodes 1000 --mode all --server-mode

# 批量训练模式（按顺序训练所有算法）
python main.py --gpu 0 --batch-train --episodes 500 --server-mode
```

#### 服务器模式参数说明
- `--gpu`: 指定GPU ID（如：0, 1, 2...）
- `--episodes`: 自定义训练轮数（覆盖配置文件设置）
- `--server-mode`: 显示详细信息和GPU内存使用
- `--batch-train`: 批量训练模式，按顺序执行所有算法
- `--seed`: 设置随机种子确保可复现性

#### 批量训练流程
```bash
# 完整的服务器批量训练（推荐）
python main.py --gpu 0 --batch-train --episodes 1000 --server-mode --seed 42
```

**执行顺序：**
1. 🔥 训练MADDPG → 😴 休息10秒
2. 🔥 训练LLM+MADDPG → 😴 休息10秒  
3. 🔥 训练LLM
4. 🧪 测试MADDPG
5. 🧪 测试LLM+MADDPG
6. 🧪 测试LLM
7. 📊 生成对比报告

#### 单算法GPU训练
```bash
# 仅在GPU上训练MADDPG
python main.py --gpu 0 --mode maddpg_only --episodes 500 --server-mode

# 仅在GPU上训练LLM+MADDPG
python main.py --gpu 1 --mode llm_maddpg_only --episodes 800 --server-mode

# 仅在GPU上训练LLM
python main.py --gpu 0 --mode llm_only --episodes 300 --server-mode
```

#### 多GPU并行训练（高级用法）
```bash
# 终端1：GPU 0训练MADDPG
python main.py --gpu 0 --mode maddpg_only --episodes 1000 --server-mode

# 终端2：GPU 1训练LLM+MADDPG
python main.py --gpu 1 --mode llm_maddpg_only --episodes 1000 --server-mode

# 终端3：GPU 2训练LLM（如果有第三个GPU）
python main.py --gpu 2 --mode llm_only --episodes 500 --server-mode
```

#### 性能建议
- **RTX 4060**: 建议 `batch_size=32-64`, `episodes=500-1000`
- **RTX 4070/4080**: 建议 `batch_size=64-128`, `episodes=1000-1500`
- **RTX 4090**: 建议 `batch_size=128-256`, `episodes=1500-3000`
- **服务器GPU**: 根据显存调整参数，建议使用 `--server-mode` 监控

### 4. 单独使用训练脚本

```bash
# 训练LLM+MADDPG
python experiments/train_llm_maddpg_complete.py

# 训练纯MADDPG
python experiments/train_maddpg.py

# 训练纯LLM
python experiments/train_llm.py
```

### 5. 单独使用测试脚本

```bash
# 测试LLM+MADDPG纯Agent模式(推荐)
python experiments/test_llm_maddpg.py

# 测试纯MADDPG
python experiments/test_maddpg.py

# 测试纯LLM  
python experiments/test_llm.py
```

## 📊 统一路径管理系统

### 自动目录结构
每次实验都会创建带时间戳的独立目录：
```
results/experiment_20250703_195030/
├── maddpg/                     # 纯MADDPG算法结果
│   ├── models/                 # 模型文件
│   │   ├── actor_agent_0_final.pth
│   │   ├── critic_agent_0_final.pth
│   │   └── ...
├── llm_maddpg/                 # LLM+MADDPG算法结果
│   ├── models/                 # 模型文件
│   │   ├── agent_0_final.pth
│   │   ├── agent_1_final.pth
│   │   └── ...
├── llm/                        # 纯LLM算法结果
│   └── models/                 # 模型文件(如有)
├── data/                       # 数据文件
│   ├── csv/                    # CSV格式训练指标
│   │   ├── MADDPG_training_metrics_*.csv
│   │   ├── LLM_MADDPG_training_metrics_*.csv
│   │   └── LLM_training_metrics_*.csv
│   └── json/                   # JSON格式详细数据
├── plots/                      # 图表文件
├── logs/                       # 日志文件
├── test_results/               # 测试结果
└── comparison/                 # 算法对比结果
```

### 模型文件格式
- **纯MADDPG**: 分离格式 - `actor_agent_{i}_final.pth`, `critic_agent_{i}_final.pth`
- **LLM+MADDPG**: 完整格式 - `agent_{i}_final.pth`
- **纯LLM**: 无需保存模型文件

## 🔧 配置说明

### 系统配置 (`config.yaml`)

```yaml
# 环境配置
environment:
  num_devices: 10      # 端设备数量
  num_edges: 5         # 边缘服务器数量
  num_clouds: 1        # 云服务器数量

# LLM+MADDPG配置
llm_maddpg:
  max_episodes: 1000          # 训练轮数
  llm_episode_interval: 2     # 每2个Episode使用LLM指导
  llm_distill_weight: 0.1     # 知识蒸馏权重

# 纯MADDPG配置
maddpg:
  max_episodes: 1000          # 训练轮数
  train_frequency: 20         # 训练频率
  
# 测试配置
testing:
  num_episodes: 200    # 测试轮数
  max_steps: 100      # 每轮最大步数

# 训练策略
training:
  save_frequency: 100  # 模型保存频率
  log_frequency: 10    # 日志输出频率
```

## 🧪 算法特性

### 知识蒸馏机制

1. **训练阶段**: LLM提供专家动作，Agent学习模仿
2. **蒸馏损失**: `L_distill = MSE(agent_action, llm_action)`
3. **总损失**: `L_total = L_policy + α * L_distill`
4. **内化效果**: Agent逐渐学会独立决策

### 测试模式对比

| 模式 | LLM指导 | 模型格式 | 适用场景 | 性能特点 |
|------|---------|----------|----------|----------|
| LLM+MADDPG(训练) | ✅ | 完整模型 | 训练阶段 | 快速收敛，高质量策略 |
| LLM+MADDPG(纯Agent) | ❌ | 完整模型 | 部署阶段 | 低延迟，无外部依赖 |
| 纯MADDPG | ❌ | 分离模型 | 基线对比 | 传统强化学习效果 |
| 纯LLM | ✅ | 无模型 | 实时决策 | 高质量但高延迟 |

### 动作空间

```python
# 三元分割决策
action = [α1, α2, α3, edge_id]
# α1: 本地执行比例
# α2: 边缘执行比例  
# α3: 云端执行比例 (α1 + α2 + α3 = 1)
# edge_id: 目标边缘服务器ID (0-4)
```

## 📊 性能指标

- **能耗效率**: 总能耗消耗 (J)
- **时延性能**: 平均任务完成时延 (s)
- **资源利用**: 系统资源利用率 (%)
- **完成率**: 任务截止时间满足率 (%)

## 🎯 核心优势

### 1. 统一路径管理
- 自动创建标准化目录结构
- 时间戳标识的独立实验
- 算法特定的文件组织
- 支持训练和测试结果分离

### 2. 知识蒸馏效果
- Agent通过训练学习LLM决策模式
- 测试时无需LLM，降低推理成本
- 保持决策质量，提升响应速度

### 3. 部署优势
- **低延迟**: 无需等待LLM推理
- **高可靠**: 不依赖外部LLM服务
- **低成本**: 减少计算资源消耗

### 4. 数据管理
- CSV格式的标准化训练指标
- JSON格式的详细实验数据
- 自动算法对比报告
- 完整的实验追踪记录

## 🔍 实验验证

### 使用统一入口验证
```bash
# 完整验证流程
python main.py --mode all --seed 42

# 查看结果
ls results/experiment_*/
```

### 预期结果
- LLM+MADDPG纯Agent模式应接近有LLM指导的性能
- 相比纯MADDPG应有显著性能提升
- 响应时间大幅降低(无LLM推理延迟)
- 所有结果保存在标准化目录结构中

## ⚠️ 注意事项

1. **训练依赖**: 训练阶段需要LLM服务器支持
2. **测试独立**: 测试阶段完全不依赖LLM
3. **路径管理**: 系统自动管理所有文件路径
4. **模型兼容**: 支持多种模型文件格式自动检测
5. **配置匹配**: 测试配置应与训练环境配置一致

## 🔄 开发迭代

### 当前版本特性
- ✅ 完成LLM+MADDPG训练框架
- ✅ 实现知识蒸馏机制
- ✅ 支持纯Agent模式测试
- ✅ 统一路径管理系统
- ✅ 标准化数据保存格式
- ✅ 完整性能对比分析

### 后续计划
- 🔲 优化知识蒸馏算法
- 🔲 增加更多评估指标
- 🔲 支持在线学习模式
- 🔲 模型压缩与加速

## 📧 联系方式

如有问题，请通过Issue或邮件联系开发团队。
