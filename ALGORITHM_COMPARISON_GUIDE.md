# LLM+MADDPG 统一训练测试系统使用指南

## 🎯 **项目概述**

基于LLM+MADDPG算法的云边端计算卸载系统，现已整合为**单一入口文件**，支持完整的训练、测试、对比功能。

### **支持的算法：**
- **LLM+MADDPG（完整版）** - 每step咨询LLM + 知识蒸馏
- **纯MADDPG** - 多智能体深度确定性策略梯度  
- **纯LLM** - 大语言模型直接决策

## 🚀 **快速开始**

### **基本使用（推荐）：**
```bash
# 完整流程：训练三种算法 → 测试 → 性能对比 → 生成报告
python main.py --mode all

# 快速测试模式（减少训练轮数）
python main.py --mode all --quick
```

### **分步执行：**
```bash
# 1. 仅训练所有算法
python main.py --mode train

# 2. 仅测试性能
python main.py --mode test

# 3. 仅生成对比报告
python main.py --mode compare
```

### **单独训练：**
```bash
# 仅训练LLM+MADDPG完整版
python main.py --mode llm_maddpg_only

# 仅训练纯MADDPG
python main.py --mode maddpg_only

# 仅训练纯LLM
python main.py --mode llm_only
```

## 📊 **运行模式详解**

| 模式 | 功能 | 使用场景 |
|------|------|----------|
| `all` | 训练+测试+对比+报告生成 | **推荐**，完整实验流程 |
| `train` | 训练三种算法 | 批量训练 |
| `test` | 测试已训练的模型 | 性能评估 |
| `compare` | 生成对比报告和图表 | 结果分析 |
| `llm_maddpg_only` | 仅训练LLM+MADDPG | 专项训练 |
| `maddpg_only` | 仅训练纯MADDPG | 基线对比 |
| `llm_only` | 仅训练纯LLM | 参考对比 |

## 🔧 **命令行参数**

```bash
python main.py [选项]

必需参数：
  --mode {all,train,test,compare,llm_maddpg_only,maddpg_only,llm_only}
                        运行模式（默认：all）

可选参数：
  --config CONFIG       配置文件路径（默认：config.yaml）
  --quick              快速测试模式（减少训练轮数）
  --no-plots           跳过图表生成
  -h, --help           显示帮助信息
```

## 📈 **输出结果结构**

```
results/
└── experiment_YYYYMMDD_HHMMSS/          # 实验时间戳目录
    ├── llm_maddpg/                      # LLM+MADDPG结果
    │   ├── training_results.json
    │   └── [模型文件]
    ├── pure_maddpg/                     # 纯MADDPG结果  
    │   ├── training_results.json
    │   └── [模型文件]
    ├── pure_llm/                        # 纯LLM结果
    │   ├── training_results.json
    │   └── [模型文件]
    ├── comparison/                      # 对比分析
    │   ├── test_results.json           # 测试结果
    │   ├── comparison_results.csv      # 对比表格
    │   └── full_report.json           # 完整报告
    ├── plots/                          # 可视化图表
    │   └── comparison_plots.png        # 性能对比图
    └── logs/                           # 日志文件
```

## 📊 **性能指标说明**

### **训练指标：**
- **Episode Rewards** - 每轮平均奖励
- **Latency** - 任务完成时延
- **Energy Consumption** - 设备能量消耗  
- **Completion Rate** - 任务完成率
- **Training Time** - 训练耗时

### **测试指标：**
- **Test Energy** - 测试阶段平均能耗
- **Test Delay** - 测试阶段平均时延
- **Test Utilization** - 资源利用率

## 🎯 **实验配置建议**

### **标准实验配置：**
```yaml
training:
  episodes: 1000              # 充分训练
  max_steps_per_episode: 100  # 每轮步数
  batch_size: 64
  learning_rate: 0.001

environment:
  num_devices: 10             # 端设备数量
  num_edge_servers: 5         # 边缘服务器数量
  num_cloud_servers: 1        # 云服务器数量
```

### **快速测试配置（--quick模式）：**
```yaml
training:
  episodes: 50                # 快速验证
  max_steps_per_episode: 20   # 减少步数
```

## 🔬 **实验流程最佳实践**

### **1. 开发阶段：**
```bash
# 快速验证算法正确性
python main.py --mode all --quick --no-plots
```

### **2. 正式实验：**
```bash
# 完整实验流程
python main.py --mode all
```

### **3. 结果分析：**
- 查看 `comparison_results.csv` 了解数值对比
- 查看 `comparison_plots.png` 了解可视化结果
- 查看 `full_report.json` 了解详细数据

### **4. 论文写作：**
- 使用CSV数据制作表格
- 使用PNG图表插入论文
- 引用JSON中的具体数值

## 🚨 **常见问题解决**

### **1. 训练失败**
```bash
# 检查配置文件
python main.py --mode llm_maddpg_only --quick  # 单独测试

# 查看详细错误信息
python main.py --mode all 2>&1 | tee experiment.log
```

### **2. LLM连接问题**
- 检查 `config.yaml` 中的LLM服务配置
- 确保LLM服务器可访问
- 使用 `--quick` 模式减少LLM调用次数

### **3. 内存不足**
```yaml
# 减少批量大小
training:
  batch_size: 32  # 从64减少到32
  buffer_size: 50000  # 减少缓冲区大小
```

### **4. 训练时间过长**
```bash
# 使用快速模式
python main.py --mode all --quick

# 或单独训练核心算法
python main.py --mode llm_maddpg_only
```

## 📋 **实验检查清单**

**实验前：**
- [ ] 配置文件参数合理
- [ ] LLM服务正常运行
- [ ] 磁盘空间充足
- [ ] Python环境完整

**实验中：**
- [ ] 监控训练进度
- [ ] 检查内存使用
- [ ] 观察收敛情况

**实验后：**
- [ ] 验证结果合理性
- [ ] 备份重要数据
- [ ] 整理实验报告

## 🎉 **成功示例**

运行成功后，您将看到类似输出：
```
✅ 实验完成！
📁 所有结果保存在: results/experiment_20231201_143022
📊 对比报告: results/experiment_20231201_143022/comparison/
📈 可视化图表: results/experiment_20231201_143022/plots/

📈 简要性能对比:
------------------------------------------------------------
LLM+MADDPG   | 能耗: 0.1234 | 时延: 0.5678
纯MADDPG     | 能耗: 0.1456 | 时延: 0.6789  
纯LLM        | 能耗: 0.1678 | 时延: 0.7890
```

现在您只需要运行 `python main.py --mode all` 就能完成整个算法对比实验！ 