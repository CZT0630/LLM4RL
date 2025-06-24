# LLM4RL: 基于LLM辅助的多智能体任务卸载系统（云-边-端三层）

## 项目简介
本项目实现了一个在边缘计算环境下，结合大语言模型（LLM）辅助的多智能体任务卸载系统。系统采用云-边-端三层架构，利用LLM生成卸载策略建议，并通过MADDPG算法实现多智能体的具体卸载决策。

## 主要功能
- 支持多终端设备、边缘服务器、云端服务器的三层异构环境建模
- LLM（如GPT）根据环境状态生成卸载建议，辅助智能体决策
- MADDPG多智能体强化学习实现任务卸载与资源分配
- 支持训练与测试流程，支持结果可视化

## 目录结构
```
LLM4RL/
├── main.py                # 主入口
├── config.yaml            # 配置文件
├── README.md              # 项目说明
├── algos/                 # MADDPG算法相关
├── environment/           # 云-边-端环境与设备模型
├── experiments/           # 训练与测试脚本
├── llm_assistant/         # LLM交互与建议解析
├── utils/                 # 工具函数与可视化
```

## 依赖安装
建议使用Python 3.8+，安装依赖：
```bash
pip install -r requirements.txt
```
如需使用OpenAI GPT，需安装openai库并配置API Key。

## 配置说明
- `config.yaml` 包含环境、LLM、MADDPG、训练与测试等参数。
- 需在llm.api_key处填写你的OpenAI API密钥。

## 运行方法
### 训练
```bash
python main.py --mode train --config config.yaml --model_path results
```
### 测试
```bash
python main.py --mode test --config config.yaml --model_path results
```

## 主要参数说明
- `environment`：环境设备数量、资源参数、任务参数
- `llm`：LLM模型名称、API密钥、建议频率
- `maddpg`：学习率、折扣因子、缓冲区、训练步数等

## 结果可视化
训练过程中会自动保存奖励曲线和动作分布图于results目录。

## 参考
- MADDPG: Multi-Agent Deep Deterministic Policy Gradient
- OpenAI GPT系列

## 联系方式
如有问题请联系作者。
