# main.py
import argparse
from experiments.train_llm_maddpg import train_llm_maddpg
from experiments.test_llm_maddpg import test_llm_maddpg
from utils.config import load_config
# from utils.config import get_default_config as load_config


def main():
    parser = argparse.ArgumentParser(description='云边端计算卸载 with LLM-MADDPG')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--model_path', type=str, default='results', help='模型保存/加载路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    if args.mode == 'train':
        print("开始训练...")
        train_llm_maddpg(config)
    else:
        print("开始测试...")
        test_llm_maddpg(args.model_path, config)


if __name__ == "__main__":
    main()