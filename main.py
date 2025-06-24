# main.py
import argparse
from experiments.train_llm_maddpg import train_llm_maddpg
from experiments.train_maddpg import train_maddpg
from experiments.train_llm import train_llm
from experiments.test_llm_maddpg import test_llm_maddpg
from experiments.test_maddpg import test_maddpg
from experiments.test_llm import test_llm
from utils.config import load_config
# from utils.config import get_default_config as load_config
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='云边端计算卸载 with LLM-MADDPG')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'maddpg', 'llm'], help='运行模式')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--model_path', type=str, default='results', help='模型保存/加载路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    set_seed(config.get('seed', 42))

    if args.mode == 'train':
        print("开始训练 LLM+MADDPG ...")
        train_llm_maddpg(config)
    elif args.mode == 'test':
        print("开始测试 ...")
        test_llm_maddpg(args.model_path, config)
    elif args.mode == 'maddpg':
        print("开始训练纯MADDPG ...")
        train_maddpg(config)
    elif args.mode == 'llm':
        print("开始训练纯LLM ...")
        train_llm(config)


if __name__ == "__main__":
    main()