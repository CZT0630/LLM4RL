"""
LLM响应解析测试脚本
测试改进后的prompt和解析功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_assistant.llm_client import LLMClient
from llm_assistant.response_parser import ResponseParser
from utils.config import load_config
import numpy as np

def test_llm_parsing():
    """测试LLM的prompt和响应解析"""
    print("🧪 开始测试LLM解析功能...")
    
    # 加载配置
    config = load_config()
    
    # 创建LLM客户端
    llm_client = LLMClient(
        model_name=config['llm']['model_name'],
        server_url=config['llm']['server_url'],
        timeout_connect=config['llm'].get('timeout_connect', 120),
        timeout_read=config['llm'].get('timeout_read', 300),
        use_mock=False  # 强制使用真实LLM测试
    )
    
    # 模拟环境状态
    device_info = [
        {"cpu": 2.0, "memory": 4.0},
        {"cpu": 2.0, "memory": 4.0},
        {"cpu": 2.0, "memory": 4.0}
    ]
    edge_info = [
        {"cpu": 8.0, "memory": 16.0},
        {"cpu": 8.0, "memory": 16.0}
    ]
    cloud_info = [
        {"cpu": 32.0, "memory": 64.0}
    ]
    
    # 构造简单的环境状态
    env_state = np.concatenate([
        # 设备状态 (3个设备, 每个4个值)
        [0.0, 0.0, 1.0, 0.0] * 3,  # 设备状态
        # 边缘服务器状态 (2个边缘, 每个3个值)
        [0.0, 0.0, 0.0] * 2,  # 边缘状态
        # 云服务器状态 (1个云, 3个值)
        [0.0, 0.0, 0.0],  # 云状态
        # 任务状态 (3个任务, 每个3个值)
        [400.0, 50.0, 30.0],  # 任务0: 400MI, 50MB, 30s
        [800.0, 20.0, 45.0],  # 任务1: 800MI, 20MB, 45s
        [200.0, 80.0, 60.0],  # 任务2: 200MI, 80MB, 60s
    ])
    
    print("\n📤 测试LLM请求和响应...")
    
    try:
        # 获取LLM策略
        strategies = llm_client.get_unload_strategy(env_state, device_info, edge_info, cloud_info)
        
        print(f"\n✅ LLM策略获取成功！")
        print(f"📋 获得 {len(strategies)} 个卸载策略:")
        
        for strategy in strategies:
            task_id = strategy.get('task_id', '未知')
            offload_ratio = strategy.get('offload_ratio', 0.0)
            target_node = strategy.get('target_node', 0)
            
            target_name = "本地"
            if target_node == 1:
                target_name = "边缘服务器0"
            elif target_node == 2:
                target_name = "边缘服务器1"
            elif target_node == 3:
                target_name = "云服务器"
                
            print(f"  任务{task_id}: 卸载比例={offload_ratio:.2f}, 目标={target_name}")
        
        # 使用ResponseParser进一步验证
        print(f"\n🔍 使用ResponseParser验证解析结果...")
        parsed_strategies = ResponseParser.parse_unload_strategy(
            strategies, len(device_info), len(edge_info), len(cloud_info)
        )
        
        if len(parsed_strategies) == len(device_info):
            print(f"✅ 解析验证成功！所有{len(device_info)}个任务都有对应策略")
        else:
            print(f"⚠️ 解析验证警告：期望{len(device_info)}个策略，实际得到{len(parsed_strategies)}个")
            
        return True
        
    except Exception as e:
        print(f"❌ LLM测试失败: {e}")
        
        print(f"\n🤖 回退到模拟策略测试...")
        # 测试模拟策略生成
        mock_strategies = llm_client._generate_mock_strategies(env_state, device_info, edge_info, cloud_info)
        
        print(f"✅ 模拟策略生成成功！")
        print(f"📋 获得 {len(mock_strategies)} 个模拟策略:")
        
        for strategy in mock_strategies:
            task_id = strategy.get('task_id', '未知')
            offload_ratio = strategy.get('offload_ratio', 0.0)
            target_node = strategy.get('target_node', 0)
            
            target_name = "本地"
            if target_node == 1:
                target_name = "边缘服务器0"
            elif target_node == 2:
                target_name = "边缘服务器1"
            elif target_node == 3:
                target_name = "云服务器"
                
            print(f"  任务{task_id}: 卸载比例={offload_ratio:.2f}, 目标={target_name}")
            
        return False

def test_json_parsing():
    """测试JSON解析函数"""
    print("\n🧪 测试JSON解析功能...")
    
    llm_client = LLMClient()
    
    # 测试用例1: 标准JSON数组
    test1 = '[{"task_id":0,"offload_ratio":0.8,"target_node":1},{"task_id":1,"offload_ratio":1.0,"target_node":3}]'
    result1 = llm_client._extract_json_from_text(test1)
    print(f"测试1 - 标准JSON: {'✅ 成功' if len(result1) == 2 else '❌ 失败'}")
    
    # 测试用例2: 包含思考过程的文本
    test2 = """我需要分析这些任务...
    
    [{"task_id": 0, "offload_ratio": 0.7, "target_node": 2}, {"task_id": 1, "offload_ratio": 1.0, "target_node": 3}]
    
    这样的策略比较合理。"""
    result2 = llm_client._extract_json_from_text(test2)
    print(f"测试2 - 包含文本: {'✅ 成功' if len(result2) == 2 else '❌ 失败'}")
    
    # 测试用例3: 格式不完整的文本
    test3 = """任务0: offload_ratio: 0.8, target_node: 1
    任务1: offload_ratio: 1.0, target_node: 3"""
    result3 = llm_client._extract_json_from_text(test3)
    print(f"测试3 - 格式不完整: {'✅ 成功' if len(result3) >= 1 else '❌ 失败'}")
    
    print(f"✅ JSON解析测试完成")

if __name__ == "__main__":
    print("🚀 开始LLM解析功能测试")
    print("=" * 50)
    
    # 测试JSON解析功能
    test_json_parsing()
    
    # 测试完整的LLM交互
    llm_success = test_llm_parsing()
    
    print("\n" + "=" * 50)
    if llm_success:
        print("🎉 所有测试通过！LLM连接和解析功能正常")
    else:
        print("⚠️ LLM连接失败，但模拟策略功能正常")
    print("🏁 测试完成") 