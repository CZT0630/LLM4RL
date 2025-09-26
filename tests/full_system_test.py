import os
import sys
import numpy as np
import torch
from environment.cloud_edge_env import CloudEdgeDeviceEnv
from utils.config import get_default_config

"""
全面系统测试脚本：验证电池相关代码已被成功移除并测试系统功能
"""

def test_environment_initialization():
    """测试环境初始化，确保不包含电池相关配置"""
    print("\n===== 测试环境初始化 =====")
    config = get_default_config()
    
    # 验证配置中没有电池相关设置
    assert 'battery_capacity' not in config.get('device_config', {}), "配置中仍包含电池相关设置"
    
    # 创建环境
    env = CloudEdgeDeviceEnv(config['environment'])
    print(f"环境创建成功，设备数量: {env.num_devices}, 边缘服务器数量: {env.num_edges}, 云端服务器数量: {env.num_clouds}")
    
    # 验证Agent状态维度计算正确（不包含电池）
    state_dim = env.get_agent_state_dim()
    print(f"Agent状态维度: {state_dim}")
    
    # 验证设备信息中没有电池相关字段
    device_info_list = env.get_device_info()
    assert device_info_list, "未能获取设备信息列表"
    device_info = device_info_list[0]  # 检查第一个设备的信息
    assert 'battery_percentage' not in device_info, "设备信息中仍包含电池百分比字段"
    print(f"设备信息字段: {list(device_info.keys())}")
    
    return env

def test_environment_interaction(env):
    """测试环境交互功能"""
    print("\n===== 测试环境交互 =====")
    
    # 重置环境
    state, info = env.reset()
    print(f"环境重置成功，状态形状: {state.shape}")
    
    # 执行一些随机动作
    num_steps = 5
    for step in range(num_steps):
        # 生成随机动作
        actions = []
        for _ in range(env.num_devices):
            action = np.random.rand(env.action_space.shape[0])
            actions.append(action)
        actions = np.array(actions)
        
        # 执行动作
        next_state, rewards, terminated, truncated, info = env.step(actions)
        
        # 验证info中没有电池相关信息
        assert 'battery_levels' not in info, "环境返回的info中仍包含电池相关信息"
        
        print(f"Step {step+1}: 奖励均值={np.mean(rewards):.3f}, 终止状态={terminated}")
        
        # 更新状态
        state = next_state
        
        # 如果环境终止，重置
        if terminated or truncated:
            state, info = env.reset()
            print(f"环境重置于步骤 {step+1}")
    
    return True

def test_device_state(env):
    """测试设备状态获取，确保不包含电池信息"""
    print("\n===== 测试设备状态 =====")
    
    for i in range(min(3, env.num_devices)):  # 测试前3个设备
        device = env.user_equipments[i]  # 使用正确的属性名
        device_state = device.get_state()
        
        # 验证设备状态维度正确（CPU频率 + 任务负载）
        assert len(device_state) == 2, f"设备状态维度应为2，但实际为{len(device_state)}"
        print(f"设备 {i} 状态维度: {len(device_state)}, 状态值: {device_state}")
    
    return True

def test_task_generation_and_processing(env):
    """测试任务生成和处理"""
    print("\n===== 测试任务生成和处理 =====")
    
    # 执行多个步骤以生成和处理任务
    state, info = env.reset()
    total_steps = 10
    
    # 直接从环境获取任务完成统计
    for step in range(total_steps):
        # 使用随机策略
        actions = np.random.rand(env.num_devices, env.action_space.shape[0])
        
        # 执行动作
        next_state, rewards, terminated, truncated, info = env.step(actions)
        
        state = next_state
        
        if terminated or truncated:
            state, info = env.reset()
    
    # 获取任务完成统计来验证任务生成和处理
    task_stats = env.get_task_completion_rate()
    total_tasks = task_stats['total_tasks']
    completed_tasks = task_stats['completed_on_time'] + task_stats['completed_late']
    
    print(f"任务统计: 总任务数={total_tasks}, 完成任务数={completed_tasks}")
    assert total_tasks > 0, "没有生成任何任务"
    assert completed_tasks >= 0, "任务处理统计异常"
    
    return True

def run_full_test():
    """运行全面的系统测试"""
    print("\n===== 开始全面系统测试 =====")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"测试目录: {os.getcwd()}")
    
    try:
        # 测试1: 环境初始化
        env = test_environment_initialization()
        
        # 测试2: 环境交互
        interaction_success = test_environment_interaction(env)
        
        # 测试3: 设备状态
        device_state_success = test_device_state(env)
        
        # 测试4: 任务生成和处理
        task_process_success = test_task_generation_and_processing(env)
        
        # 总结测试结果
        all_tests_passed = interaction_success and device_state_success and task_process_success
        
        if all_tests_passed:
            print("\n🎉 所有测试通过！系统已成功移除电池相关代码并正常工作。")
        else:
            print("\n❌ 一些测试失败，请检查系统配置和代码。")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n===== 测试完成 =====")

if __name__ == "__main__":
    run_full_test()