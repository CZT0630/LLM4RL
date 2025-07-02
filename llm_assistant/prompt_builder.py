# llm_assistant/prompt_builder.py
"""
LLM提示词构建器 - 简化设备模型版本
适配UE、ES、CS的简化属性和差异化通信延迟
"""
import json


class PromptBuilder:
    """LLM提示词构建器"""
    
    @staticmethod
    def build_offloading_strategy_prompt(env_state, device_info, edge_info, cloud_info, tasks_info):
        """
        构建计算卸载策略提示词 - 简化设备模型版本
        
        Args:
            env_state: 环境状态信息
            device_info: UE设备信息列表
            edge_info: ES边缘服务器信息列表  
            cloud_info: CS云服务器信息列表
            tasks_info: 当前任务信息列表
            
        Returns:
            str: 完整的提示词
        """
        
        prompt = f"""你是一个云边端计算卸载系统的专家。当前系统采用简化的设备模型：

## 🏗️ **系统架构简化说明**

### 端侧设备(UE)属性：
- **CPU频率**: 计算能力指标
- **电池状态**: 能耗约束限制  
- **任务负载**: 当前执行剩余时间 + 队列中所有任务处理时间总和

### 边缘服务器(ES)属性：
- **CPU频率**: 处理能力指标
- **任务负载**: 当前繁忙程度和预期等待时间

### 云服务器(CS)属性：
- **CPU频率**: 处理能力（资源无限）
- **无任务负载**: 云资源视为无限，任务可立即执行

### 🚀 **通信延迟差异化**
- **边缘卸载**: UE→ES直连，1Gbps，低延迟
- **云端卸载**: UE→ES→CS中转，100Mbps总带宽，高延迟
- **设计意图**: 平衡计算能力与通信成本

## 📊 **当前系统状态**

### UE设备状态 ({len(device_info)}个设备):
"""

        # UE设备状态
        for i, device in enumerate(device_info):
            battery_status = "🔴低电量" if device['battery_percentage'] < 0.3 else "🟡中等" if device['battery_percentage'] < 0.7 else "🟢充足"
            load_status = "🔴繁忙" if device['task_load'] > 30 else "🟡适中" if device['task_load'] > 10 else "🟢空闲"
            
            prompt += f"""
UE{device['device_id']}: CPU={device['cpu_frequency']:.1f}GHz, 电池={device['battery_percentage']:.0%}({battery_status}), 任务负载={device['task_load']:.1f}s({load_status})"""

        prompt += f"""

### ES边缘服务器状态 ({len(edge_info)}个服务器):
"""

        # ES边缘服务器状态
        for i, server in enumerate(edge_info):
            load_status = "🔴繁忙" if server['task_load'] > 60 else "🟡适中" if server['task_load'] > 20 else "🟢空闲"
            
            prompt += f"""
ES{server['server_id']}: CPU={server['cpu_frequency']}GHz, 任务负载={server['task_load']:.1f}s({load_status})"""

        prompt += f"""

### CS云服务器状态:
"""

        # CS云服务器状态
        for i, server in enumerate(cloud_info):
            prompt += f"""
CS{server['server_id']}: CPU={server['cpu_frequency']}GHz, 状态=🟢无限资源"""

        prompt += f"""

## 📋 **当前任务队列** ({len(tasks_info)}个任务):
"""

        # 任务信息
        for task in tasks_info:
            urgency = "🔴紧急" if task['remaining_time'] < 10 else "🟡一般" if task['remaining_time'] < 30 else "🟢宽松"
            size_level = "小型" if task['data_size'] <= 5 else "中型" if task['data_size'] <= 50 else "大型"
            
            prompt += f"""
Device{task['device_id']}: {size_level}任务({task['data_size']:.1f}MB, {task['cpu_cycles']/1e9:.2f}Gcycles), 剩余{task['remaining_time']:.1f}s({urgency})"""

        prompt += """

## 🎯 **卸载策略优化目标**

### 主要考虑因素：
1. **时延优化**: 减少任务完成时间，考虑等待+计算+通信时间
2. **能耗约束**: UE电池限制，平衡计算与传输能耗
3. **负载均衡**: 避免边缘服务器过载，充分利用云资源
4. **通信效率**: 权衡边缘快速通信vs云端高性能计算
5. **截止时间**: 确保任务在规定时间内完成

### 决策权衡：
- **本地执行**: 零通信延迟，消耗UE电池和计算时间
- **边缘卸载**: 低通信延迟，边缘服务器可能排队等待
- **云端卸载**: 高通信延迟，无计算等待但需要中转

## 🤖 **请为每个设备提供卸载策略**

为每个UE设备分析当前状况并给出最优的任务分割策略：

**输出格式**：
```json
{
  "strategies": [
    {
      "device_id": 0,
      "rationale": "分析设备状况和任务特点的决策理由",
      "local_ratio": 0.3,
      "edge_ratio": 0.5, 
      "cloud_ratio": 0.2,
      "target_edge_server": 2,
      "expected_latency": "预估总延迟(秒)",
      "battery_impact": "电池影响评估"
    }
  ]
}
```

**决策要点**：
1. 电池低的设备优先卸载到边缘/云端
2. 紧急任务考虑通信延迟成本
3. 大型任务充分利用云端并行能力
4. 负载均衡选择空闲的边缘服务器
5. 综合考虑设备能力、任务特点、网络条件

请分析并给出具体的卸载策略建议："""

        return prompt

    @staticmethod
    def build_system_analysis_prompt(env_state, performance_metrics):
        """
        构建系统性能分析提示词
        
        Args:
            env_state: 环境状态信息
            performance_metrics: 性能指标数据
            
        Returns:
            str: 系统分析提示词
        """
        
        prompt = f"""请分析当前云边端计算卸载系统的性能表现：

## 📈 **性能指标**

### 时延性能：
- 平均任务完成时延: {performance_metrics.get('avg_latency', 'N/A')}s
- 通信延迟占比: {performance_metrics.get('comm_latency_ratio', 'N/A')}%
- 计算延迟占比: {performance_metrics.get('comp_latency_ratio', 'N/A')}%

### 能耗性能：
- 总能耗消耗: {performance_metrics.get('total_energy', 'N/A')}J
- 平均设备电池消耗: {performance_metrics.get('avg_battery_consumption', 'N/A')}%

### 系统效率：
- 截止时间满足率: {performance_metrics.get('deadline_satisfaction', 'N/A')}%
- 边缘服务器平均负载: {performance_metrics.get('avg_edge_load', 'N/A')}%
- 云端卸载比例: {performance_metrics.get('cloud_offload_ratio', 'N/A')}%

## 🔍 **请分析以下问题**：

1. **性能瓶颈识别**: 当前系统的主要瓶颈是什么？
2. **资源利用效率**: 边缘vs云端资源利用是否合理？
3. **通信延迟影响**: 差异化通信延迟是否有效平衡了卸载选择？
4. **改进建议**: 针对简化设备模型，有什么优化建议？

请提供详细的分析和建议。"""

        return prompt

    @staticmethod  
    def build_device_status_prompt(device_info):
        """
        构建设备状态监控提示词
        
        Args:
            device_info: 设备信息列表
            
        Returns:
            str: 设备状态分析提示词
        """
        
        prompt = """请分析当前设备状态并给出监控建议：

## 📱 **设备状态详情**

"""
        
        for device in device_info:
            status_indicators = []
            
            # 电池状态分析
            if device['battery_percentage'] < 0.2:
                status_indicators.append("🔴电池危险")
            elif device['battery_percentage'] < 0.5:
                status_indicators.append("🟡电池偏低")
                
            # 负载状态分析  
            if device['task_load'] > 30:
                status_indicators.append("🔴高负载")
            elif device['task_load'] > 15:
                status_indicators.append("🟡中负载")
                
            # CPU能力评估
            if device['cpu_frequency'] < 0.7:
                status_indicators.append("⚡低性能")
            elif device['cpu_frequency'] > 0.9:
                status_indicators.append("⚡高性能")
                
            status_str = " ".join(status_indicators) if status_indicators else "🟢正常"
            
            prompt += f"""
**UE{device['device_id']}**: CPU={device['cpu_frequency']:.1f}GHz, 电池={device['battery_percentage']:.0%}, 负载={device['task_load']:.1f}s
状态: {status_str}
"""

        prompt += """

## 🔔 **请提供**：
1. **告警设备**: 需要重点关注的设备
2. **负载建议**: 高负载设备的卸载建议  
3. **能耗管理**: 低电量设备的节能策略
4. **性能优化**: 整体设备性能优化建议

请给出具体的监控和优化建议。"""

        return prompt