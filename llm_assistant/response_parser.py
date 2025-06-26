# llm_assistant/response_parser.py
import json

class ResponseParser:
    @staticmethod
    def parse_unload_strategy(response, num_devices, num_edges, num_clouds):
        """解析LLM返回的卸载策略
        
        Args:
            response: LLM返回的策略JSON或字符串
            num_devices: 设备数量
            num_edges: 边缘服务器数量
            num_clouds: 云服务器数量
        
        Returns:
            解析后的卸载策略列表
        """
        print(f"开始解析LLM卸载策略，设备数: {num_devices}")
        
        # 如果response已经是列表，直接使用
        if isinstance(response, list):
            strategies = response
        else:
            try:
                # 如果是字符串，先尝试解析JSON
                strategies = json.loads(response)
                if not isinstance(strategies, list):
                    strategies = [strategies]
            except Exception as e:
                print(f"解析卸载策略失败: {e}")
                return []
        
        # 标准化策略格式
        parsed_strategies = []
        
        # 遍历所有设备，确保每个设备都有对应的策略
        for i in range(num_devices):
            # 查找当前设备的策略
            task_strategy = next((s for s in strategies if s.get('task_id') == i), None)
            
            if not task_strategy:
                # 如果没有找到对应的策略，使用默认策略（本地执行）
                parsed_strategies.append({
                    "task_id": i,
                    "offload_ratio": 0.0,
                    "target_node": 0  # 本地执行
                })
                continue
            
            # 检查并转换策略格式
            # 格式1：使用local_ratio, edge_ratio, cloud_ratio和target_edge
            if all(k in task_strategy for k in ["local_ratio", "edge_ratio", "cloud_ratio", "target_edge"]):
                local_ratio = float(task_strategy.get("local_ratio", 0.0))
                edge_ratio = float(task_strategy.get("edge_ratio", 0.0))
                cloud_ratio = float(task_strategy.get("cloud_ratio", 0.0))
                
                # 计算总卸载比例（边缘+云）
                offload_ratio = edge_ratio + cloud_ratio
                
                # 确定目标节点
                if offload_ratio > 0:
                    if edge_ratio > cloud_ratio:
                        # 主要卸载到边缘
                        target_edge = task_strategy.get("target_edge", 0)
                        target_node = target_edge + 1  # 转换为1-based索引
                    else:
                        # 主要卸载到云
                        target_node = num_edges + 1  # 云节点索引
                else:
                    # 不卸载
                    target_node = 0
            else:
                # 格式2：直接使用offload_ratio和target_node
                offload_ratio = task_strategy.get("offload_ratio", 0.0)
                target_node = task_strategy.get("target_node", 0)
            
            # 确保比例在[0,1]范围内
            offload_ratio = max(0.0, min(1.0, float(offload_ratio)))
            
            # 确保目标节点在合法范围内
            target_node = max(0, min(num_edges + num_clouds, int(target_node)))
            
            parsed_strategies.append({
                "task_id": i,
                "offload_ratio": offload_ratio,
                "target_node": target_node
            })
        
        print(f"解析完成，共{len(parsed_strategies)}个策略")
        return parsed_strategies