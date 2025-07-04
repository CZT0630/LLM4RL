# llm_assistant/response_parser.py
import json
import re

class ResponseParser:
    @staticmethod
    def parse_unload_strategy(response, num_devices, num_edges, num_clouds):
        """解析LLM返回的卸载策略（支持markdown格式和多种嵌套结构）
        
        Args:
            response: LLM返回的策略JSON或字符串
            num_devices: 设备数量
            num_edges: 边缘服务器数量
            num_clouds: 云服务器数量
        
        Returns:
            解析后的卸载策略列表，格式为:
            [
                {
                    "device_id": 0,
                    "local_ratio": 0.3,
                    "edge_ratio": 0.5,
                    "cloud_ratio": 0.2,
                    "target_edge": 1
                }, ...
            ]
        """
        print(f"开始解析LLM卸载策略（格式1：三元分割），设备数: {num_devices}")
        
        # 如果response已经是列表，直接使用
        if isinstance(response, list):
            strategies = response
        else:
            # 尝试多种解析策略
            strategies = ResponseParser._extract_strategies_from_text(response)
            
        if not strategies:
            print("⚠️ 无法解析LLM响应中的有效策略，使用默认策略")
            return ResponseParser._generate_default_strategies(num_devices)
        
        # 标准化策略格式
        parsed_strategies = []
        
        # 遍历所有设备，确保每个设备都有对应的策略
        for i in range(num_devices):
            # 查找当前设备的策略
            device_strategy = next(
                (s for s in strategies if s.get('device_id') == i), 
                None
            )
            
            if not device_strategy:
                # 如果没有找到对应的策略，使用默认策略（全云端执行）
                parsed_strategies.append({
                    "device_id": i,
                    "local_ratio": 0.0,
                    "edge_ratio": 0.0,
                    "cloud_ratio": 1.0,
                    "target_edge": 0
                })
                print(f"  设备{i}: 未找到策略，使用默认（全云端执行）")
                continue
            
            # 转换字段名（处理不同的命名格式）
            device_strategy = ResponseParser._convert_field_names(device_strategy)
            
            # 解析格式1：三元分割格式
            if all(k in device_strategy for k in ["local_ratio", "edge_ratio", "cloud_ratio"]):
                local_ratio = float(device_strategy.get("local_ratio", 0.0))
                edge_ratio = float(device_strategy.get("edge_ratio", 0.0))
                cloud_ratio = float(device_strategy.get("cloud_ratio", 0.0))
                target_edge = int(device_strategy.get("target_edge", 0))
                
                # 归一化比例，确保和为1
                total = local_ratio + edge_ratio + cloud_ratio
                if total > 0:
                    local_ratio = local_ratio / total
                    edge_ratio = edge_ratio / total
                    cloud_ratio = cloud_ratio / total
                else:
                    # 如果所有比例都为0，默认全云端执行
                    local_ratio, edge_ratio, cloud_ratio = 0.0, 0.0, 1.0
                
                # 确保目标边缘服务器在合法范围内
                target_edge = max(0, min(num_edges - 1, target_edge))
                
                parsed_strategies.append({
                    "device_id": i,
                    "local_ratio": local_ratio,
                    "edge_ratio": edge_ratio,
                    "cloud_ratio": cloud_ratio,
                    "target_edge": target_edge
                })
                
                print(f"  设备{i}: 本地{local_ratio:.2f}, 边缘{edge_ratio:.2f}, 云端{cloud_ratio:.2f} → Edge{target_edge}")
            else:
                # 如果格式不正确，使用默认策略（全云端执行）
                parsed_strategies.append({
                    "device_id": i,
                    "local_ratio": 0.0,
                    "edge_ratio": 0.0,
                    "cloud_ratio": 1.0,
                    "target_edge": 0
                })
                print(f"  设备{i}: 格式不正确，使用默认（全云端执行）")
        
        print(f"解析完成，共{len(parsed_strategies)}个策略")
        return parsed_strategies
    
    @staticmethod
    def _extract_strategies_from_text(text):
        """从文本中提取策略信息，支持多种格式"""
        if not text:
            return []
        
        # 策略1: 直接解析为JSON
        try:
            data = json.loads(text)
            if isinstance(data, list):
                print("✅ 策略1成功：直接解析为JSON列表")
                return data
            elif isinstance(data, dict) and 'strategies' in data:
                print("✅ 策略1成功：从JSON对象的strategies字段提取")
                return data['strategies']
        except:
            pass
        
        # 策略2: 从markdown代码块中提取JSON
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_content = json_match.group(1).strip()
                data = json.loads(json_content)
                if isinstance(data, dict) and 'strategies' in data:
                    print("✅ 策略2成功：从markdown代码块中提取strategies")
                    return data['strategies']
                elif isinstance(data, list):
                    print("✅ 策略2成功：从markdown代码块中提取列表")
                    return data
        except Exception as e:
            print(f"策略2失败: {e}")
        
        # 策略3: 查找任何包含strategies的JSON对象
        try:
            strategies_match = re.search(r'"strategies"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if strategies_match:
                strategies_content = f'[{strategies_match.group(1)}]'
                strategies = json.loads(strategies_content)
                print("✅ 策略3成功：提取strategies数组")
                return strategies
        except Exception as e:
            print(f"策略3失败: {e}")
        
        # 策略4: 查找JSON数组格式
        try:
            array_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
            if array_match:
                array_content = array_match.group(0)
                strategies = json.loads(array_content)
                print("✅ 策略4成功：提取JSON数组")
                return strategies
        except Exception as e:
            print(f"策略4失败: {e}")
        
        # 策略5: 查找单个JSON对象并构造列表
        try:
            obj_matches = re.findall(r'\{[^{}]*"device_id"[^{}]*\}', text)
            if obj_matches:
                strategies = []
                for obj_str in obj_matches:
                    obj = json.loads(obj_str)
                    strategies.append(obj)
                print(f"✅ 策略5成功：提取{len(strategies)}个JSON对象")
                return strategies
        except Exception as e:
            print(f"策略5失败: {e}")
        
        print("❌ 所有解析策略均失败")
        return []
    
    @staticmethod
    def _convert_field_names(strategy):
        """转换字段名，统一不同的命名格式"""
        converted = strategy.copy()
        
        # 转换target_edge_server为target_edge
        if 'target_edge_server' in converted:
            converted['target_edge'] = converted.pop('target_edge_server')
        
        # 处理target_edge为-1的情况（表示不使用边缘服务器）
        if converted.get('target_edge', 0) == -1:
            converted['target_edge'] = 0
        
        # 移除额外的字段
        extra_fields = ['rationale', 'expected_latency', 'battery_impact']
        for field in extra_fields:
            converted.pop(field, None)
        
        return converted
    
    @staticmethod
    def _generate_default_strategies(num_devices):
        """生成默认策略（全云端执行）"""
        return [
            {
                "device_id": i,
                "local_ratio": 0.0,
                "edge_ratio": 0.0,
                "cloud_ratio": 1.0,
                "target_edge": 0
            }
            for i in range(num_devices)
        ]