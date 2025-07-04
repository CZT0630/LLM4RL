# llm_assistant/llm_client.py
import requests
import json
import time
import numpy as np
import re


class LLMClient:
    """LLM客户端类 - 适配简化设备模型"""
    
    def __init__(self, api_key="", model_name="qwen3-14b", server_url="http://10.200.1.35:8888/v1/completions",
                 timeout_connect=120, timeout_read=300, use_mock=True, config=None):
    # def __init__(self, api_key="sk-1907a18fea6640c6aac5b4194920169f", model_name="qwen-plus", server_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    #              timeout_connect=120, timeout_read=300, use_mock=True, config=None):
        """
        初始化LLM客户端
        
        Args:
            api_key: API密钥
            model_name: 模型名称
            server_url: LLM服务器URL
            timeout_connect: 连接超时时间（秒）
            timeout_read: 读取超时时间（秒）
            use_mock: 是否在失败时使用模拟模式
            config: 配置字典，用于读取max_tokens等参数
        """
        # 优先从配置文件读取LLM设置
        if config and 'llm' in config:
            llm_config = config['llm']
            self.api_key = llm_config.get('api_key', api_key)
            self.model_name = llm_config.get('model_name', model_name)
            self.server_url = llm_config.get('base_url', server_url)
            self.timeout_connect = llm_config.get('timeout', timeout_connect)
            self.timeout_read = llm_config.get('read_timeout', timeout_read)
            self.max_tokens = llm_config.get('max_tokens', 4096)
            self.temperature = llm_config.get('temperature', 0.3)
        else:
            # 使用传入的参数或默认值
            self.api_key = api_key
            self.model_name = model_name
            self.server_url = server_url
            self.timeout_connect = timeout_connect
            self.timeout_read = timeout_read
            self.max_tokens = 4096
            self.temperature = 0.3
        
        self.use_mock = use_mock
        
        # 判断API类型（根据URL判断使用哪种API格式）
        self.is_chat_api = '/chat/completions' in self.server_url
        self.is_completions_api = '/completions' in self.server_url and '/chat/completions' not in self.server_url
        
        print(f"✅ LLM客户端初始化:")
        print(f"  服务器: {self.server_url}")
        print(f"  API类型: {'Chat Completions' if self.is_chat_api else 'Completions' if self.is_completions_api else 'Unknown'}")
        print(f"  模型: {self.model_name}")
        print(f"  最大tokens: {self.max_tokens}")
        print(f"  温度: {self.temperature}")
        print(f"  超时设置: 连接{self.timeout_connect}s, 读取{self.timeout_read}s")
        print(f"  模拟模式: {'启用' if self.use_mock else '禁用'}")

    def query(self, prompt):
        """向LLM服务器发送查询请求"""
        print(f"📡 向LLM服务器发送查询请求...")
        print(f"🌐 服务器: {self.server_url}")
        print(f"🤖 模型: {self.model_name}")
        
        headers = {
            'Content-Type': 'application/json',
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        # 根据API类型选择请求格式
        if self.is_chat_api:
            # Chat Completions API格式（如阿里云DashScope）
            data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": 0.9,
                "stream": False
            }
            print(f"📋 使用Chat Completions API格式 (messages)")
        elif self.is_completions_api:
            # Completions API格式（如本地qwen服务器）
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": 0.9,
                "stream": False
            }
            print(f"📋 使用Completions API格式 (prompt)")
        else:
            # 默认使用Completions格式
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            print(f"📋 使用默认Completions API格式 (prompt)")
        
        try:
            # 建立连接并发送请求
            print(f"⏱️ 连接超时: {self.timeout_connect}秒, 读取超时: {self.timeout_read}秒")
            
            response = requests.post(
                self.server_url,
                headers=headers,
                json=data,
                timeout=(self.timeout_connect, self.timeout_read)
            )
            
            # 检查HTTP状态码
            if response.status_code != 200:
                error_msg = f"LLM服务器返回错误状态码: {response.status_code}"
                print(f"\n❌ {error_msg}")
                print(f"响应内容: {response.text}")
                raise Exception(error_msg)
            
            # 解析JSON响应
            response_data = response.json()
            if 'choices' not in response_data or not response_data['choices']:
                error_msg = "LLM响应格式异常：缺少choices字段"
                print(f"\n❌ {error_msg}")
                raise Exception(error_msg)
            
            # 提取文本内容 - 根据API类型适配不同格式
            choice = response_data['choices'][0]
            if self.is_chat_api and 'message' in choice and 'content' in choice['message']:
                # Chat Completions API格式
                response_text = choice['message']['content'].strip()
            elif 'text' in choice:
                # Completions API格式
                response_text = choice['text'].strip()
            else:
                error_msg = "LLM响应格式异常：无法提取内容"
                print(f"\n❌ {error_msg}")
                print(f"响应结构: {choice}")
                raise Exception(error_msg)
            
            if not response_text:
                error_msg = "LLM返回空响应"
                print(f"\n❌ {error_msg}")
                raise Exception(error_msg)
            
            print(f"✅ 成功收到LLM响应 (长度: {len(response_text)}字符)")
            
            # 只保存解析后的回复到last_response.txt
            self._save_response_to_file(response_text, is_mock=False)
            
            return response_text
                
        except requests.exceptions.ConnectTimeout:
            error_msg = f"连接超时: 无法在{self.timeout_connect}秒内连接到LLM服务器 {self.server_url}"
            print(f"\n❌ {error_msg}")
            raise Exception(error_msg)
        except requests.exceptions.ReadTimeout:
            error_msg = f"读取超时: LLM服务器在{self.timeout_read}秒内未返回完整响应"
            print(f"\n❌ {error_msg}")
            raise Exception(error_msg)
        except requests.exceptions.ConnectionError as e:
            error_msg = f"连接错误: 无法建立与LLM服务器的连接 - {str(e)}"
            print(f"\n❌ {error_msg}")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"LLM查询失败: {str(e)}"
            print(f"\n❌ {error_msg}")
            raise Exception(error_msg)

    def get_unload_strategy(self, env_state, device_info, edge_info, cloud_info, tasks_info=None):
        """获取LLM生成的卸载策略 - 仅支持格式1（三元分割格式）
        
        Returns:
            list: 包含格式1策略的列表
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
        print("\n🚀 开始LLM卸载策略咨询（格式1：三元分割）...")
        
        try:
            # 构建提示词
            prompt = self._build_prompt(env_state, device_info, edge_info, cloud_info, tasks_info)
            
            # 调用LLM
            print(f"📡 向LLM服务器发送请求: {self.server_url}")
            response = self.query(prompt)
            
            print(f"\n📋 开始解析LLM响应...")
            # 解析响应（仅格式1）
            strategies = self._extract_format1_from_text(response)
            
            if not strategies:
                print("⚠️ 无法解析LLM响应中的有效策略，使用默认策略")
                strategies = self._generate_default_format1_strategies(len(device_info))
            else:
                print(f"✅ 成功解析LLM策略: {len(strategies)}个设备的卸载决策")
                # 打印解析到的策略概要
                for strategy in strategies:
                    device_id = strategy.get('device_id', 0)
                    local_ratio = strategy.get('local_ratio', 1.0)
                    edge_ratio = strategy.get('edge_ratio', 0.0)
                    cloud_ratio = strategy.get('cloud_ratio', 0.0)
                    target_edge = strategy.get('target_edge', 0)
                    print(f"  设备{device_id}: 本地{local_ratio:.2f}, 边缘{edge_ratio:.2f}, 云端{cloud_ratio:.2f} → Edge{target_edge}")
            
            return strategies
            
        except Exception as e:
            print(f"\n❌ LLM服务调用失败: {e}")
            if self.use_mock:
                print("🔄 启用模拟模式，使用规则生成策略继续训练...")
                # 生成模拟策略（格式1）
                return self._generate_mock_format1_strategies(env_state, device_info, edge_info, cloud_info)
            else:
                print("🔄 返回默认策略...")
                return self._generate_default_format1_strategies(len(device_info))

    def _generate_mock_format1_strategies(self, env_state, device_info, edge_info, cloud_info):
        """在LLM服务不可用时生成智能模拟策略（格式1：三元分割）
        
        基于任务特征和资源状态生成合理的三元分割卸载策略
        """
        print("🤖 生成智能模拟卸载策略（格式1：三元分割）...")
        strategies = []
        
        # 简单解析环境状态
        try:
            env_state_array = np.array(env_state)
            device_states = env_state_array[:len(device_info) * 4].reshape(len(device_info), 4)
            edge_states = env_state_array[len(device_info) * 4: len(device_info) * 4 + len(edge_info) * 3].reshape(len(edge_info), 3)
            cloud_states = env_state_array[len(device_info) * 4 + len(edge_info) * 3:len(device_info) * 4 + len(edge_info) * 3 + len(cloud_info) * 3].reshape(len(cloud_info), 3)
            task_states = env_state_array[len(device_info) * 4 + len(edge_info) * 3 + len(cloud_info) * 3:].reshape(len(device_info), 3)
        except:
            # 异常情况下使用默认策略
            print("⚠️ 环境状态解析失败，使用默认策略")
            return self._generate_default_format1_strategies(len(device_info))
        
        print(f"📊 分析{len(device_info)}个设备的任务特征...")
        
        # 根据任务特征和资源状态生成智能三元分割策略
        for i in range(len(device_info)):
            device_battery = device_states[i][2] if len(device_states[i]) > 2 else 1.0  # 电池状态
            task_computation = task_states[i][0]    # 计算量 (MI)
            task_data_size = task_states[i][1]      # 数据量 (MB)
            task_deadline = task_states[i][2]       # 截止时间 (秒)
            
            # 计算处理时间估算
            local_time = task_computation / 2.0     # 本地处理时间 (2GHz)
            edge_time = task_computation / 8.0      # 边缘处理时间 (8GHz)
            cloud_time = task_computation / 32.0    # 云端处理时间 (32GHz)
            
            # 简单的传输时间估算
            edge_transmission = task_data_size / 50.0   # 假设50MB/s到边缘
            cloud_transmission = task_data_size / 25.0  # 假设25MB/s到云端
            
            # 三元分割决策逻辑
            local_ratio = 0.0
            edge_ratio = 0.0
            cloud_ratio = 0.0
            target_edge = i % len(edge_info)  # 轮询分配边缘服务器
            
            # 根据截止时间紧急程度和电池状态决定分割策略
            if local_time > task_deadline:
                # 本地无法在截止时间内完成，必须卸载
                if device_battery < 0.2:
                    # 电池低，全部卸载
                    edge_total = edge_time + edge_transmission
                    cloud_total = cloud_time + cloud_transmission
                    
                    if edge_total <= cloud_total:
                        # 边缘更快，主要卸载到边缘
                        local_ratio = 0.0
                        edge_ratio = 0.8
                        cloud_ratio = 0.2
                    else:
                        # 云端更快，主要卸载到云端
                        local_ratio = 0.0
                        edge_ratio = 0.3
                        cloud_ratio = 0.7
                else:
                    # 电池充足，混合执行
                    local_ratio = 0.2
                    edge_ratio = 0.5
                    cloud_ratio = 0.3
                    
            elif device_battery < 0.3:
                # 电池低但时间充足，节能优先
                if task_computation > 500:
                    # 计算密集型，大部分卸载
                    local_ratio = 0.1
                    edge_ratio = 0.6
                    cloud_ratio = 0.3
                else:
                    # 轻量任务，适度卸载
                    local_ratio = 0.3
                    edge_ratio = 0.5
                    cloud_ratio = 0.2
                    
            else:
                # 电池充足，性能优先
                if task_computation > 800:
                    # 高计算量，利用云端并行
                    local_ratio = 0.3
                    edge_ratio = 0.4
                    cloud_ratio = 0.3
                elif task_computation > 400:
                    # 中等计算量，主要用边缘
                    local_ratio = 0.4
                    edge_ratio = 0.5
                    cloud_ratio = 0.1
                else:
                    # 轻量任务，主要本地执行
                    local_ratio = 0.7
                    edge_ratio = 0.2
                    cloud_ratio = 0.1
            
            # 确保比例和为1
            total = local_ratio + edge_ratio + cloud_ratio
            if total > 0:
                local_ratio /= total
                edge_ratio /= total
                cloud_ratio /= total
            else:
                local_ratio, edge_ratio, cloud_ratio = 1.0, 0.0, 0.0
            
            strategy = {
                "device_id": i,
                "local_ratio": round(local_ratio, 2),
                "edge_ratio": round(edge_ratio, 2),
                "cloud_ratio": round(cloud_ratio, 2),
                "target_edge": target_edge
            }
            strategies.append(strategy)
            
            # 打印决策理由
            print(f"  设备{i}: 计算量{task_computation:.0f}MI, 电池{device_battery:.1%}, 截止{task_deadline:.1f}s")
            print(f"    → 本地{local_ratio:.2f}, 边缘{edge_ratio:.2f}, 云端{cloud_ratio:.2f}, Edge{target_edge}")
        
        print(f"✅ 生成{len(strategies)}个智能模拟策略（格式1）")
        return strategies

    def _extract_format1_from_text(self, text):
        """从文本中提取格式1（三元分割）JSON内容"""
        print(f"开始解析LLM响应文本（格式1），长度: {len(text)}字符")
        
        # 预处理：移除可能的markdown标记和多余的空白
        text = text.strip()
        
        # 策略1: 尝试直接解析整个文本为JSON
        try:
            result = json.loads(text)
            if isinstance(result, list):
                print("✅ 策略1成功：直接解析为JSON数组")
                return self._validate_format1_strategies(result)
            elif isinstance(result, dict):
                if 'strategies' in result:
                    print("✅ 策略1成功：解析带strategies字段的JSON")
                    return self._validate_format1_strategies(result['strategies'])
                else:
                    print("✅ 策略1成功：解析为单个JSON对象，转换为数组")
                    return self._validate_format1_strategies([result])
        except:
            pass
        
        # 策略2: 查找markdown代码块中的JSON
        json_code_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.findall(json_code_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in json_matches:
            try:
                result = json.loads(match.strip())
                if isinstance(result, dict):
                    if 'strategies' in result:
                        print("✅ 策略2成功：从markdown代码块中提取strategies")
                        strategies = result['strategies']
                        # 字段名转换
                        strategies = self._convert_field_names(strategies)
                        return self._validate_format1_strategies(strategies)
                    else:
                        print("✅ 策略2成功：从markdown代码块中提取JSON对象")
                        converted = self._convert_field_names([result])
                        return self._validate_format1_strategies(converted)
                elif isinstance(result, list):
                    print("✅ 策略2成功：从markdown代码块中提取JSON数组")
                    converted = self._convert_field_names(result)
                    return self._validate_format1_strategies(converted)
            except:
                continue
        
        # 策略3: 查找并提取 "strategies": [...] 格式
        strategies_pattern = r'"strategies":\s*\[[\s\S]*?\]'
        strategies_match = re.search(strategies_pattern, text, re.IGNORECASE)
        
        if strategies_match:
            try:
                strategies_text = "{" + strategies_match.group(0) + "}"
                result = json.loads(strategies_text)
                print("✅ 策略3成功：提取strategies字段")
                strategies = self._convert_field_names(result['strategies'])
                return self._validate_format1_strategies(strategies)
            except:
                pass
        
        # 策略4: 查找并提取 [...] 格式的JSON数组
        array_pattern = r'\[[\s\S]*?\]'
        array_matches = re.findall(array_pattern, text)
        
        for match in array_matches:
            try:
                result = json.loads(match)
                if isinstance(result, list):
                    converted = self._convert_field_names(result)
                    validated = self._validate_format1_strategies(converted)
                    if validated:
                        print("✅ 策略4成功：提取JSON数组格式")
                        return validated
            except:
                continue

        # 策略5: 查找并提取 {...} 格式的JSON对象
        object_pattern = r'\{[\s\S]*?\}'
        object_matches = re.findall(object_pattern, text)
        
        valid_objects = []
        for match in object_matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict) and ('device_id' in obj or 'local_ratio' in obj):
                    valid_objects.append(obj)
            except:
                continue
        
        if valid_objects:
            converted = self._convert_field_names(valid_objects)
            validated = self._validate_format1_strategies(converted)
            if validated:
                print(f"✅ 策略5成功：提取到{len(validated)}个JSON对象")
                return validated
        
        print("❌ 所有解析策略均失败")
        print(f"📝 响应文本预览: {text[:200]}...")
        return []

    def _convert_field_names(self, strategies):
        """转换字段名，适配不同LLM的输出格式"""
        if not isinstance(strategies, list):
            return strategies
        
        converted = []
        for strategy in strategies:
            if not isinstance(strategy, dict):
                converted.append(strategy)
                continue
                
            # 创建新的策略对象
            new_strategy = {}
            
            # 复制已知字段
            for key in ['device_id', 'local_ratio', 'edge_ratio', 'cloud_ratio']:
                if key in strategy:
                    new_strategy[key] = strategy[key]
            
            # 转换target_edge_server到target_edge
            if 'target_edge' in strategy:
                new_strategy['target_edge'] = strategy['target_edge']
            elif 'target_edge_server' in strategy:
                new_strategy['target_edge'] = strategy['target_edge_server']
            
            # 忽略其他额外字段（如rationale, expected_latency等）
            converted.append(new_strategy)
        
        return converted
    
    def _validate_format1_strategies(self, strategies):
        """验证并修正格式1策略"""
        if not isinstance(strategies, list):
            return []
        
        validated_strategies = []
        
        for i, strategy in enumerate(strategies):
            if not isinstance(strategy, dict):
                continue
                
            # 提取并验证字段
            device_id = strategy.get('device_id', i)
            local_ratio = float(strategy.get('local_ratio', 1.0))
            edge_ratio = float(strategy.get('edge_ratio', 0.0))
            cloud_ratio = float(strategy.get('cloud_ratio', 0.0))
            target_edge = int(strategy.get('target_edge', 0))
            
            # 归一化比例
            total = local_ratio + edge_ratio + cloud_ratio
            if total > 0:
                local_ratio /= total
                edge_ratio /= total
                cloud_ratio /= total
            else:
                local_ratio, edge_ratio, cloud_ratio = 1.0, 0.0, 0.0
            
            validated_strategies.append({
                "device_id": device_id,
                "local_ratio": round(local_ratio, 3),
                "edge_ratio": round(edge_ratio, 3),
                "cloud_ratio": round(cloud_ratio, 3),
                "target_edge": max(0, target_edge)
            })
        
        return validated_strategies
    
    def _generate_default_format1_strategies(self, num_devices):
        """生成默认的格式1策略（全本地执行）"""
        return [
            {
                "device_id": i,
                "local_ratio": 1.0,
                "edge_ratio": 0.0,
                "cloud_ratio": 0.0,
                "target_edge": 0
            }
            for i in range(num_devices)
        ]

    def _build_prompt(self, env_state, device_info, edge_info, cloud_info, tasks_info=None):
        """构建LLM提示词 - 适配简化设备模型"""
        try:
            # 导入新的提示词构建器
            from llm_assistant.prompt_builder import PromptBuilder
            
            # 如果没有提供任务信息，从device_info推断
            if tasks_info is None:
                tasks_info = []
                for i in range(len(device_info)):
                    tasks_info.append({
                        'task_id': f'task_{i}',
                        'device_id': i,
                        'task_type': 'medium',
                        'data_size': 25.0,
                        'cpu_cycles': 5e9,
                        'deadline': 30.0,
                        'remaining_time': 25.0
                    })
            
            # 使用新的提示词构建器
            prompt = PromptBuilder.build_offloading_strategy_prompt(
                env_state, device_info, edge_info, cloud_info, tasks_info
            )
            print("✅ 使用简化设备模型的提示词构建器")
            return prompt
            
        except ImportError as e:
            print(f"⚠️ 提示词构建器导入失败: {e}")
            # 如果导入失败，使用简化的备用提示模板
            return self._build_fallback_prompt(device_info, edge_info, cloud_info)
    
    def _build_fallback_prompt(self, device_info, edge_info, cloud_info):
        """备用简化提示模板 - 适配简化设备模型"""
        print("🔄 使用备用简化提示模板")
        
        prompt = f"""你是云边端计算卸载专家。系统采用简化设备模型：

**设备状态（简化版）**:
- UE设备({len(device_info)}个): CPU频率 + 电池 + 任务负载
- ES服务器({len(edge_info)}个): CPU频率 + 任务负载
- CS服务器({len(cloud_info)}个): CPU频率（资源无限）

**当前状态**:
"""
        
        # UE设备状态
        for i, device in enumerate(device_info):
            battery_pct = device.get('battery_percentage', 0.5) * 100
            cpu_freq = device.get('cpu_frequency', 0.8)
            task_load = device.get('task_load', 0.0)
            prompt += f"UE{i}: {cpu_freq:.1f}GHz, 电池{battery_pct:.0f}%, 负载{task_load:.1f}s\n"
        
        # ES服务器状态  
        for i, server in enumerate(edge_info):
            cpu_freq = server.get('cpu_frequency', 8)
            task_load = server.get('task_load', 0.0)
            prompt += f"ES{i}: {cpu_freq}GHz, 负载{task_load:.1f}s\n"
            
        prompt += f"""
**通信延迟差异**:
- 边缘卸载: 低延迟(1Gbps直连)
- 云端卸载: 高延迟(需要中转)

**任务分割策略**:
每个UE设备需要决策: [α1, α2, α3, edge_id]
- α1: 本地执行比例
- α2: 边缘执行比例  
- α3: 云端执行比例
- edge_id: 目标边缘服务器(0-{len(edge_info)-1})

要求直接返回JSON格式的策略数组:

[
  {{"device_id": 0, "local_ratio": 0.3, "edge_ratio": 0.5, "cloud_ratio": 0.2, "target_edge": 1}},
  {{"device_id": 1, "local_ratio": 0.0, "edge_ratio": 0.8, "cloud_ratio": 0.2, "target_edge": 0}}
]

JSON:"""

        return prompt
    
    def _save_response_to_file(self, response_text, is_mock=False):
        """保存LLM响应文本到文件"""
        try:
            mode_label = "模拟模式" if is_mock else "LLM服务器"
            
            # 只保存最新的响应
            with open("last_response.txt", "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"LLM返回的原始内容 ({mode_label})\n")
                f.write("=" * 60 + "\n")
                f.write(response_text)
                f.write("\n" + "=" * 60 + "\n")
            
            print(f"✅ LLM响应已保存到: last_response.txt")
        except Exception as e:
            print(f"⚠️ 保存LLM响应失败: {e}")