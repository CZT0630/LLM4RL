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
        self.api_key = api_key
        self.model_name = model_name
        self.server_url = server_url
        self.timeout_connect = timeout_connect
        self.timeout_read = timeout_read
        self.use_mock = use_mock
        
        # 从配置中读取LLM参数
        if config and 'llm' in config:
            llm_config = config['llm']
            self.max_tokens = llm_config.get('max_tokens', 4096)
            self.temperature = llm_config.get('temperature', 0.3)
        else:
            # 默认值
            self.max_tokens = 4096
            self.temperature = 0.3
        
        print(f"✅ LLM客户端初始化:")
        print(f"  服务器: {self.server_url}")
        print(f"  模型: {self.model_name}")
        print(f"  最大tokens: {self.max_tokens}")
        print(f"  温度: {self.temperature}")
        print(f"  超时设置: 连接{self.timeout_connect}s, 读取{self.timeout_read}s")
        print(f"  模拟模式: {'启用' if self.use_mock else '禁用'}")

    def query(self, prompt):
        """
        查询LLM服务器
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: LLM返回的文本响应
        """
        try:
            # 构建请求头
            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            # 构建请求数据
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": self.max_tokens,    # 使用配置中的值
                "temperature": self.temperature,  # 使用配置中的值
                "stream": False      # 不使用流式返回
            }
            
            print(f"\n=== 发送LLM请求 ===")
            print(f"目标服务器: {self.server_url}")
            print(f"模型名称: {self.model_name}")
            print(f"提示长度: {len(prompt)}字符")
            print(f"最大tokens: {self.max_tokens}")
            print(f"温度: {self.temperature}")
            print(f"连接超时: {self.timeout_connect}秒")
            print(f"读取超时: {self.timeout_read}秒")
            print("开始发送请求，请耐心等待LLM响应...")
            
            # 记录开始时间
            start_time = time.time()
            
            # 发送请求并等待响应
            response = requests.post(
                self.server_url, 
                headers=headers, 
                json=data, 
                timeout=(self.timeout_connect, self.timeout_read)  # (连接超时, 读取超时)
            )
            
            # 计算响应时间
            response_time = time.time() - start_time
            print(f"请求完成，耗时: {response_time:.2f}秒")
            print(f"响应状态码: {response.status_code}")
            
            # 检查响应状态
            if response.status_code != 200:
                error_text = response.text[:500] if response.text else "无错误信息"
                print(f"LLM服务返回错误状态码: {response.status_code}")
                print(f"错误响应内容: {error_text}")
                raise Exception(f"LLM服务返回错误: HTTP {response.status_code}")
            
            # 解析响应JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                print(f"响应JSON解析失败: {e}")
                print(f"原始响应内容: {response.text[:1000]}")
                raise Exception("LLM响应格式不是有效的JSON")
            
            print(f"\n=== LLM响应解析 ===")
            print(f"响应数据字段: {list(response_data.keys())}")
            
            # 提取响应文本
            response_text = None
            
            # 适应不同API结构
            if "choices" in response_data and len(response_data["choices"]) > 0:
                # OpenAI风格API
                response_text = response_data["choices"][0].get("text", "").strip()
                print("使用OpenAI风格API响应格式")
            elif "response" in response_data:
                # 自定义API风格1
                response_text = response_data["response"].strip()
                print("使用自定义API响应格式1")
            elif "output" in response_data:
                # 自定义API风格2
                response_text = response_data["output"].strip()
                print("使用自定义API响应格式2")
            elif "completion" in response_data:
                # 自定义API风格3
                response_text = response_data["completion"].strip()
                print("使用自定义API响应格式3")
            else:
                # 尝试返回整个JSON字符串
                response_text = json.dumps(response_data)
                print("使用完整JSON作为响应")
            
            if not response_text:
                raise Exception("LLM响应为空或无法提取有效内容")
            
            print(f"\n=== LLM原始响应内容 ===")
            print("=" * 60)
            print(response_text)
            print("=" * 60)
            print(f"响应长度: {len(response_text)}字符")
            
            # 保存原始响应JSON和解析后的文本
            self._save_raw_response_to_file(response_data)
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
        """获取卸载策略建议 - 适配简化设备模型"""
        # 构建提示
        prompt = self._build_prompt(env_state, device_info, edge_info, cloud_info, tasks_info)
        
        # 保存提示到文件以便调试
        with open("last_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"已保存提示到 last_prompt.txt (长度: {len(prompt)}字符)")
        
        print("\n📤 发送卸载策略请求到LLM...")
        # 查询LLM
        try:
            response = self.query(prompt)
            
            print(f"\n✅ LLM响应获取成功!")
            
            # 保存响应到文件以便调试
            with open("last_response.txt", "w", encoding="utf-8") as f:
                f.write(response)
            print(f"已保存LLM响应到 last_response.txt")
            
            print(f"\n📋 开始解析LLM响应...")
            # 解析响应
            strategies = self._extract_json_from_text(response)
            
            if not strategies:
                print("⚠️ 无法解析LLM响应中的有效策略，使用默认策略")
                strategies = self._generate_default_strategies(len(device_info))
            else:
                print(f"✅ 成功解析LLM策略: {len(strategies)}个任务的卸载决策")
                # 打印解析到的策略概要
                for i, strategy in enumerate(strategies):
                    task_id = strategy.get('task_id', i)
                    offload_ratio = strategy.get('offload_ratio', 0.0)
                    target_node = strategy.get('target_node', 0)
                    print(f"  任务{task_id}: 卸载比例={offload_ratio:.2f}, 目标节点={target_node}")
            
            return strategies
            
        except Exception as e:
            print(f"\n❌ LLM服务调用失败: {e}")
            if self.use_mock:
                print("🔄 启用模拟模式，使用规则生成策略继续训练...")
                # 生成模拟策略
                return self._generate_mock_strategies(env_state, device_info, edge_info, cloud_info)
            else:
                print("🔄 返回默认策略...")
                return self._generate_default_strategies(len(device_info))

    def _generate_mock_strategies(self, env_state, device_info, edge_info, cloud_info):
        """在LLM服务不可用时生成智能模拟策略
        
        基于任务特征和资源状态生成合理的卸载策略
        """
        print("🤖 生成智能模拟卸载策略...")
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
            return self._generate_default_strategies(len(device_info))
        
        print(f"📊 分析{len(device_info)}个任务的特征...")
        
        # 根据任务特征和资源状态生成智能策略
        for i in range(len(device_info)):
            device_battery = device_states[i][2] if len(device_states[i]) > 2 else 1.0  # 电池状态
            task_computation = task_states[i][0]    # 计算量 (MI)
            task_data_size = task_states[i][1]      # 数据量 (MB)
            task_deadline = task_states[i][2]       # 截止时间 (秒)
            
            # 计算处理时间估算
            local_time = task_computation / 2.0     # 本地处理时间 (2GHz)
            edge_time = task_computation / 8.0      # 边缘处理时间 (8GHz)
            cloud_time = task_computation / 32.0    # 云端处理时间 (32GHz)
            
            # 简单的传输时间估算 (假设网络带宽)
            edge_transmission = task_data_size / 50.0   # 假设50MB/s到边缘
            cloud_transmission = task_data_size / 25.0  # 假设25MB/s到云端
            
            # 决策逻辑
            offload_ratio = 0.0
            target_node = 0  # 默认本地
            
            # 如果本地处理时间超过截止时间，必须卸载
            if local_time > task_deadline:
                # 比较边缘和云端的总时间
                edge_total_time = edge_time + edge_transmission
                cloud_total_time = cloud_time + cloud_transmission
                
                if edge_total_time <= task_deadline and edge_total_time <= cloud_total_time:
                    # 卸载到边缘服务器
                    offload_ratio = 1.0
                    target_node = (i % len(edge_info)) + 1  # 轮询分配边缘服务器
                    
                elif cloud_total_time <= task_deadline:
                    # 卸载到云端
                    offload_ratio = 1.0
                    target_node = len(edge_info) + 1  # 云端节点
                    
                else:
                    # 即使超时也选择最快的选项
                    if cloud_total_time < edge_total_time:
                        offload_ratio = 1.0
                        target_node = len(edge_info) + 1  # 云端
                    else:
                        offload_ratio = 1.0
                        target_node = (i % len(edge_info)) + 1  # 边缘
                        
            else:
                # 本地可以完成，但考虑电池状态
                if device_battery < 0.2:  # 电池低于20%
                    # 部分卸载以节省电量
                    if task_computation > 500:  # 计算密集型任务
                        offload_ratio = 0.7
                        # 选择更高效的目标
                        if cloud_time + cloud_transmission < edge_time + edge_transmission:
                            target_node = len(edge_info) + 1  # 云端
                        else:
                            target_node = (i % len(edge_info)) + 1  # 边缘
                    else:
                        offload_ratio = 0.3
                        target_node = (i % len(edge_info)) + 1  # 边缘
                else:
                    # 电池充足，可以考虑本地处理或轻度卸载
                    if task_computation > 800:  # 非常计算密集
                        offload_ratio = 0.5
                        target_node = len(edge_info) + 1  # 云端
                    elif task_computation > 400:  # 中等计算量
                        offload_ratio = 0.3
                        target_node = (i % len(edge_info)) + 1  # 边缘
                    # else: 保持本地处理 (offload_ratio = 0.0, target_node = 0)
            
            # 确保值在合法范围内
            offload_ratio = max(0.0, min(1.0, offload_ratio))
            target_node = max(0, min(len(edge_info) + len(cloud_info), target_node))
            
            strategy = {
                "task_id": i,
                "offload_ratio": round(offload_ratio, 2),
                "target_node": target_node
            }
            strategies.append(strategy)
            
            # 打印决策理由
            target_name = "本地"
            if target_node >= 1 and target_node <= len(edge_info):
                target_name = f"边缘服务器{target_node-1}"
            elif target_node > len(edge_info):
                target_name = "云端"
                
            print(f"  任务{i}: 计算量{task_computation:.0f}MI, 截止{task_deadline:.1f}s -> "
                  f"卸载{offload_ratio:.2f}到{target_name}")
        
        print(f"✅ 生成{len(strategies)}个智能模拟策略")
        return strategies

    def _extract_json_from_text(self, text):
        """从文本中提取JSON内容，使用多种策略确保解析成功"""
        print(f"开始解析LLM响应文本，长度: {len(text)}字符")
        
        # 预处理：移除可能的markdown标记和多余的空白
        text = text.strip()
        
        # 策略1: 尝试直接解析整个文本为JSON
        try:
            result = json.loads(text)
            if isinstance(result, list):
                print("✅ 策略1成功：直接解析为JSON数组")
                return result
            elif isinstance(result, dict):
                print("✅ 策略1成功：解析为单个JSON对象，转换为数组")
                return [result]
        except:
            pass
        
        # 策略2: 查找并提取 [...]  格式的JSON数组
        array_pattern = r'\[[\s\S]*?\]'
        array_matches = re.findall(array_pattern, text)
        
        for match in array_matches:
            try:
                result = json.loads(match)
                if isinstance(result, list):
                    print("✅ 策略2成功：提取JSON数组格式")
                    return result
            except:
                continue
        
        # 策略3: 查找并提取 {...} 格式的JSON对象
        object_pattern = r'\{[\s\S]*?\}'
        object_matches = re.findall(object_pattern, text)
        
        valid_objects = []
        for match in object_matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict) and 'task_id' in obj:
                    valid_objects.append(obj)
            except:
                continue
        
        if valid_objects:
            print(f"✅ 策略3成功：提取到{len(valid_objects)}个JSON对象")
            return valid_objects
        
        # 策略4: 使用正则表达式直接提取关键信息
        pattern = r'task_id["\s]*:[\s]*(\d+)[\s\S]*?offload_ratio["\s]*:[\s]*([\d.]+)[\s\S]*?target_node["\s]*:[\s]*(\d+)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        if matches:
            result = []
            for match in matches:
                try:
                    task_id = int(match[0])
                    offload_ratio = float(match[1])
                    target_node = int(match[2])
                    result.append({
                        "task_id": task_id,
                        "offload_ratio": offload_ratio,
                        "target_node": target_node
                    })
                except:
                    continue
            
            if result:
                print(f"✅ 策略4成功：通过正则表达式提取{len(result)}个策略")
                return result
        
        # 策略5: 查找数字序列，尝试构建策略
        lines = text.split('\n')
        result = []
        
        for line in lines:
            # 查找包含任务信息的行
            task_match = re.search(r'任务(\d+).*?(\d+\.?\d*).*?(\d+)', line)
            if task_match:
                try:
                    task_id = int(task_match.group(1))
                    offload_ratio = float(task_match.group(2)) if '.' in task_match.group(2) else float(task_match.group(2)) / 10
                    target_node = int(task_match.group(3))
                    
                    # 确保值在合理范围内
                    if 0 <= offload_ratio <= 1 and 0 <= target_node <= 3:
                        result.append({
                            "task_id": task_id,
                            "offload_ratio": offload_ratio,
                            "target_node": target_node
                        })
                except:
                    continue
        
        if result:
            print(f"✅ 策略5成功：从文本行中提取{len(result)}个策略")
            return result
        
        print("❌ 所有解析策略均失败")
        return []
    
    def _generate_default_strategies(self, num_tasks):
        """生成默认卸载策略"""
        return [
            {
                "task_id": i,
                "offload_ratio": 0.0,
                "target_node": 0
            }
            for i in range(num_tasks)
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
    
    def _save_prompt_to_file(self, prompt):
        """保存提示词到文件"""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存最新的提示词
            with open("last_prompt.txt", "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("发送给LLM的提示词内容\n")
                f.write("=" * 60 + "\n")
                f.write(prompt)
                f.write("\n" + "=" * 60 + "\n")
            
            # 保存带时间戳的提示词
            with open(f"prompt_{timestamp}.txt", "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"发送给LLM的提示词内容 - {timestamp}\n")
                f.write("=" * 60 + "\n")
                f.write(prompt)
                f.write("\n" + "=" * 60 + "\n")
            
            print(f"✅ 提示词已保存到: last_prompt.txt 和 prompt_{timestamp}.txt")
        except Exception as e:
            print(f"⚠️ 保存提示词失败: {e}")
    
    def _save_response_to_file(self, response_text, is_mock=False):
        """保存LLM响应文本到文件"""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_label = "模拟模式" if is_mock else "LLM服务器"
            
            # 保存最新的响应
            with open("last_response.txt", "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"LLM返回的原始内容 ({mode_label})\n")
                f.write("=" * 60 + "\n")
                f.write(response_text)
                f.write("\n" + "=" * 60 + "\n")
            
            # 保存带时间戳的响应
            with open(f"response_{timestamp}.txt", "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"LLM返回的原始内容 ({mode_label}) - {timestamp}\n")
                f.write("=" * 60 + "\n")
                f.write(response_text)
                f.write("\n" + "=" * 60 + "\n")
            
            print(f"✅ LLM响应已保存到: last_response.txt 和 response_{timestamp}.txt")
        except Exception as e:
            print(f"⚠️ 保存LLM响应失败: {e}")
    
    def _save_raw_response_to_file(self, raw_json):
        """保存LLM原始JSON响应到文件"""
        try:
            import datetime
            import json
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存最新的原始响应
            with open("last_raw_response.json", "w", encoding="utf-8") as f:
                json.dump(raw_json, f, ensure_ascii=False, indent=2)
            
            # 保存带时间戳的原始响应
            with open(f"raw_response_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(raw_json, f, ensure_ascii=False, indent=2)
            
            print(f"✅ LLM原始响应已保存到: last_raw_response.json 和 raw_response_{timestamp}.json")
        except Exception as e:
            print(f"⚠️ 保存LLM原始响应失败: {e}")