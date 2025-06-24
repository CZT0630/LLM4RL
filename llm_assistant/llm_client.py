# llm_assistant/llm_client.py
import requests
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient:
    def __init__(self, model_name="qwne3-14b", server_url="http://10.200.1.35"):
        # self.api_key = api_key
        self.model_name = model_name
        self.server_url = server_url
        self.retry_attempts = 3
        self.wait_min = 1
        self.wait_max = 4


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(self, prompt):
        """向LLM发送查询并获取响应"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            data = {
                "model": self.model_name,
                "prompt": prompt
            }
            response = requests.post(f"{self.server_url}/generate", headers=headers, json=data)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"LLM查询失败: {e}")
            raise

    def get_unload_strategy(self, env_state, device_info, edge_info, cloud_info):
        """获取卸载策略建议"""
        # 构建提示
        prompt = self._build_prompt(env_state, device_info, edge_info, cloud_info)

        # 查询LLM
        response = self.query(prompt)

        # 解析响应
        try:
            strategies = json.loads(response)
            if not isinstance(strategies, list):
                strategies = [strategies]
            return strategies
        except json.JSONDecodeError:
            print(f"解析LLM响应失败: {response}")
            # 尝试多次查询以提高可靠性
            strategies = []
            for _ in range(3):
                time.sleep(1)  # 避免过快请求
                response = self.query(prompt)
                try:
                    strategy = json.loads(response)
                    if isinstance(strategy, list):
                        strategies.extend(strategy)
                    else:
                        strategies.append(strategy)
                except json.JSONDecodeError:
                    continue

            return strategies if strategies else []

    def _build_prompt(self, env_state, device_info, edge_info, cloud_info):
        """构建提示模板"""
        # 解析环境状态
        device_states = env_state[:len(device_info) * 4].reshape(len(device_info), 4)
        edge_states = env_state[len(device_info) * 4: len(device_info) * 4 + len(edge_info) * 3].reshape(len(edge_info),
                                                                                                         3)
        cloud_states = env_state[
                       len(device_info) * 4 + len(edge_info) * 3: len(device_info) * 4 + len(edge_info) * 3 + len(
                           cloud_info) * 3].reshape(len(cloud_info), 3)
        task_states = env_state[len(device_info) * 4 + len(edge_info) * 3 + len(cloud_info) * 3:].reshape(
            len(device_info), 3)

        # 构建提示
        prompt = "当前边缘环境状态：\n"

        # 添加设备信息
        prompt += "- 终端设备状态：\n"
        for i, (device, state) in enumerate(zip(device_info, device_states)):
            prompt += f"  • 设备{i}: CPU容量={device['cpu']}GHz, 负载={state[1]:.2f}, "
            prompt += f"电池={state[2]:.2f}, 内存使用率={state[3]:.2f}\n"

        # 添加边缘服务器信息
        prompt += "- 边缘服务器状态：\n"
        for i, (edge, state) in enumerate(zip(edge_info, edge_states)):
            prompt += f"  • 边缘服务器{i}: CPU容量={edge['cpu']}GHz, 负载={state[1]:.2f}, "
            prompt += f"内存使用率={state[2]:.2f}\n"

        # 添加云端服务器信息
        prompt += "- 云端服务器状态：\n"
        for i, (cloud, state) in enumerate(zip(cloud_info, cloud_states)):
            prompt += f"  • 云端服务器{i}: CPU容量={cloud['cpu']}GHz, 负载={state[1]:.2f}, "
            prompt += f"内存使用率={state[2]:.2f}\n"

        # 添加任务信息
        prompt += "- 当前任务状态：\n"
        for i, state in enumerate(task_states):
            prompt += f"  • 任务{i}: 计算量={state[0]:.2f}MI, 数据量={state[1]:.2f}MB, "
            prompt += f"截止时间={state[2]:.2f}秒\n"

        # 请求卸载策略
        prompt += """
请为每个任务生成卸载策略建议（需包含卸载目标节点选择、卸载比例建议），用JSON数组格式输出：
[
  {
    "task_id": 0,
    "target_node": ["Edge0", "Cloud"],
    "unload_ratio": 0.7
  },
  ...
]
每个对象对应一个任务的卸载策略。
"""

        return prompt