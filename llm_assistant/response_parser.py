# llm_assistant/response_parser.py
class ResponseParser:
    @staticmethod
    def parse_unload_strategy(response, num_tasks, num_edges, num_clouds):
        """解析LLM返回的卸载策略"""
        try:
            strategies = json.loads(response)

            # 标准化策略格式
            parsed_strategies = []
            for i in range(num_tasks):
                task_strategy = next((s for s in strategies if s.get('task_id') == i), None)
                if not task_strategy:
                    # 默认策略：不卸载
                    task_strategy = {
                        "task_id": i,
                        "target_node": [],
                        "unload_ratio": 0.0
                    }

                # 解析目标节点
                target_idx = 0  # 默认本地
                if task_strategy["target_node"]:
                    target = task_strategy["target_node"][0]
                    if target.startswith("Edge"):
                        edge_id = int(target.replace("Edge", ""))
                        if 0 <= edge_id < num_edges:
                            target_idx = edge_id + 1  # 1-边缘节点
                    elif target == "Cloud":
                        target_idx = num_edges + 1  # 云端

                parsed_strategies.append({
                    "task_id": i,
                    "unload_ratio": min(max(task_strategy["unload_ratio"], 0.0), 1.0),
                    "target_idx": target_idx
                })

            return parsed_strategies

        except Exception as e:
            print(f"解析卸载策略失败: {e}")
            # 返回默认策略
            return [{
                "task_id": i,
                "unload_ratio": 0.0,
                "target_idx": 0
            } for i in range(num_tasks)]