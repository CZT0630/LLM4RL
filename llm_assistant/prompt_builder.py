# llm_assistant/prompt_builder.py
class PromptBuilder:
    @staticmethod
    def build_unload_strategy_prompt(env_state, device_info, edge_info, cloud_info):
        """构建获取卸载策略的提示"""
        # 解析环境状态并构建自然语言描述
        # 实现细节在llm_client.py中
        pass