from typing import List, Dict
from openai import OpenAI
from config.config import LLMConfig


class LLMClient:
    """LLM客户端封装类"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """发送聊天完成请求"""
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM请求失败: {e}")