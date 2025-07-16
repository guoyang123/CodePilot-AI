import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM配置类，统一管理模型相关配置"""
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """从环境变量创建配置实例"""
        return cls(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus"  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        )