import os
import threading
import hashlib
from typing import Optional, Dict, Any, Tuple
from langchain_community.llms import Tongyi
from langchain_community.chat_models import ChatTongyi


class TongyiLLM:
    """
    通义千问模型封装类
    用于统一管理通义千问模型的初始化和配置
    """
    
    # 支持的模型列表
    SUPPORTED_MODELS = [
        "qwen-turbo",
        "qwen-plus", 
        "qwen-max",
        "qwen-max-1201",
        "qwen-max-longcontext"
    ]
    
    def __init__(self, 
                 model_name: str = "qwen-plus",
                 temperature: float = 0.7,
                 api_key: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None):
        """
        初始化通义千问模型配置
        
        Args:
            model_name: 模型名称，默认为qwen-plus
            temperature: 温度参数，控制输出随机性，默认0.7
            api_key: API密钥，如果不提供则从环境变量获取
            max_tokens: 最大输出token数
            top_p: 核采样参数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # 验证模型名称
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {self.SUPPORTED_MODELS}")
        
        # 获取API密钥
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未找到DASHSCOPE_API_KEY，请设置环境变量或传入api_key参数")
    
    def get_chat_model(self) -> ChatTongyi:
        """
        获取ChatTongyi聊天模型实例
        
        Returns:
            ChatTongyi: 配置好的聊天模型实例
        """
        # 构建模型参数
        model_kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "dashscope_api_key": self.api_key
        }
        
        # 添加可选参数
        if self.max_tokens is not None:
            model_kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            model_kwargs["top_p"] = self.top_p
        
        return ChatTongyi(**model_kwargs)
    
    def get_llm_model(self) -> Tongyi:
        """
        获取Tongyi LLM模型实例
        
        Returns:
            Tongyi: 配置好的LLM模型实例
        """
        # 构建模型参数
        model_kwargs = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "dashscope_api_key": self.api_key
        }
        
        # 添加可选参数
        if self.max_tokens is not None:
            model_kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            model_kwargs["top_p"] = self.top_p
        
        return Tongyi(**model_kwargs)
    
    @classmethod
    def create_default_chat_model(cls, api_key: Optional[str] = None) -> ChatTongyi:
        """
        创建默认配置的聊天模型
        
        Args:
            api_key: API密钥，可选
            
        Returns:
            ChatTongyi: 默认配置的聊天模型实例
        """
        instance = cls(api_key=api_key)
        return instance.get_chat_model()
    
    @classmethod
    def create_custom_chat_model(cls, 
                                model_name: str = "qwen-plus",
                                temperature: float = 0.7,
                                api_key: Optional[str] = None,
                                **kwargs) -> ChatTongyi:
        """
        创建自定义配置的聊天模型
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
            api_key: API密钥
            **kwargs: 其他参数
            
        Returns:
            ChatTongyi: 自定义配置的聊天模型实例
        """
        instance = cls(model_name=model_name, temperature=temperature, api_key=api_key, **kwargs)
        return instance.get_chat_model()
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"TongyiLLM(model={self.model_name}, temperature={self.temperature})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"TongyiLLM(model_name='{self.model_name}', temperature={self.temperature}, "
                f"max_tokens={self.max_tokens}, top_p={self.top_p})")
    
    def get_config_hash(self) -> str:
        """获取配置的哈希值，用于缓存键"""
        config_str = f"{self.model_name}_{self.temperature}_{self.max_tokens}_{self.top_p}_{self.api_key}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]


class TongyiModelManager:
    """
    通义千问模型管理器
    实现模型实例的缓存和复用，避免重复创建模型对象
    """
    
    def __init__(self, max_cache_size: int = 10):
        """
        初始化模型管理器
        
        Args:
            max_cache_size: 最大缓存数量，默认10个
        """
        self._cache: Dict[str, ChatTongyi] = {}  # 模型缓存字典
        self._llm_cache: Dict[str, Tongyi] = {}  # LLM模型缓存字典
        self._access_order: Dict[str, int] = {}  # 访问顺序，用于LRU
        self._access_counter = 0  # 访问计数器
        self._max_cache_size = max_cache_size  # 最大缓存大小
        self._lock = threading.Lock()  # 线程锁，确保线程安全
    
    def _generate_cache_key(self, 
                           model_name: str = "qwen-plus",
                           temperature: float = 0.7,
                           api_key: Optional[str] = None,
                           max_tokens: Optional[int] = None,
                           top_p: Optional[float] = None) -> str:
        """
        生成缓存键
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
            api_key: API密钥
            max_tokens: 最大token数
            top_p: 核采样参数
            
        Returns:
            str: 缓存键
        """
        # 使用API密钥的哈希值而不是原始值，保护隐私
        api_key_hash = hashlib.md5((api_key or "").encode()).hexdigest()[:8] if api_key else "none"
        config_str = f"{model_name}_{temperature}_{max_tokens}_{top_p}_{api_key_hash}"
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _evict_lru_if_needed(self) -> None:
        """
        如果缓存超过最大大小，移除最少使用的项
        """
        if len(self._cache) >= self._max_cache_size:
            # 找到最少使用的项
            lru_key = min(self._access_order.keys(), key=lambda k: self._access_order[k])
            # 移除缓存项
            self._cache.pop(lru_key, None)
            self._llm_cache.pop(lru_key, None)
            self._access_order.pop(lru_key, None)
    
    def get_chat_model(self, 
                      model_name: str = "qwen-plus",
                      temperature: float = 0.7,
                      api_key: Optional[str] = None,
                      max_tokens: Optional[int] = None,
                      top_p: Optional[float] = None) -> ChatTongyi:
        """
        获取ChatTongyi模型实例（带缓存）
        
        Args:
            model_name: 模型名称，默认qwen-plus
            temperature: 温度参数，默认0.7
            api_key: API密钥，如果不提供则从环境变量获取
            max_tokens: 最大输出token数
            top_p: 核采样参数
            
        Returns:
            ChatTongyi: 聊天模型实例
        """
        # 获取或使用默认API密钥
        effective_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        
        # 生成缓存键
        cache_key = self._generate_cache_key(model_name, temperature, effective_api_key, max_tokens, top_p)
        
        with self._lock:  # 线程安全
            # 检查缓存中是否存在
            if cache_key in self._cache:
                # 更新访问顺序
                self._access_counter += 1
                self._access_order[cache_key] = self._access_counter
                return self._cache[cache_key]
            
            # 缓存中不存在，创建新实例
            try:
                # 使用TongyiLLM创建模型
                tongyi_llm = TongyiLLM(
                    model_name=model_name,
                    temperature=temperature,
                    api_key=effective_api_key,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                model = tongyi_llm.get_chat_model()
                
                # 如果需要，移除最少使用的项
                self._evict_lru_if_needed()
                
                # 添加到缓存
                self._cache[cache_key] = model
                self._access_counter += 1
                self._access_order[cache_key] = self._access_counter
                
                return model
                
            except Exception as e:
                raise RuntimeError(f"创建模型失败: {e}")
    
    def get_llm_model(self, 
                     model_name: str = "qwen-plus",
                     temperature: float = 0.7,
                     api_key: Optional[str] = None,
                     max_tokens: Optional[int] = None,
                     top_p: Optional[float] = None) -> Tongyi:
        """
        获取Tongyi LLM模型实例（带缓存）
        
        Args:
            model_name: 模型名称，默认qwen-plus
            temperature: 温度参数，默认0.7
            api_key: API密钥，如果不提供则从环境变量获取
            max_tokens: 最大输出token数
            top_p: 核采样参数
            
        Returns:
            Tongyi: LLM模型实例
        """
        # 获取或使用默认API密钥
        effective_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        
        # 生成缓存键
        cache_key = self._generate_cache_key(model_name, temperature, effective_api_key, max_tokens, top_p)
        
        with self._lock:  # 线程安全
            # 检查缓存中是否存在
            if cache_key in self._llm_cache:
                # 更新访问顺序
                self._access_counter += 1
                self._access_order[cache_key] = self._access_counter
                return self._llm_cache[cache_key]
            
            # 缓存中不存在，创建新实例
            try:
                # 使用TongyiLLM创建模型
                tongyi_llm = TongyiLLM(
                    model_name=model_name,
                    temperature=temperature,
                    api_key=effective_api_key,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                model = tongyi_llm.get_llm_model()
                
                # 如果需要，移除最少使用的项
                self._evict_lru_if_needed()
                
                # 添加到缓存
                self._llm_cache[cache_key] = model
                self._access_counter += 1
                self._access_order[cache_key] = self._access_counter
                
                return model
                
            except Exception as e:
                raise RuntimeError(f"创建LLM模型失败: {e}")
    
    def clear_cache(self) -> None:
        """
        清空所有缓存
        """
        with self._lock:
            self._cache.clear()
            self._llm_cache.clear()
            self._access_order.clear()
            self._access_counter = 0
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            Dict: 包含缓存统计信息的字典
        """
        with self._lock:
            return {
                "chat_model_count": len(self._cache),
                "llm_model_count": len(self._llm_cache),
                "max_cache_size": self._max_cache_size,
                "total_accesses": self._access_counter,
                "cached_keys": list(self._cache.keys()) + list(self._llm_cache.keys())
            }
    
    def remove_model(self, 
                    model_name: str = "qwen-plus",
                    temperature: float = 0.7,
                    api_key: Optional[str] = None,
                    max_tokens: Optional[int] = None,
                    top_p: Optional[float] = None) -> bool:
        """
        从缓存中移除特定配置的模型
        
        Args:
            model_name: 模型名称
            temperature: 温度参数
            api_key: API密钥
            max_tokens: 最大token数
            top_p: 核采样参数
            
        Returns:
            bool: 是否成功移除
        """
        effective_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        cache_key = self._generate_cache_key(model_name, temperature, effective_api_key, max_tokens, top_p)
        
        with self._lock:
            removed = False
            if cache_key in self._cache:
                del self._cache[cache_key]
                removed = True
            if cache_key in self._llm_cache:
                del self._llm_cache[cache_key]
                removed = True
            if cache_key in self._access_order:
                del self._access_order[cache_key]
            return removed


# 全局模型管理器实例
_global_model_manager = None
_manager_lock = threading.Lock()


def get_global_model_manager() -> TongyiModelManager:
    """
    获取全局模型管理器实例（单例模式）
    
    Returns:
        TongyiModelManager: 全局模型管理器实例
    """
    global _global_model_manager
    
    if _global_model_manager is None:
        with _manager_lock:
            if _global_model_manager is None:  # 双重检查锁定
                _global_model_manager = TongyiModelManager()
    
    return _global_model_manager


# 便捷函数
def get_cached_chat_model(model_name: str = "qwen-plus",
                         temperature: float = 0.7,
                         api_key: Optional[str] = None,
                         max_tokens: Optional[int] = None,
                         top_p: Optional[float] = None) -> ChatTongyi:
    """
    获取缓存的ChatTongyi模型实例（便捷函数）
    
    Args:
        model_name: 模型名称，默认qwen-plus
        temperature: 温度参数，默认0.7
        api_key: API密钥，如果不提供则从环境变量获取
        max_tokens: 最大输出token数
        top_p: 核采样参数
        
    Returns:
        ChatTongyi: 聊天模型实例
    """
    manager = get_global_model_manager()
    return manager.get_chat_model(model_name, temperature, api_key, max_tokens, top_p)


def get_cached_llm_model(model_name: str = "qwen-plus",
                        temperature: float = 0.7,
                        api_key: Optional[str] = None,
                        max_tokens: Optional[int] = None,
                        top_p: Optional[float] = None) -> Tongyi:
    """
    获取缓存的Tongyi LLM模型实例（便捷函数）
    
    Args:
        model_name: 模型名称，默认qwen-plus
        temperature: 温度参数，默认0.7
        api_key: API密钥，如果不提供则从环境变量获取
        max_tokens: 最大输出token数
        top_p: 核采样参数
        
    Returns:
        Tongyi: LLM模型实例
    """
    manager = get_global_model_manager()
    return manager.get_llm_model(model_name, temperature, api_key, max_tokens, top_p)