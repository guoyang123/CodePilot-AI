#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通义千问模型管理器使用示例
展示如何使用TongyiModelManager实现模型缓存和复用
"""

import os
from llm_tongyi import (
    TongyiLLM,
    TongyiModelManager,
    get_global_model_manager,
    get_cached_chat_model,
    get_cached_llm_model
)

def example_1_basic_usage():
    """
    示例1: 基本使用方法
    """
    print("\n=== 示例1: 基本使用方法 ===")
    
    # 方法1: 传统方式（每次创建新实例）
    print("\n传统方式:")
    tongyi_llm1 = TongyiLLM(model_name="qwen-plus", temperature=0.7)
    model1 = tongyi_llm1.get_chat_model()
    print(f"模型1 ID: {id(model1)}")
    
    tongyi_llm2 = TongyiLLM(model_name="qwen-plus", temperature=0.7)
    model2 = tongyi_llm2.get_chat_model()
    print(f"模型2 ID: {id(model2)}")
    print(f"是否为同一个对象: {model1 is model2}")
    
    # 方法2: 使用模型管理器（缓存复用）
    print("\n使用模型管理器:")
    manager = TongyiModelManager()
    
    cached_model1 = manager.get_chat_model(model_name="qwen-plus", temperature=0.7)
    print(f"缓存模型1 ID: {id(cached_model1)}")
    
    cached_model2 = manager.get_chat_model(model_name="qwen-plus", temperature=0.7)
    print(f"缓存模型2 ID: {id(cached_model2)}")
    print(f"是否为同一个对象: {cached_model1 is cached_model2}")
    
    # 显示缓存信息
    cache_info = manager.get_cache_info()
    print(f"缓存信息: {cache_info}")


def example_2_global_manager():
    """
    示例2: 使用全局管理器
    """
    print("\n=== 示例2: 使用全局管理器 ===")
    
    # 使用便捷函数获取模型
    model1 = get_cached_chat_model(model_name="qwen-plus", temperature=0.7)
    model2 = get_cached_chat_model(model_name="qwen-plus", temperature=0.7)
    
    print(f"全局模型1 ID: {id(model1)}")
    print(f"全局模型2 ID: {id(model2)}")
    print(f"是否为同一个对象: {model1 is model2}")
    
    # 获取全局管理器实例
    global_manager = get_global_model_manager()
    cache_info = global_manager.get_cache_info()
    print(f"全局管理器缓存信息: {cache_info}")


def example_3_different_models():
    """
    示例3: 不同模型配置的缓存
    """
    print("\n=== 示例3: 不同模型配置的缓存 ===")
    
    manager = TongyiModelManager(max_cache_size=5)
    
    # 创建不同配置的模型
    configs = [
        {"model_name": "qwen-plus", "temperature": 0.7, "name": "标准配置"},
        {"model_name": "qwen-plus", "temperature": 0.5, "name": "低温度配置"},
        {"model_name": "qwen-turbo", "temperature": 0.7, "name": "快速模型"},
        {"model_name": "qwen-plus", "temperature": 0.9, "name": "高温度配置"},
    ]
    
    models = {}
    
    for config in configs:
        name = config.pop("name")
        model = manager.get_chat_model(**config)
        models[name] = model
        
        cache_info = manager.get_cache_info()
        print(f"{name}: 模型ID {id(model)}, 缓存数量: {cache_info['chat_model_count']}")
    
    # 再次获取第一个配置，应该使用缓存
    print("\n再次获取标准配置:")
    model_again = manager.get_chat_model(model_name="qwen-plus", temperature=0.7)
    print(f"是否使用缓存: {models['标准配置'] is model_again}")
    
    final_cache_info = manager.get_cache_info()
    print(f"最终缓存信息: {final_cache_info}")


def example_4_cache_management():
    """
    示例4: 缓存管理操作
    """
    print("\n=== 示例4: 缓存管理操作 ===")
    
    manager = TongyiModelManager(max_cache_size=3)
    
    # 创建一些模型
    model1 = manager.get_chat_model(model_name="qwen-plus", temperature=0.7)
    model2 = manager.get_chat_model(model_name="qwen-plus", temperature=0.5)
    
    print(f"创建2个模型后的缓存: {manager.get_cache_info()['chat_model_count']}")
    
    # 移除特定模型
    removed = manager.remove_model(model_name="qwen-plus", temperature=0.7)
    print(f"移除模型成功: {removed}")
    print(f"移除后的缓存: {manager.get_cache_info()['chat_model_count']}")
    
    # 清空所有缓存
    manager.clear_cache()
    print(f"清空后的缓存: {manager.get_cache_info()['chat_model_count']}")


def example_5_performance_comparison():
    """
    示例5: 性能对比
    """
    print("\n=== 示例5: 性能对比 ===")
    
    import time
    
    # 测试传统方式
    print("\n测试传统方式（3次创建）:")
    start_time = time.time()
    for i in range(3):
        tongyi_llm = TongyiLLM(model_name="qwen-plus", temperature=0.7)
        model = tongyi_llm.get_chat_model()
    traditional_time = time.time() - start_time
    print(f"传统方式耗时: {traditional_time:.4f}秒")
    
    # 测试缓存方式
    print("\n测试缓存方式（3次获取）:")
    manager = TongyiModelManager()
    start_time = time.time()
    for i in range(3):
        model = manager.get_chat_model(model_name="qwen-plus", temperature=0.7)
    cached_time = time.time() - start_time
    print(f"缓存方式耗时: {cached_time:.4f}秒")
    
    if cached_time > 0:
        improvement = traditional_time / cached_time
        print(f"性能提升: {improvement:.2f}倍")
    
    cache_info = manager.get_cache_info()
    print(f"缓存信息: {cache_info}")


def example_6_real_world_usage():
    """
    示例6: 实际应用场景
    """
    print("\n=== 示例6: 实际应用场景 ===")
    
    # 模拟一个需要多次使用模型的应用
    def process_multiple_requests():
        """模拟处理多个请求，每个请求都需要使用模型"""
        requests = [
            "今天天气怎么样？",
            "明天会下雨吗？",
            "这周末适合出游吗？",
            "下周的天气预报如何？"
        ]
        
        print("\n处理多个请求（使用缓存模型）:")
        for i, request in enumerate(requests, 1):
            # 每次请求都获取模型，但实际上会复用缓存的实例
            model = get_cached_chat_model(model_name="qwen-plus", temperature=0.7)
            print(f"请求 {i}: {request}")
            print(f"  使用模型ID: {id(model)}")
    
    process_multiple_requests()
    
    # 显示全局缓存状态
    global_manager = get_global_model_manager()
    final_cache_info = global_manager.get_cache_info()
    print(f"\n最终全局缓存状态: {final_cache_info}")


def main():
    """
    主函数：运行所有示例
    """
    print("通义千问模型管理器使用示例")
    print("=" * 50)
    
    # 检查API密钥
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("警告: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("某些示例可能会失败")
        return
    
    try:
        # 运行所有示例
        example_1_basic_usage()
        example_2_global_manager()
        example_3_different_models()
        example_4_cache_management()
        example_5_performance_comparison()
        example_6_real_world_usage()
        
    except Exception as e:
        print(f"\n示例运行过程中出现错误: {e}")
        print("请确保已正确设置 DASHSCOPE_API_KEY 环境变量")
    
    print("\n" + "="*50)
    print("所有示例运行完成")
    print("="*50)
    
    # 最终建议
    print("\n使用建议:")
    print("1. 对于简单应用，直接使用 get_cached_chat_model() 函数")
    print("2. 对于复杂应用，可以创建自己的 TongyiModelManager 实例")
    print("3. 相同配置的模型会自动复用，提高性能")
    print("4. 可以通过 get_cache_info() 监控缓存状态")
    print("5. 使用 clear_cache() 在需要时清理缓存")


if __name__ == "__main__":
    main()