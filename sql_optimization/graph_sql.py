#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于LangGraph的SQL优化工具
实现：SQL信息输入 -> LLM SQL分析 -> 优化建议生成 的步骤
支持LangSmith追踪和监控
"""

import os
import sys
import logging
import requests
from typing import TypedDict, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

# LangSmith支持
try:
    from langsmith import traceable
    from langsmith import Client as LangSmithClient
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    # 如果LangSmith不可用，创建一个空的装饰器
    def traceable(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator

# LangGraph相关
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入项目中的LLM模块
from llm.llm_tongyi import get_cached_chat_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_langsmith():
    """配置LangSmith追踪"""
    if LANGSMITH_AVAILABLE:
        # 设置环境变量（如果未设置）
        if not os.getenv('LANGCHAIN_TRACING_V2'):
            os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        
        # 检查API密钥
        api_key = os.getenv('LANGCHAIN_API_KEY')
        if not api_key:
            logger.warning("LANGCHAIN_API_KEY未设置，LangSmith追踪将不可用")
            return False
        
        # 设置项目名称
        if not os.getenv('LANGCHAIN_PROJECT'):
            os.environ['LANGCHAIN_PROJECT'] = 'sql-optimization-workflow'
        
        logger.info(f"LangSmith追踪已启用，项目: {os.getenv('LANGCHAIN_PROJECT')}")
        return True
    else:
        logger.info("LangSmith不可用，跳过追踪配置")
        return False

# 初始化LangSmith配置
langsmith_enabled = configure_langsmith()


class SQLOptimizationState(TypedDict):
    """SQL优化工作流的状态定义"""
    origin_sql: str  # 输入的原始SQL
    type: Optional[str]  # 数据库类型，默认MySQL
    ddl: Optional[str]  # 建表语句
    analysis: Optional[str]  # SQL分析结果
    optimization: Optional[str]  # 优化建议内容
    markdown_file_path: Optional[str]  # 生成的markdown文件路径
    error: Optional[str]  # 错误信息
    metadata: Optional[Dict[str, Any]]  # 元数据信息


# 工作流节点函数
@traceable(name="validate_sql_node")
def validate_sql_node(state: SQLOptimizationState) -> SQLOptimizationState:
    """验证SQL输入节点"""
    logger.info("开始验证SQL输入")
    
    try:
        # 检查SQL是否为空
        if not state.get("origin_sql") or not state["origin_sql"].strip():
            raise ValueError("SQL语句不能为空")
        
        # 基本SQL语法检查
        sql = state["origin_sql"].strip().upper()
        valid_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
        
        if not any(sql.startswith(keyword) for keyword in valid_keywords):
            raise ValueError("无效的SQL语句，必须以有效的SQL关键字开头")
        
        # 设置默认数据库类型
        if not state.get("type"):
            state["type"] = "MySQL"
        
        # 初始化元数据
        state["metadata"] = {
            "validation_time": datetime.now().isoformat(),
            "sql_length": len(state["origin_sql"]),
            "database_type": state["type"]
        }
        
        logger.info(f"SQL验证成功，数据库类型: {state['type']}")
        return state
        
    except Exception as e:
        error_msg = f"SQL验证失败: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        return state


@traceable(name="analyze_sql_node")
def analyze_sql_node(state: SQLOptimizationState) -> SQLOptimizationState:
    """SQL分析节点 - 使用通义千问模型分析SQL"""
    logger.info("开始SQL分析")
    
    # 如果前面有错误，直接返回
    if state.get("error"):
        return state
    
    try:
        # 获取通义千问模型
        logger.info("正在获取通义千问模型...")
        llm = get_cached_chat_model()
        logger.info("通义千问模型获取成功")
        
        # 构建分析提示词
        ddl_section = ""
        if state.get('ddl'):
            ddl_section = f"建表语句:\n```sql\n{state['ddl']}\n```\n"
        
        analysis_prompt = f"""
请对以下SQL语句进行详细分析：

数据库类型: {state.get('type', 'MySQL')}
SQL语句:
```sql
{state['origin_sql']}
```
建表语句:
```sql
{ddl_section}
```

请从以下几个方面进行分析：
1. SQL语句的功能和目的
2. 涉及的表和字段
3. 查询逻辑分析
4. 可能的性能问题
5. 索引使用情况
6. JOIN操作分析（如果有）
7. WHERE条件分析
8. 潜在的安全风险

请用中文回答，分析要详细且专业。
"""
        
        logger.info("正在调用模型进行SQL分析...")
        # 调用模型进行分析
        messages = [HumanMessage(content=analysis_prompt)]
        response = llm.invoke(messages)
        logger.info("模型调用完成")
        
        # 提取分析结果
        analysis_result = response.content if hasattr(response, 'content') else str(response)
        state["analysis"] = analysis_result
        logger.info(f"分析结果长度: {len(analysis_result)} 字符")
        
        # 更新元数据
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["analysis_time"] = datetime.now().isoformat()
        state["metadata"]["analysis_length"] = len(analysis_result)
        
        logger.info("SQL分析完成")
        return state
        
    except Exception as e:
        error_msg = f"SQL分析失败: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        return state


@traceable(name="generate_optimization_node")
def generate_optimization_node(state: SQLOptimizationState) -> SQLOptimizationState:
    """生成SQL优化建议节点"""
    logger.info("开始生成优化建议")
    
    # 如果前面有错误，直接返回
    if state.get("error"):
        return state
    
    try:
        # 获取通义千问模型
        logger.info("正在获取通义千问模型...")
        llm = get_cached_chat_model()
        logger.info("通义千问模型获取成功")
        
        # 构建优化建议提示词
        ddl_section = ""
        if state.get('ddl'):
            ddl_section = f"建表语句:\n```sql\n{state['ddl']}\n```\n"
        
        optimization_prompt = f"""
基于以下SQL分析结果，请提供详细的优化建议：

数据库类型: {state.get('type', 'MySQL')}
原始SQL:
```sql
{state['origin_sql']}
```

{ddl_section}
SQL分析结果:
{state.get('analysis', '暂无分析结果')}

请提供以下优化建议：
1. 索引优化建议
2. 查询重写建议
3. 表结构优化建议
4. 性能优化技巧
5. 最佳实践建议
6. 优化后的SQL示例（如果可以改进）
7. 预期的性能提升效果

请用中文回答，建议要具体可操作，并提供代码示例。
"""
        
        logger.info("正在调用模型生成优化建议...")
        # 调用模型生成优化建议
        messages = [HumanMessage(content=optimization_prompt)]
        response = llm.invoke(messages)
        logger.info("模型调用完成")
        
        # 提取优化建议
        optimization_result = response.content if hasattr(response, 'content') else str(response)
        state["optimization"] = optimization_result
        logger.info(f"优化建议长度: {len(optimization_result)} 字符")
        
        # 更新元数据
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["optimization_time"] = datetime.now().isoformat()
        state["metadata"]["optimization_length"] = len(optimization_result)
        
        logger.info("优化建议生成完成")
        return state
        
    except Exception as e:
        error_msg = f"优化建议生成失败: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        return state


@traceable(name="write_markdown_node")
def write_markdown_node(state: SQLOptimizationState) -> SQLOptimizationState:
    """写入Markdown文件节点"""
    logger.info("开始写入Markdown文件")
    
    # 如果前面有错误，直接返回
    if state.get("error"):
        return state
    
    try:
        # 确保输出目录存在
        output_dir = Path("docs/SQL优化")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SQL优化分析_{timestamp}.md"
        file_path = output_dir / filename
        
        # 构建Markdown内容
        ddl_section = ""
        if state.get('ddl'):
            ddl_section = f"## 建表语句\n\n```sql\n{state['ddl']}\n```\n"
        
        markdown_content = f"""# SQL优化分析报告

## 基本信息
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **数据库类型**: {state.get('type', 'MySQL')}
- **SQL长度**: {len(state['origin_sql'])} 字符

## 原始SQL语句

```sql
{state['origin_sql']}
```

{ddl_section}
## SQL分析结果

{state.get('analysis', '暂无分析结果')}

## 优化建议

{state.get('optimization', '暂无优化建议')}

## 元数据信息

{_format_metadata(state.get('metadata', {}))}

---
*本报告由SQL优化工具自动生成*
"""
        
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # 更新状态
        state["markdown_file_path"] = str(file_path)
        
        # 更新元数据
        if "metadata" not in state:
            state["metadata"] = {}
        state["metadata"]["markdown_write_time"] = datetime.now().isoformat()
        state["metadata"]["markdown_file_path"] = str(file_path)
        state["metadata"]["markdown_file_size"] = len(markdown_content)
        
        logger.info(f"Markdown文件写入成功: {file_path}")
        return state
        
    except Exception as e:
        error_msg = f"Markdown文件写入失败: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        return state


def _format_metadata(metadata: Dict[str, Any]) -> str:
    """格式化元数据为Markdown表格"""
    if not metadata:
        return "暂无元数据"
    
    lines = ["| 项目 | 值 |", "| --- | --- |"]
    for key, value in metadata.items():
        lines.append(f"| {key} | {value} |")
    
    return "\n".join(lines)


# 工作流创建和运行函数
def create_sql_optimization_workflow() -> StateGraph:
    """创建SQL优化工作流"""
    logger.info("创建SQL优化工作流")
    
    # 创建状态图
    workflow = StateGraph(SQLOptimizationState)
    
    # 添加节点
    workflow.add_node("validate_sql", validate_sql_node)  # SQL验证节点
    workflow.add_node("analyze_sql", analyze_sql_node)    # SQL分析节点
    workflow.add_node("generate_optimization", generate_optimization_node)  # 优化建议生成节点
    workflow.add_node("write_markdown", write_markdown_node)  # Markdown写入节点
    
    # 定义工作流边
    workflow.add_edge(START, "validate_sql")  # 开始 -> SQL验证
    workflow.add_edge("validate_sql", "analyze_sql")  # SQL验证 -> SQL分析
    workflow.add_edge("analyze_sql", "generate_optimization")  # SQL分析 -> 优化建议生成
    workflow.add_edge("generate_optimization", "write_markdown")  # 优化建议生成 -> Markdown写入
    workflow.add_edge("write_markdown", END)  # Markdown写入 -> 结束
    
    logger.info("SQL优化工作流创建完成")
    return workflow


def run_sql_optimization_workflow(origin_sql: str, 
                                 db_type: str = "MySQL", 
                                 ddl: Optional[str] = None) -> SQLOptimizationState:
    """运行SQL优化工作流
    
    Args:
        origin_sql: 原始SQL语句
        db_type: 数据库类型，默认MySQL
        ddl: 建表语句，可选
        
    Returns:
        SQLOptimizationState: 工作流执行结果
    """
    logger.info("开始运行SQL优化工作流")
    
    try:
        # 创建工作流
        workflow = create_sql_optimization_workflow()
        app = workflow.compile()
        
        # 初始化状态
        initial_state: SQLOptimizationState = {
            "origin_sql": origin_sql,
            "type": db_type,
            "ddl": ddl,
            "analysis": None,
            "optimization": None,
            "markdown_file_path": None,
            "error": None,
            "metadata": None
        }
        
        # 运行工作流
        result = app.invoke(initial_state)
        
        logger.info("SQL优化工作流运行完成")
        return result
        
    except Exception as e:
        error_msg = f"工作流运行失败: {str(e)}"
        logger.error(error_msg)
        return {
            "origin_sql": origin_sql,
            "type": db_type,
            "ddl": ddl,
            "analysis": None,
            "optimization": None,
            "markdown_file_path": None,
            "error": error_msg,
            "metadata": None
        }


# 测试用例
if __name__ == "__main__":
    # 测试SQL语句
    test_sql = """
    SELECT u.id, u.name, u.email, p.title, p.content, p.created_at
    FROM users u
    LEFT JOIN posts p ON u.id = p.user_id
    WHERE u.status = 'active'
    AND p.created_at > '2024-01-01'
    ORDER BY p.created_at DESC
    LIMIT 100
    """
    
    # 测试建表语句
    test_ddl = """
    CREATE TABLE users (
        id INT PRIMARY KEY AUTO_INCREMENT,
        name VARCHAR(100) NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        status ENUM('active', 'inactive') DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE TABLE posts (
        id INT PRIMARY KEY AUTO_INCREMENT,
        user_id INT NOT NULL,
        title VARCHAR(255) NOT NULL,
        content TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """
    
    print("开始测试SQL优化工作流...")
    
    # 运行工作流
    result = run_sql_optimization_workflow(
        origin_sql=test_sql,
        db_type="MySQL",
        ddl=test_ddl
    )
    
    # 输出结果
    if result.get("error"):
        print(f"工作流执行失败: {result['error']}")
    else:
        print(f"工作流执行成功!")
        print(f"Markdown文件路径: {result.get('markdown_file_path')}")
        print(f"元数据: {result.get('metadata')}")
        
        # 如果有分析结果，显示前200个字符
        if result.get("analysis"):
            print(f"\nSQL分析结果预览: {result['analysis'][:200]}...")
        
        # 如果有优化建议，显示前200个字符
        if result.get("optimization"):
            print(f"\n优化建议预览: {result['optimization'][:200]}...")

