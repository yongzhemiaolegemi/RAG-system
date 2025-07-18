import json
import datetime
import requests
from typing import Dict, Any, List, Callable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import lightrag_service_url

# ================================ 如果要添加新工具，需要修改从这里开始往上的代码。 ================================
# 修改流程为：
# 1. 在 AVAILABLE_TOOLS 中添加新工具的定义
# 2. 在 TOOL_FUNCTIONS 中添加新工具的映射
# 3. 在 开头 def 你的工具函数体

def get_current_time() -> str:
    """获取当前时间"""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def get_server_location() -> str:
    """获取当前服务器的地理位置"""
    return "中国上海"

def send_query_to_RAG_server(query: str, mode: str = "hybrid", url: str = lightrag_service_url) -> str:
    """向RAG服务器发送查询"""
    # POST to url
    # {
    #     "message": "你的查询问题",
    #     "mode": "查询模式"
    # }
    data = {
        "message": query,
        "mode": mode
    }
    response = requests.post(url, json=data)
    return response.json()["result"]

# 工具定义
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_server_location",
            "description": "获取当前服务器的地理位置",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_query_to_RAG_server",
            "description": "向RAG服务器发送查询，返回查询结果。发送查询的语言和返回查询结果的语言都是自然语言，以问答的形式进行。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "你想要询问RAG系统的问题"},
                    "mode": {"type": "string", "description": "查询模式，一般为hybrid。不提供这个参数的话，默认也是hybrid。", "enum": ["naive", "local", "global", "hybrid"]}
                },
                "required": ["query"]
            }
        }
    }
]

# 工具执行映射
TOOL_FUNCTIONS: Dict[str, Callable] = {
    "get_current_time": get_current_time,
    "get_server_location": get_server_location,
    "send_query_to_RAG_server": send_query_to_RAG_server,
}





# ================================ 如果要添加新工具，需要修改从这里开始往上的代码。 ================================

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """执行指定的工具"""
    if tool_name not in TOOL_FUNCTIONS:
        return f"错误: 未找到工具 {tool_name}"
    
    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**arguments)
        return str(result)
    except Exception as e:
        return f"工具执行错误: {str(e)}"

def format_tool_call_message(tool_calls: List[Dict]) -> Dict:
    """格式化工具调用消息"""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls
    }

def format_tool_result_message(tool_call_id: str, tool_name: str, result: str) -> Dict:
    """格式化工具执行结果消息"""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": result
    }

def parse_tool_calls(response_content: str) -> List[Dict]:
    """从模型响应中解析工具调用"""
    # 这里可以根据具体的模型响应格式来解析
    # 暂时返回空列表，实际实现需要根据模型的具体响应格式
    return [] 