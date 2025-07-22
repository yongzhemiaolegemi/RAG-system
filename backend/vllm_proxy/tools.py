import json
import datetime
import requests
from typing import Dict, Any, List, Callable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import lightrag_service_url
import functools

# 定义日志装饰器：打印函数名和输入参数
def log_function_call(func: Callable) -> Callable:
    @functools.wraps(func)  # 保留原函数的元数据
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # 打印函数名
        print(f"******** 调用函数: {func.__name__} ********")
        
        # 打印参数
        if args:
            print(f"位置参数: {args}")
        if kwargs:
            print(f"关键字参数: {kwargs}")
        
        # 调用原函数并返回结果
        return func(*args, **kwargs)
    return wrapper


# ================================ 如果要添加新工具，需要修改从这里开始往上的代码。 ================================
# 修改流程为：
# 1. 在 AVAILABLE_TOOLS 中添加新工具的定义
# 2. 在 TOOL_FUNCTIONS 中添加新工具的映射 (这个改为了动态映射，不需要手动添加了)
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


def calculator(code: str) -> str:
    """计算数学运算的值。例如：3*5"""
    try:
        # 限制只允许数学运算相关的安全操作
        allowed_names = {
            'abs': abs,
            'pow': pow,
            'max': max,
            'min': min,
            'round': round,
            'sum': sum
        }

        # 使用eval计算表达式，限制可用函数以提高安全性
        result = eval(code, {"__builtins__": None}, allowed_names)
        return str(result)
    except SyntaxError:
        return f"语法错误：无法解析表达式 '{code}'"
    except NameError as e:
        return f"名称错误：{str(e)}"
    except TypeError as e:
        return f"类型错误：{str(e)}"
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误：{str(e)}" 

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
    },
        {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "用python解释器计算数学运算的值。例如：3*5",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "数学表达式，例如：3*5"},
                },
                "required": ["code"]
            }
        }
    }
]

# # 工具执行映射
# TOOL_FUNCTIONS: Dict[str, Callable] = {
#     "get_current_time": get_current_time,
#     "get_server_location": get_server_location,
#     "send_query_to_RAG_server": send_query_to_RAG_server,
# }

# 从AVAILABLE_TOOLS动态构建工具函数映射，并应用装饰器
TOOL_FUNCTIONS: Dict[str, Callable] = {}
for tool in AVAILABLE_TOOLS:
    func_name = tool["function"]["name"]
    if func_name in globals():
        # 获取原始函数并应用日志装饰器
        original_func = globals()[func_name]
        decorated_func = log_function_call(original_func)
        TOOL_FUNCTIONS[func_name] = decorated_func
    else:
        raise ValueError(f"工具函数 '{func_name}' 未定义，请检查实现")





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