from django.shortcuts import render
from openai import OpenAI

# Create your views here.
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
import uuid
from .tools import AVAILABLE_TOOLS, execute_tool, format_tool_call_message, format_tool_result_message
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import django_llm_url, django_llm_key, django_model, django_vllm_url

VLLM_URL = django_vllm_url  # VLLM的URL

def post_to_openai_api(messages, model, stream, collect_stream=False, tools=None):
    client = OpenAI(
        api_key=django_llm_key,
        base_url=django_llm_url
        )
    
    # 构建API调用参数
    api_params = {
        "model": model,
        "messages": messages,
        "extra_body": {"enable_thinking": True},
        "stream": stream,
    }
    
    # 如果提供了工具，添加到API调用中
    if tools:
        api_params["tools"] = tools
        api_params["tool_choice"] = "auto"
    
    completion = client.chat.completions.create(**api_params)
    
    if stream:
        if collect_stream:
            # 收集所有流式数据并拼接
            collected_content = ""
            tool_calls = []
            full_response = None
            
            for chunk in completion:
                if chunk.choices and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    delta = choice.delta
                    
                    # 收集内容
                    if hasattr(delta, 'content') and delta.content:
                        collected_content += delta.content
                    
                    # 收集工具调用
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call.index >= len(tool_calls):
                                tool_calls.extend([None] * (tool_call.index + 1 - len(tool_calls)))
                            
                            if tool_calls[tool_call.index] is None:
                                tool_calls[tool_call.index] = {
                                    "id": tool_call.id or f"call_{uuid.uuid4().hex[:8]}",
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name or "",
                                        "arguments": tool_call.function.arguments or ""
                                    }
                                }
                            else:
                                # 追加参数
                                if tool_call.function.arguments:
                                    tool_calls[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
                    
                    # 保存最后一个chunk作为模板
                    full_response = chunk.model_dump()
            
            # 构建完整响应
            if full_response:
                choice_data = {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": collected_content if collected_content else None,
                    },
                    "finish_reason": "stop"
                }
                
                # 如果有工具调用，添加到响应中
                if tool_calls and any(tc for tc in tool_calls):
                    choice_data["message"]["tool_calls"] = [tc for tc in tool_calls if tc]
                    choice_data["finish_reason"] = "tool_calls"
                
                full_response["choices"] = [choice_data]
                
            return full_response or {
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant", 
                        "content": collected_content if collected_content else None
                    },
                    "finish_reason": "stop"
                }]
            }
        else:
            # 返回生成器函数用于流式响应
            def generate():
                for chunk in completion:
                    chunk_data = chunk.model_dump()
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                yield "data: [DONE]\n\n"
            return generate()
    else:
        # 非流式响应，直接返回字典
        return completion.model_dump()

def handle_tool_calls(messages, model, stream, collect_stream, max_iterations=5):
    """处理工具调用的多轮对话"""
    current_messages = messages.copy()
    iteration = 0
    
    while iteration < max_iterations:
        # 调用API
        response = post_to_openai_api(
            messages=current_messages,
            model=model,
            stream=stream,
            collect_stream=collect_stream or not stream,  # 工具调用时需要完整响应
            tools=AVAILABLE_TOOLS
        )
        
        # 如果是流式且未收集，直接返回
        if stream and not collect_stream:
            return response
        
        # 检查是否有工具调用
        if (response.get("choices") and 
            len(response["choices"]) > 0 and 
            response["choices"][0].get("message", {}).get("tool_calls")):
            
            # 添加助手的工具调用消息
            assistant_message = response["choices"][0]["message"]
            current_messages.append(assistant_message)
            
            # 执行工具调用
            tool_calls = assistant_message["tool_calls"]
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                try:
                    arguments = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    arguments = {}
                
                # 执行工具
                result = execute_tool(tool_name, arguments)
                
                # 添加工具结果消息
                tool_message = format_tool_result_message(
                    tool_call["id"], 
                    tool_name, 
                    result
                )
                current_messages.append(tool_message)
            
            iteration += 1
        else:
            # 没有工具调用，返回最终响应
            return response
    
    # 达到最大迭代次数
    return {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "抱歉，工具调用达到最大迭代次数限制。"
            },
            "finish_reason": "stop"
        }]
    }

@csrf_exempt
def vllm_proxy(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are accepted'}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    # 获取参数，默认值
    stream = data.get('stream', False)
    collect_stream = data.get('collect_stream', False)
    enable_tools = data.get('enable_tools', True)  # 默认启用工具
    
    # 构建消息列表
    if 'messages' in data:
        messages = data['messages']
    elif 'prompt' in data:
        messages = [{"role": "user", "content": data['prompt']}]
    else:
        return JsonResponse({'error': 'Missing messages or prompt'}, status=400)
    
    model = data.get('model', 'qwq-plus-latest')

    try:
        if enable_tools:
            # 使用工具调用处理
            response_data = handle_tool_calls(
                messages=messages,
                model=model,
                stream=stream,
                collect_stream=collect_stream
            )
        else:
            # 不使用工具，直接调用API
            response_data = post_to_openai_api(
                messages=messages,
                model=model,
                stream=stream,
                collect_stream=collect_stream
            )
        
        if stream and not collect_stream:
            # 真正的流式响应
            response = StreamingHttpResponse(response_data, content_type='text/event-stream')
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'
            return response
        else:
            # 非流式响应或收集后的完整响应
            return JsonResponse(response_data)
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
