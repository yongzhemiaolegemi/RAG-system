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
    try:
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
        
        print(f"调用API: {django_llm_url}")
        print(f"模型: {model}")
        print(f"消息数量: {len(messages)}")
        
        completion = client.chat.completions.create(**api_params)
        
    except Exception as api_error:
        print(f"API调用错误详情: {str(api_error)}")
        print(f"API URL: {django_llm_url}")
        print(f"API Key: {django_llm_key[:20]}...")
        raise api_error
    
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
    """处理工具调用的多轮对话，返回完整的对话历史和最终响应"""
    current_messages = messages.copy()
    iteration = 0
    all_messages = []  # 用于收集所有的对话消息，包括工具调用
    
    # 对于工具调用，我们总是需要完整的响应来解析工具调用，所以：
    # 1. 如果原本就是collect_stream=True，保持不变
    # 2. 如果原本是stream=True但collect_stream=False，我们需要强制收集来处理工具调用
    force_collect = stream and not collect_stream  # 是否需要强制收集
    actual_collect_stream = collect_stream or force_collect
    
    while iteration < max_iterations:
        # 调用API
        response = post_to_openai_api(
            messages=current_messages,
            model=model,
            stream=stream,
            collect_stream=actual_collect_stream,
            tools=AVAILABLE_TOOLS
        )
        
        # 检查是否有工具调用
        if (response.get("choices") and 
            len(response["choices"]) > 0 and 
            response["choices"][0].get("message", {}).get("tool_calls")):
            
            # 添加助手的工具调用消息
            assistant_message = response["choices"][0]["message"]
            current_messages.append(assistant_message)
            all_messages.append(assistant_message)
            
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
                all_messages.append(tool_message)
            
            iteration += 1
        else:
            # 没有工具调用，返回最终响应
            # 如果有工具调用历史，需要将最终的assistant消息也添加到历史中
            if all_messages:
                final_assistant_message = response["choices"][0]["message"]
                all_messages.append(final_assistant_message)
                # 创建包含完整对话历史的响应
                enhanced_response = response.copy()
                enhanced_response["conversation_messages"] = all_messages
                return enhanced_response, current_messages, force_collect
            else:
                return response, current_messages, False
    
    # 达到最大迭代次数
    error_response = {
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "抱歉，工具调用达到最大迭代次数限制。"
            },
            "finish_reason": "stop"
        }],
        "conversation_messages": all_messages
    }
    return error_response, current_messages, force_collect

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
            response_data, final_messages, was_force_collected = handle_tool_calls(
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
            final_messages = messages
            was_force_collected = False
        
        # 决定返回格式的逻辑：
        # 1. 如果原始请求是stream=True且collect_stream=False，且没有被强制收集，返回流式
        # 2. 其他情况（包括collect_stream=True的情况）都返回完整响应
        should_return_stream = stream and not collect_stream and not was_force_collected
        
        if should_return_stream:
            # 真正的流式响应
            response = StreamingHttpResponse(response_data, content_type='text/event-stream')
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'
            return response
        else:
            # 非流式响应或收集后的完整响应
            # 为了客户端的多轮对话需求，我们可以选择性地返回对话历史
            if data.get('include_conversation_history', False):
                response_data['final_messages'] = final_messages
            return JsonResponse(response_data)
            
    except Exception as e:
        print(f"Django视图错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print(f"错误堆栈: {traceback.format_exc()}")
        
        # 返回更详细的错误信息
        error_info = {
            'error': str(e),
            'error_type': type(e).__name__,
            'config': {
                'api_url': django_llm_url,
                'model': data.get('model', 'qwq-plus-latest'),
                'stream': data.get('stream', False),
                'collect_stream': data.get('collect_stream', False),
                'enable_tools': data.get('enable_tools', True)
            }
        }
        return JsonResponse(error_info, status=500)
