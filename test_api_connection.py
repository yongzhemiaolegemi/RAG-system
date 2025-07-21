#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API连接测试脚本
用于诊断Django后端的API连接问题
"""

import requests
import sys
import time
from openai import OpenAI
from config import django_llm_url, django_llm_key, django_model

def test_api_connection():
    """测试API连接"""
    print("🔍 开始测试API连接...")
    print(f"API URL: {django_llm_url}")
    print(f"API Key: {django_llm_key[:20]}...")
    print(f"模型: {django_model}")
    print("-" * 50)
    
    try:
        # 创建OpenAI客户端
        print("1. 创建OpenAI客户端...")
        client = OpenAI(
            api_key=django_llm_key,
            base_url=django_llm_url
        )
        print("✅ 客户端创建成功")
        
        # 测试简单的API调用
        print("\n2. 测试简单API调用...")
        test_messages = [
            {"role": "user", "content": "Hello, please respond with 'API connection test successful'"}
        ]
        
        response = client.chat.completions.create(
            model=django_model,
            messages=test_messages,
            max_tokens=50,
            stream=False
        )
        
        print("✅ API调用成功")
        print(f"响应: {response.choices[0].message.content}")
        
        # 测试流式调用
        print("\n3. 测试流式API调用...")
        stream_response = client.chat.completions.create(
            model=django_model,
            messages=test_messages,
            max_tokens=50,
            stream=True
        )
        
        collected_content = ""
        for chunk in stream_response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    collected_content += delta.content
        
        print("✅ 流式调用成功")
        print(f"流式响应: {collected_content}")
        
        # 测试工具调用
        print("\n4. 测试工具调用...")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前时间",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
        
        tool_messages = [
            {"role": "user", "content": "请调用工具获取当前时间"}
        ]
        
        tool_response = client.chat.completions.create(
            model=django_model,
            messages=tool_messages,
            tools=tools,
            tool_choice="auto",
            stream=False
        )
        
        print("✅ 工具调用测试成功")
        if tool_response.choices[0].message.tool_calls:
            print(f"工具调用: {tool_response.choices[0].message.tool_calls}")
        else:
            print("模型未使用工具，但调用成功")
        
        print("\n🎉 所有测试通过！API连接正常。")
        return True
        
    except Exception as e:
        print(f"\n❌ API连接测试失败")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        
        # 提供诊断建议
        print("\n🔧 诊断建议:")
        if "Connection" in str(e):
            print("- 检查网络连接")
            print("- 确认API URL是否正确")
            print("- 检查防火墙设置")
        elif "401" in str(e) or "Unauthorized" in str(e):
            print("- 检查API密钥是否正确")
            print("- 确认API密钥是否过期")
        elif "404" in str(e):
            print("- 检查API URL路径是否正确")
            print("- 确认模型名称是否支持")
        elif "timeout" in str(e).lower():
            print("- 网络超时，请检查网络连接")
            print("- 可能是API服务器响应慢")
        else:
            print("- 查看完整错误堆栈信息")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
        
        return False

def test_django_endpoint():
    """测试Django端点"""
    print("\n🔍 测试Django端点...")
    
    from config import django_service_url
    
    try:
        test_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": django_model,
            "stream": True,
            "collect_stream": True,
            "enable_tools": False  # 先测试不带工具的简单调用
        }
        
        response = requests.post(
            django_service_url,
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Django端点测试成功")
            result = response.json()
            if "choices" in result:
                print(f"响应内容: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"❌ Django端点测试失败")
            print(f"状态码: {response.status_code}")
            print(f"响应: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Django端点连接失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🧪 API连接诊断工具")
    print("=" * 60)
    
    # 首先测试直接API连接
    api_ok = test_api_connection()
    
    if api_ok:
        # 如果API连接正常，测试Django端点
        django_ok = test_django_endpoint()
        
        if django_ok:
            print("\n🎉 所有测试通过！系统配置正常。")
        else:
            print("\n⚠️ Django端点有问题，请检查Django服务器配置。")
    else:
        print("\n❌ API连接有问题，请根据上述建议检查配置。")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 