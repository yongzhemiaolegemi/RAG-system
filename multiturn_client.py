#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import sys
from typing import List, Dict
from utils import config, set_config

class MultiTurnChatClient:
    def __init__(self, base_url: str = config().django_service_url):
        self.base_url = base_url
        self.conversation_history: List[Dict[str, str]] = []
        self.session = requests.Session()
    
    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        self.conversation_history.append({"role": role, "content": content})
    
    def send_message(self, user_input: str, enable_tools: bool = True) -> str:
        """发送用户消息并获取回复"""
        # 添加用户消息到历史
        self.add_message("user", user_input)
        
        # 构建请求数据
        # 使用stream=True和collect_stream=True以兼容只支持流式输出的模型
        request_data = {
            "messages": self.conversation_history,
            "model": config().django_model,
            "stream": True,
            "collect_stream": True,
            "enable_tools": enable_tools
        }
        
        try:
            # 发送请求
            response = self.session.post(
                self.base_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=1200  # 2分钟超时
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    assistant_message = response_data["choices"][0]["message"]["content"]
                    if assistant_message:
                        # 添加助手回复到历史
                        self.add_message("assistant", assistant_message)
                        return assistant_message
                    else:
                        return "助手没有返回内容。"
                else:
                    return f"API响应格式错误: {response_data}"
            else:
                return f"请求失败，状态码: {response.status_code}, 错误: {response.text}"
                
        except requests.exceptions.Timeout:
            return "请求超时，请稍后重试。"
        except requests.exceptions.ConnectionError:
            return f"连接失败，请确认服务器 {self.base_url} 是否正常运行。"
        except Exception as e:
            return f"发送消息时发生错误: {str(e)}"
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        print("对话历史已清空。")
    
    def show_history(self):
        """显示对话历史"""
        if not self.conversation_history:
            print("暂无对话历史。")
            return
        
        print("\n=== 对话历史 ===")
        for i, message in enumerate(self.conversation_history, 1):
            role = "用户" if message["role"] == "user" else "助手"
            print(f"{i}. {role}: {message['content']}")
        print("=== 历史结束 ===\n")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
=== 多轮对话客户端帮助 ===

基本用法：
- 直接输入你的问题，系统会给出回答
- 系统支持工具调用，可以查询时间、位置和RAG知识库

特殊命令：
- /help    : 显示此帮助信息
- /clear   : 清空对话历史
- /history : 查看对话历史
- /exit 或 /quit : 退出程序

示例问题：
- "现在几点了？"
- "服务器在哪里？"
- "查询一下狄更斯的相关信息"
- "继续上一个话题..."

注意：系统会记住对话历史，支持上下文连续对话。
"""
        print(help_text)

def main():
    """主函数"""
    print("🤖 多轮对话系统启动中...")
    
    client = MultiTurnChatClient()
    
    # 测试连接
    print(f"正在连接服务器: {client.base_url}")
    try:
        test_response = client.session.get(client.base_url.replace('/chat/completions', '/admin/'), timeout=5)
        print("✅ 服务器连接正常")
    except:
        print("⚠️  警告: 无法连接到服务器，请确认Django服务是否启动")
    
    print("\n欢迎使用多轮对话系统!")
    print("输入 '/help' 查看帮助，输入 '/exit' 退出程序, 输入/set <config_key> <value> 设置配置项, 输入/config 查看当前配置")
    print("-" * 50)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 您: ").strip()
            
            # 处理特殊命令
            if user_input.lower() in ['/exit', '/quit']:
                print("👋 再见!")
                break
            elif user_input.lower() == '/help':
                client.show_help()
                continue
            elif user_input.lower() == '/clear':
                client.clear_history()
                continue
            elif user_input.lower() == '/history':
                client.show_history()
                continue
            elif user_input.lower().startswith('/set '):
                parts = user_input.split(maxsplit=2)
                if len(parts) == 3:
                    key, value = parts[1], parts[2]
                    set_config(key, value)
                    print(f"✅ 配置项 '{key}' 已设置为 '{value}'")
                else:
                    print("❌ 错误: 请使用 '/set <config_key> <value>' 格式设置配置项")
                continue
            elif user_input.lower() == '/config':
                current_config = config()
                print("\n=== 当前配置 ===")
                for key, value in current_config.__dict__.items():
                    print(f"{key}: {value}")
                print("===      ===\n")
                continue
            elif user_input == '':
                print("请输入您的问题，或输入 '/help' 查看帮助。")
                continue
            
            # 发送消息并获取回复
            print("🤖 助手: ", end="", flush=True)
            response = client.send_message(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 程序已中断，再见!")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            print("请重试或输入 '/exit' 退出程序")

if __name__ == "__main__":
    main()
