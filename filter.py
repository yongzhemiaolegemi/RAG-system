from utils import config
from openai import OpenAI
from typing import Tuple

user_prompt = '''
严格判断以下文本是否满足给定条件，仅关注两种条件类型：
1. **日期范围**（如 "2023-2024"）：若文本中任何日期在范围内 → True
2. **关键词**（如 "人工智能"）：若文本出现关键词或其同义表述 → True

必须按此格式响应：
"Answer: True/False
Reason: [1句话解释]"

规则：
- 日期处理：
  - 模糊匹配（如 "2023" 匹配 "2023年3月"）
  - 忽略非内容日期（如页脚版权日期）
- 关键词处理：
  - 接受同义词和核心语义匹配
  - 大小写不敏感

待分析文本：
{txt}

待检查条件：
{condition}
'''

def filter_by_llm(txt: str, condition: str) -> Tuple[bool, str]:
    """Returns (judgment, reason) tuple without text extraction"""
    prompt = user_prompt.format(txt=txt, condition=condition)

    
    client = OpenAI(api_key=config().django_llm_key, base_url=config().django_llm_url)

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a binary content-checking assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
        extra_body={"enable_thinking": False},
    )
    
    return parse_llm_response(response.choices[0].message.content)

def parse_llm_response(response: str):
    """Extracts True/False and reason from response"""
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    
    judgment = False
    reason = "No reason provided"
    
    for line in lines:
        if line.startswith('Answer:'):
            judgment = 'True' in line
        elif line.startswith('Reason:'):
            reason = line.replace('Reason:', '').strip()
    
    return judgment, reason