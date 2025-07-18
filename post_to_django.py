import requests
import json
from config import django_service_url, django_model


url = django_service_url
headers = {"Content-Type": "application/json"}
payload = {
    "model": django_model,
    "prompt": "描述一下Scrooge的人物关系，你可以使用RAG系统进行查询。并且告诉我你是否成功用RAG进行了查询。",
    "max_tokens": 2000,
    "temperature": 0.85,
    "stream": True,
    "collect_stream": True,
    "enable_tools": True
}

with requests.post(url, headers=headers, data=json.dumps(payload), stream=True) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if line:
            # 假设返回的每行都是合法的 JSON 片段
            chunk = json.loads(line)
            print(chunk, end="", flush=True)
