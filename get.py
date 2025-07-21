import requests
from config import lightrag_service_port
# 目标 URL（因为服务器运行在同一台设备，用 localhost）
url = f"http://localhost:{lightrag_service_port}/receive_string"

# 要发送的字符串
message = "描述一下Scrooge的人物关系"

# 发送 POST 请求（JSON 格式）
response = requests.post(
    url,
    json={"message": message
           , "mode": "naive"},  # 可以修改 mode 为 'local', 'global', 'hybrid' 等
    headers={"Content-Type": "application/json"}
)

print("Status Code:", response.status_code)
print("Response:", response.json()['result'])