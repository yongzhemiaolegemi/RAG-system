import requests
import argparse

from utils import config

def get_lightrag():
    """
    发送 POST 请求到 LightRAG 服务，获取响应结果。
    """
    # 导入配置文件中的 lightrag_service_port
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

def get_rerank():
    """
    发送 POST 请求到 LightRAG 的 rerank 服务，获取响应结果。
    """ 

    # 要发送的查询和文档
    
    query = "英国的首都是哪里?"  # 我试了，query和docs的语种可以不同
    docs = [
        "The capital of France is Paris.",
        "Tokyo is the capital of Japan.",
        "London is the capital of England.",
    ]

    api_key = getattr(config(), "rerank_api_key", "invalid")

    response = requests.post(
        config().rerank_service_url,
        json={"model": config().rerank_model, "query": query, "documents": docs},
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    )

    print("Status Code:", response.status_code)
    print("Response:", response.json())

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='选择要调用的服务')
    # 添加一个位置参数，指定要运行的函数
    parser.add_argument('service', choices=['lightrag', 'rerank'], default='lightrag',
                        help='指定要调用的服务: lightrag 或 rerank')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据参数调用对应的函数
    if args.service == 'lightrag':
        get_lightrag()
    elif args.service == 'rerank':
        get_rerank()
