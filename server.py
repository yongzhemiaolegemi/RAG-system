from flask import Flask, request, jsonify
import asyncio
from lightrag import QueryParam
from utils import config
from wlc_demo import initialize_rag
import torch
from modelscope import AutoModelForSequenceClassification, AutoTokenizer
import threading
import argparse

lightrag_app = Flask(__name__)

# 获取事件循环
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

rag = loop.run_until_complete(initialize_rag())


@lightrag_app.route('/receive_string', methods=['POST'])
def receive_string():
    data = request.get_json()
    
    if 'message' not in data:
        return jsonify({"error": "No 'message' field provided"}), 400
    
    if 'mode' not in data:
        return jsonify({"error": "No 'mode' field provided"}), 400
    
    received_string = data['message']
    request_mode = data['mode']
    user_prompt = data.get('user_prompt', '')
    deep_research = data.get('deep_research', False)
    
    print(f"Received mode: {request_mode}")
    print(f"Is deep_research: {deep_research}")
    print(f"Received message: {received_string}")

    final_result, log_file_path = loop.run_until_complete(rag.aquery(
        received_string, param=QueryParam(
            mode=request_mode, 
            deep_research=deep_research,
            user_prompt=user_prompt
        )
    ))
    print(f"Final result: {final_result}")
    
    return jsonify({
        "message": f"Received string: {received_string}", 
        "mode": request_mode, 
        "result": final_result,
        "log_file_path": log_file_path
    })


rerank_app = Flask(__name__)

# 预加载rerank模型和tokenizer，避免每次请求都加载
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rerank_tokenizer = AutoTokenizer.from_pretrained(config().rerank_model)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(config().rerank_model)
    rerank_model = rerank_model.to(device)
    print(f'rerank model on device: {device}')
    rerank_model.eval()
except Exception as e:
    print(f"Failed to load rerank model: {e}")
    rerank_tokenizer = None
    rerank_model = None


@rerank_app.route('/rerank', methods=['POST'])
def rerank():
    if not rerank_tokenizer or not rerank_model:
        return jsonify({"error": "Rerank model not initialized"}), 500
        
    data = request.get_json()
    
    # 验证输入
    if 'query' not in data or 'documents' not in data:
        return jsonify({"error": "Missing 'query' or 'documents' field"}), 400
    
    query = data['query']
    docs = data['documents']
    
    # 创建查询-文档对
    pairs = [[query, doc] for doc in docs]
    print('reranking ...')
    
    try:
        with torch.no_grad():
            inputs = rerank_tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=2048
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            scores = rerank_model(** inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.cpu()
            # print(f"Rerank scores: {scores.tolist()}")
            # 将文档和分数组合成元组列表
            # 同时记录原始索引、文档和分数
            combined = list(zip(range(len(docs)), docs, scores))
            # 按照分数降序排序（排序时不影响原始索引的记录）
            combined_sorted = sorted(combined, key=lambda x: x[2], reverse=True)
            # 提取排序后的文档列表，包含原始索引
            reranked_docs = [{
                'index': idx,          # 原始列表中的索引
                'document':{'text': doc},
                'relevance_score': score.item()
            } for idx, doc, score in combined_sorted]
        return jsonify({
            "results": reranked_docs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
def run_rag_server():
    from config import lightrag_service_port
    lightrag_app.run(host='0.0.0.0', port=lightrag_service_port, debug=True, use_reloader=False)

def run_rerank_server():
    from config import rerank_service_port 
    rerank_app.run(host='0.0.0.0', port=rerank_service_port, debug=True, use_reloader=False)

if __name__ == '__main__':
    threads = []

    rag_thread = threading.Thread(target=run_rag_server)
    threads.append(rag_thread)
    rag_thread.start()

    if hasattr(config, 'rerank_service_port'): # 说明配置了本地rerank模型，要启用rerank服务
        rerank_thread = threading.Thread(target=run_rerank_server)
        threads.append(rerank_thread)
        rerank_thread.start()

    for thread in threads:
        thread.join()
