from flask import Flask, request, jsonify
from config import lightrag_service_port
import asyncio
from lightrag import  QueryParam
from wlc_demo import initialize_rag, run_demo
app = Flask(__name__)

# Get an event loop
try:
    loop = asyncio.get_running_loop()
except RuntimeError:  # 'RuntimeError: There is no current event loop...'
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

rag = loop.run_until_complete(initialize_rag())


@app.route('/receive_string', methods=['POST'])
def receive_string():
    # 获取客户端发送的数据
    data = request.get_json()
    
    if 'message' not in data:
        return jsonify({"error": "No 'message' field provided"}), 400
    
    if 'mode' not in data:
        return jsonify({"error": "No 'mode' field provided"}), 400
    
    received_string = data['message']
    request_mode = data['mode']
    deep_research = data['deep_research'] if 'deep_research' in data else False
    print(f"Received mode: {request_mode}")
    print(f"Is deep_research: {deep_research}")
    print(f"Received message: {received_string}")

    final_result, log_file_path = loop.run_until_complete(rag.aquery(
        received_string, param=QueryParam(mode=request_mode, deep_research=deep_research)
    ))
    print(f"Final result: {final_result}")
    
    # 返回响应
    return jsonify({
        "message": f"Received string: {received_string}", 
        "mode": request_mode, 
        "result": final_result,
        "log_file_path": log_file_path
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=lightrag_service_port, debug=True)