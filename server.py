from flask import Flask, request, jsonify
from wlc_demo import run_demo
from config import lightrag_service_port
app = Flask(__name__)

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
    print(f"Received mode: {request_mode}")
    print(f"Received message: {received_string}")
    result_list  = run_demo(received_string)
    modes = ['naive', 'local', 'global', 'hybrid']
    if request_mode not in modes:
        return jsonify({"error": f"Invalid mode. Choose from {modes}"}), 400
    final_result = result_list[modes.index(request_mode)]
    
    

    
    # 返回响应
    return jsonify({"message": f"Received string: {received_string}", "mode": request_mode, "result": final_result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=lightrag_service_port, debug=True)