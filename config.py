import os
project_dir = os.getcwd()

# LightRAG configuration
lightrag_llm_model = 'gpt-4.1-mini'
lightrag_llm_url = 'https://api.deerapi.com/v1'
lightrag_llm_key = ""

# The model selection for LightRAG is hardcoded in the code as gpt-4o-mini, may be changed later.
lightrag_service_port = 5001
lightrag_service_url = f"http://127.0.0.1:{lightrag_service_port}/receive_string"

# 如果要创建新的rag实例，请确保 lightrag_working_dir 目录为空，否则程序会误认为 lightrag_working_dir 中已有rag实例，而不会重新创建。
lightrag_working_dir = 'dickens_new' # 这一项的值要和 LightRAG/.env 中的 WORKING_DIR 一致！

# lightrag_knowledge_base_files为创建新的rag实例时，读取的知识库以及文件路径（以dict形式存储）
lightrag_knowledge_base_files = {
    # 书名: 文件路径,
    'A Christmas Carol': 'book.txt', 
}

# Web scraping configuration

webscrap_enable_save= True  # whether to save the scraped content
webscrap_base_dir = 'ooo'


# Django configuration

django_llm_url = "https://api.deepseek.com/v1"    
django_llm_key = ""
django_vllm_url = "http://localhost:30000/v1/completions"
django_model =""
django_service_port = '8006'
django_service_url = f"http://127.0.0.1:{django_service_port}/api/v1/chat/completions"
