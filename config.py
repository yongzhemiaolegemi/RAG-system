import os
project_dir = os.getcwd()

# LightRAG configuration

lightrag_llm_url = 'https://api.deerapi.com/v1'
lightrag_llm_key = ""
# The model selection for LightRAG is hardcoded in the code as gpt-4o-mini, may be changed later.
lightrag_service_port = 5001
lightrag_service_url = f"http://127.0.0.1:{lightrag_service_port}/receive_string"
lightrag_working_dir = "dickens"
lightrag_knowledge_base_file = "book.txt"

# Web scraping configuration

webscrap_enable_save= True  # whether to save the scraped content
webscrap_base_dir = "webcontent"


# Django configuration

django_llm_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"    # 阿里云百炼
django_llm_key = ""
django_vllm_url = "http://localhost:30000/v1/completions"
django_model = "qwq-plus-latest"
django_service_port = 8006
django_service_url = f"http://127.0.0.1:{django_service_port}/api/v1/chat/completions"
