import os
project_dir = os.getcwd()

# LightRAG configuration
lightrag_llm_model = 'doubao-1-5-lite-32k-250115'
lightrag_embedding_model = 'doubao-embedding-text-240715'

lightrag_llm_url = 'https://ark.cn-beijing.volces.com/api/v3/'
lightrag_llm_key = ''

# The model selection for LightRAG is hardcoded in the code as gpt-4o-mini, may be changed later.
lightrag_service_port = 5001
lightrag_service_url = f"http://127.0.0.1:{lightrag_service_port}/receive_string"

# lightrag_working_dir 为生成的rag实例的存储Entities, Relationships, Document Chunks数据的目录
# 如果要创建新的rag实例，请确保 lightrag_working_dir 目录为空，否则程序会在已有的rag实例的基础上创建。
lightrag_working_dir = 'africa2024_2_database' 

# lightrag_knowledge_base_dir 为创建新的rag实例时，读取的知识库文件所在的目录
lightrag_knowledge_base_dir = 'africa2024_2_raw_files'

# 在使用LightRAG-webui时需要用到. 对openai的embedding模型：1536. 对doubao的embedding模型：2560.
embedding_dim = 2560
# Web scraping configuration

webscrap_enable_save= True  # whether to save the scraped content
webscrap_base_dir = 'ooo'


# Django configuration

django_llm_url = "https://api.chatanywhere.tech/v1"    
django_llm_key = ""
django_vllm_url = "http://localhost:30000/v1/completions"
django_model ="deepseek-chat"
django_service_port = '8006'
django_service_url = f"http://127.0.0.1:{django_service_port}/api/v1/chat/completions"

# DeepResearch config
dr_model = "qwen3-235b-a22b-instruct-2507"
# dr_llm_url&key is django's
