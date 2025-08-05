import os
import asyncio
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.rerank import custom_rerank
from lightrag.utils import logger, set_verbose_debug
from functools import wraps
import time
from utils import config
from functools import partial
from types import SimpleNamespace

# grok-3
os.environ["OPENAI_API_KEY"] = config().lightrag_llm_key
os.environ["OPENAI_API_BASE"] = config().lightrag_llm_url

def timeit_decorator(func):
    """用于测量函数执行时间的装饰器"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"[TIMING] {func.__name__} 耗时: {end_time - start_time:.4f}秒")
        return result
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[TIMING] {func.__name__} 耗时: {end_time - start_time:.4f}秒")
        return result
     
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper



def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


async def my_rerank_func(query: str, documents: list, top_k: int = None, **kwargs):
    """Custom rerank function with all settings included"""
    return await custom_rerank(
        query=query,
        documents=documents,
        model=config().rerank_model,
        base_url=config().rerank_service_url,
        api_key=getattr(config(),'rerank_api_key','invalid'),
        # top_k=top_k or 10,  # Default top_k if not provided
        **kwargs,
    )


async def initialize_rag():
    WORKING_DIR = config().lightrag_working_dir
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=openai_complete,
        llm_model_name=config().lightrag_llm_model,
        rerank_model_func=my_rerank_func,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def process_directory(directory, rag):
    """递归处理目录中的所有文件和子目录"""
    # 遍历当前目录中的所有项目
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # 如果是目录，则递归处理
        if os.path.isdir(item_path):
            await process_directory(item_path, rag)
        
        # 如果是txt文件，则进行处理
        elif os.path.isfile(item_path) and item.endswith(".txt"):
            title = os.path.splitext(item)[0]
            print(f"Inserting knowledge base file: {item_path}")
            with open(item_path, "r", encoding="utf-8") as f:
                await rag.ainsert(f.read(), file_paths=title)

@timeit_decorator
async def main(input_str,mode='hybrid',deep_research=False):
    WORKING_DIR = config().lightrag_working_dir
    KNOWLEDGE_BASE_DIR = config().lightrag_knowledge_base_dir
    # Check if OPENAI_API_KEY environment variable exists
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
        )
        print("You can set the environment variable by running:")
        print("  export OPENAI_API_KEY='your-openai-api-key'")
        return  # Exit the async function

    try:
        # Clear old data files
        files_to_delete = [
            # "graph_chunk_entity_relation.graphml",
            # "kv_store_doc_status.json",
            # "kv_store_full_docs.json",
            # "kv_store_text_chunks.json",
            # "vdb_chunks.json",
            # "vdb_entities.json",
            # "vdb_relationships.json",
            'kv_store_llm_response_cache.json'
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]


        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        # Insert knowledge base files
        await process_directory(KNOWLEDGE_BASE_DIR, rag)
 

        result_list = []
        
        print("Query mode: ", mode)  # for m in ['naive', 'local', 'global', 'hybrid']: 

        str, log_file_path = await rag.aquery(
            input_str, param=QueryParam(mode=mode,deep_research=deep_research)
        )
        result_list.append(str)
        print(str)
        return result_list
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


def run_demo(str,mode='hybrid'):
    # Configure logging before running the main function
    configure_logging()
    print("\n============ Initializing RAG storage ============")
    res = asyncio.run(main(str,mode))
    print("\nDone!")
    return res

if __name__ == "__main__":
    configure_logging()
    print("\n============ Initializing RAG storage ============")
    res = asyncio.run(main("主要人物关系",deep_research=True,mode='naive'))
    print("\nDone! ")

