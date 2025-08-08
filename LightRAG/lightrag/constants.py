"""
Centralized configuration constants for LightRAG.

This module defines default values for configuration constants used across
different parts of the LightRAG system. Centralizing these values ensures
consistency and makes maintenance easier.
"""

# Default values for environment variables
DEFAULT_MAX_GLEANING = 1
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 4
DEFAULT_WOKERS = 2
DEFAULT_TIMEOUT = 150

# Query and retrieval configuration defaults

# 默认查询时的top_k值
DEFAULT_TOP_K = 20

# 每次查询时用到的entity个数
DEFAULT_ENTITY_TOP_K = 20
# 经过rerank后，最终保留的entity个数 
DEFAULT_ENTITY_RERANK_TOP_K = 5

# 每次查询时用到的ralation个数
DEFAULT_RELATION_TOP_K = 20
# 经过rerank后，最终保留的relation个数 
DEFAULT_RELATION_RERANK_TOP_K = 5

# 每次naive模式查询时最初返回的document chunk个数
DEFAULT_CHUNK_TOP_K = 20
# 经过rerank后，最终保留的document chunk个数 
DEFAULT_CHUNK_RERANK_TOP_K = 5
'''
在mix模式下，dc 有两个来源:
1. 通过naive模式（传统的rag），查询出来的dc
2. 先通过hybrid模式获取entity和relation，再用获取到的这些entity和relation找到相关的dc
'''


DEFAULT_MAX_ENTITY_TOKENS = 5000
DEFAULT_MAX_RELATION_TOKENS = 5000
DEFAULT_MAX_TOTAL_TOKENS = 15000
DEFAULT_HISTORY_TURNS = 0
DEFAULT_ENABLE_RERANK = True
DEFAULT_COSINE_THRESHOLD = 0.2
DEFAULT_RELATED_CHUNK_NUMBER = 10

# Separator for graph fields
GRAPH_FIELD_SEP = "<SEP>"

# Logging configuration defaults
DEFAULT_LOG_MAX_BYTES = 10485760  # Default 10MB
DEFAULT_LOG_BACKUP_COUNT = 5  # Default 5 backups
DEFAULT_LOG_FILENAME = "lightrag.log"  # Default log filename
