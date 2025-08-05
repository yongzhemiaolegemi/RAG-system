# An implementation of RAG System

## Quick Start (LightRAG)

### Easy Installation

```bash
# clone the repository
git clone https://github.com/yongzhemiaolegemi/RAG-system.git

# create virtual env
conda create -n rag python==3.11
conda activate rag
cd LightRAG
pip install -e .
cd ..
```
### Start a demo

```bash
python wlc_demo.py
```
### Start a Server cli

```bash
# run server
python server.py

# run a simple lightrag request
python get.py lightrag
```


## 配置rerank功能

rerank做的事情：是对所有检索到的文本进行相关性排序并选择top K。

> 没有配置rerank也可以跑通。配置了会让rag查询返回的结果更好。

下面两种方式**二选一**即可：调用api / 用本地的rerank模型

### 1. 加载本地rerank模型

```python
# pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(model_id="BAAI/bge-reranker-v2-m3") # 会下载到$MODELSCOPE_CACHE/models
```

`BAAI/bge-reranker-v2-m3`是北京智源研究院提出的rerank模型（是lightrag官方的默认模型）。大概会占用2G显存，感觉可以接受。

然后在`config.py`中进行配置：

```python
# 用本地的rerank模型
rerank_model = "BAAI/bge-reranker-v2-m3"
rerank_service_port = 5002
rerank_service_url = f"http://127.0.0.1:{rerank_service_port}/rerank"
```

### 2. 调用api

- rerank模型不属于常见的生成式llm，所以大部分LLM服务商没有提供这类模型的api。
- 有一家叫做[jina.ai](https://jina.ai/)的公司有提供这类api，是可以用的。不过它貌似只提供了一部分免费的token，如果长期使用需要用国外的visa卡充值。
- jina的api只提供它们自己家的模型。可以用`jina-reranker-v2-base-multilingual`。

然后在`config.py`中进行配置：

```python
# 调用api
rerank_model="jina-reranker-v2-base-multilingual"
rerank_service_url="https://api.jina.ai/v1/rerank"
rerank_api_key="" # 自行填充
```






### run a simple rerank request
```bash
# run server
python server.py # 如果是调用api，则无需执行。如果是加载本地rerank模型，则需要执行。

# run a simple rerank request
python get.py rerank
```


## Start a LightRAG GUI demo

首先确保已经跑过上面的**Start a demo**部分，这样确保在工作目录（对demo来说，就是`dickens/`）下已经有保存好的rag文件。然后执行：

```bash
lightrag-server 
```
然后打开启动好的url地址（一般是`http://0.0.0.0:9621`），即可在“知识图谱”中看到可视化的知识图谱。

（“检索”功能貌似还有点问题）。

## Quick Start (Django)

### Installation

```bash
conda create -n django python=3.11
conda activate django
cd backend
pip install -r requirements.txt
cd ..
```

### Start Django server

```bash
bash start_django_backend.sh
```

### Run a command-line program

```bash
python multiturn_client.py
```

> **WARNING**  
> Before running **multiturn_client.py**, please make sure you have modified **config.py** and that both the **LightRAG server CLI** and the **Django server** are already running.



