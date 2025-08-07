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



## 查询模式说明

| 模式        | 说明                                                         | query_context包含的数据 | 对查询结果有影响的超参数                                     |
| ----------- | ------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------ |
| `naive`     | 用传统的方法从RAG中检索出DC                                  | DC                      | `chunk_(rerank_)top_k`                                       |
| `local`     | 用low level关键词对RAG进行检索，得到一些E；<br />再通过这些E找到一些相关的R；<br />再通过E,R得到一些相关的DC | E, R, DC                | `entity_(rerank_)top_k`, `relation_(rerank_)top_k, ` `chunk_(rerank_)_top_k` |
| `global`    | 用high level关键词对RAG进行检索，得到一些R；<br />再通过这些R找到一些相关的E；<br />再通过E,R得到一些相关的DC | E, R, DC                | `entity_(rerank_)top_k`, `relation_(rerank_)top_k, ` `chunk_(rerank_)_top_k` |
| `hybrid`    | 用`local`和`global`模式得到结果，再把结果合并到一起。<br />(如果启用了rerank，则再取top_k；否则不再过滤，此时检索得到的结果数量上限为`2*entity/relation/chunk_top_k`) | E, R, DC                | `entity_(rerank_)top_k`, `relation_(rerank_)top_k, ` `chunk_(rerank_)_top_k` |
| `mix`       | 用`hybrid`和`naive`模式得到结果，再把结果合并到一起。<br />(如果启用了rerank，则再取top_k；否则不再过滤，此时检索得到的结果数量上限为`3*entity/relation/chunk_top_k`) | E, R, DC                | `entity_(rerank_)top_k`, `relation_(rerank_)top_k, ` `chunk_(rerank_)_top_k` |
| `hybrid_dc` | `hybrid`模式的结果去掉E,R。只保留DC。                        | DC                      | `entity_(rerank_)top_k`, `relation_(rerank_)top_k, ` `chunk_(rerank_)_top_k` |
| `mix_dc`    | `mix`模式的结果去掉E,R。只保留DC                             | DC                      | `entity_(rerank_)top_k`, `relation_(rerank_)top_k, ` `chunk_(rerank_)_top_k` |



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


### rerank相关参数说明

下列参数可以在`LightRAG/lightrag/constants.py`中设定。

> `entity_top_k`对应文件中的`DEFAULT_ENTITY_TOP_K`，其他的类似

- `entity_top_k`：初始检索返回的E数量上限
- `relation_top_k`：初始检索返回的R数量上限
- `chunk_top_k`：初始检索返回的DC数量上限

- `entity_rerank_top_k`：经过rerank后保留的E数量上限
- `relation_rerank_top_k`：经过rerank后保留的R数量上限
- `chunk_rerank_top_k`：经过rerank后保留的DC数量上限

如果启用rerank（即设置`DEFAULT_ENABLE_RERANK = True`），则：

- 最终的query_log的E,R,DC数量上限由`{entity/relation/chunk}_rerank_top_k`决定

- 最终的query_log的E,R,DC数量上限由`{entity/relation/chunk}_rerank_top_k`决定

如果不启用rerank（即设置`DEFAULT_ENABLE_RERANK = False`），则：

- 最终的query_log的E,R,DC数量上限由`{entity/relation/chunk}_top_k`决定
- `{entity/relation/chunk}_rerank_top_k`参数完全不会用到。

> 此外，还有一个参数`top_k`：一个原LightRAG库提供的默认top_k值。如果查询运行时，发现某一个top_k的值异常（比如为0或者未设定），就有可能会用top_k来代替。只是记录一下，在本项目里算是deprecated。



