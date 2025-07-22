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

# run request
python get.py
```

### Start a LightRAG GUI demo

首先确保已经跑过上面的**Start a demo**部分，这样确保在工作目录（对demo来说，就是`dickens/`）下已经有保存好的rag文件。然后执行：

```bash
lightrag-server 
```
然后打开启动好的url地址（一般是http://0.0.0.0:9621），即可在“知识图谱”中看到可视化的知识图谱。

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
