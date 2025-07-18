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

### Run a demo

```bash
python post_to_django.py
```

> **WARNING**  
> Before running **post_to_django.py**, please make sure you have modified **config.py** and that both the **LightRAG server CLI** and the **Django server** are already running.