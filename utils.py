# config_loader.py
import sys
from types import SimpleNamespace

_cfg = None  # 全局单例

def load_config():
    """加载或重新加载 config.py"""
    global _cfg
    config_vars = {}
    with open('config.py', 'r', encoding='utf-8') as f:
        exec(f.read(), {}, config_vars)
    _cfg = SimpleNamespace(**{
        k: v for k, v in config_vars.items()
        if not k.startswith('__')
    })

def config():
    """获取全局单例 config"""

    load_config()
    return _cfg

def set_config(key: str, value):
    """修改配置参数"""
    if not hasattr(_cfg, key):
        raise AttributeError(f"Config has no attribute '{key}'")
    setattr(_cfg, key, value)
    # save to config.py
    with open('config.py', 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if line.startswith(key):
                f.write(f"{key} = {repr(value)}\n")
            else:
                f.write(line)
        f.truncate()

# 初始化
load_config()