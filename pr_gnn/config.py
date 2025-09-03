# pr_gnn/config.py
import yaml
import os
from pathlib import Path

_config = None

def load_config():
    """加载并返回配置字典"""
    global _config
    if _config is None:
        config_path = Path(__file__).parent / 'config' / 'default_config.yaml'
        with open(config_path) as f:
            _config = yaml.safe_load(f) or {}
        
        # 设置默认值
        _config.setdefault('C1', 1.5)
        _config.setdefault('C2', 0.01)
        _config.setdefault('C3', 0.3)
        _config.setdefault('C4', 0.005)
        _config.setdefault('C5', 0.5)
        _config.setdefault('C6', 0.001)
        _config.setdefault('C7', 0.1)
        _config.setdefault('gamma', 1.4)
        _config.setdefault('free_stream', [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    return _config

def get_config_value(key, default=None):
    """获取指定配置项的值"""
    config = load_config()
    return config.get(key, default)
