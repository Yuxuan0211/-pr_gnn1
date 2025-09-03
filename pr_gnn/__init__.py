# pr_gnn package initialization
import os
import sys
from pr_gnn.config import get_config_value

# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_config():
    """获取配置字典"""
    return get_config_value()

__all__ = ['get_config']
