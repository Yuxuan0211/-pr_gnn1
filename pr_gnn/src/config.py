# src/config.py
import yaml
import os
from pathlib import Path

# 获取配置文件绝对路径
config_path = Path(__file__).parent.parent / 'config' / 'default_config.yaml'

# 加载默认配置文件
with open(config_path) as f:
    CONFIG = yaml.safe_load(f) or {}

# 区域划分参数默认值（确保关键参数存在）
CONFIG.setdefault('C1', 1.5)  # 激波区阈值
CONFIG.setdefault('C2', 0.01) # 边界层区μt阈值
CONFIG.setdefault('C3', 0.3)  # 边界层区速度阈值
CONFIG.setdefault('C4', 0.005) # 尾流区μt阈值
CONFIG.setdefault('C5', 0.5)  # 尾流区Vx阈值
CONFIG.setdefault('C6', 0.001) # 无粘区/来流区μt阈值
CONFIG.setdefault('C7', 0.1)  # 来流区物理量偏差阈值
CONFIG.setdefault('gamma', 1.4) # 比热比
CONFIG.setdefault('free_stream', [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]) # 来流条件

# 导出CONFIG到模块全局变量
globals().update({'CONFIG': CONFIG})
