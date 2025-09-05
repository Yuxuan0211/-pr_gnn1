import numpy as np
import pandas as pd

def load_adjacency_from_csv(filepath):
    """Load adjacency matrix from CSV file with improved parsing"""
    try:
        # First try reading with header detection
        df = pd.read_csv(filepath, sep=None, engine='python')
        
        # Check for non-numeric columns and drop them
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < len(df.columns):
            df = df[numeric_cols]
            print(f"⚠️  检测到非数值列，已自动忽略")
            
        # If no numeric columns found, try reading without header
        if len(numeric_cols) == 0:
            df = pd.read_csv(filepath, header=None, sep=None, engine='python')
            return df.astype(float).values
            
        return df.values.astype(float)
    except Exception as e:
        raise ValueError(f"CSV文件解析失败: {filepath}\n错误详情: {str(e)}\n请确保文件包含有效的邻接矩阵数据")

def save_adjacency_csv(adj_matrix, output_path):
    """Save adjacency matrix to CSV file"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(adj_matrix)
    df.to_csv(output_path, header=False, index=False)
