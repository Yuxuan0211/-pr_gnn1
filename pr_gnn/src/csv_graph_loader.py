import numpy as np
import pandas as pd

def load_adjacency_from_csv(filepath):
    """Load adjacency matrix from CSV file with improved parsing"""
    try:
        # Try reading with flexible delimiter and no header
        df = pd.read_csv(filepath, header=None, sep=None, engine='python')
        # Convert to numpy array and ensure numeric values
        return df.astype(float).values
    except Exception as e:
        raise ValueError(f"Failed to parse CSV file {filepath}: {str(e)}")

def save_adjacency_csv(adj_matrix, output_path):
    """Save adjacency matrix to CSV file"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(adj_matrix)
    df.to_csv(output_path, header=False, index=False)
