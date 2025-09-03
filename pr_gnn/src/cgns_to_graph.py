# src/cgns_to_graph.py
import numpy as np
import pandas as pd
import h5py
import sys
from typing import Tuple

def read_cgns_elements(filename: str) -> Tuple[np.ndarray, int]:
    with h5py.File(filename, 'r') as f:
        try:
            elements = np.array(f['Base/Zone/GridElements/ElementConnectivity']) - 1
            n_nodes = int(f['Base/Zone/GridCoordinates/CoordinateX/DataArray'][:].shape[0])
            elem_type = np.array(f['Base/Zone/GridElements/ElementType'])[0]
            print(f"Detected element type: {elem_type}, Number of nodes: {n_nodes}")
            return elements.reshape(-1, 4), n_nodes
        except KeyError as e:
            print("CGNS结构未按预期解析，请检查路径。")
            raise e

def build_adjacency_from_elements(elements: np.ndarray, num_nodes: int) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.int8)
    for elem in elements:
        for i in range(len(elem)):
            for j in range(i + 1, len(elem)):
                u, v = elem[i], elem[j]
                adj[u, v] = adj[v, u] = 1
    return adj

def save_adjacency_csv(adj_matrix: np.ndarray, output_path: str):
    df = pd.DataFrame(adj_matrix)
    df.to_csv(output_path, index=False, header=False)
    print(f"邻接矩阵已保存至: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert CGNS mesh to adjacency matrix CSV")
    parser.add_argument("--input", type=str, required=True, help="Path to .cgns file")
    parser.add_argument("--output", type=str, default="data/processed/adjacency.csv", help="Output CSV path")
    args = parser.parse_args()

    elements, n_nodes = read_cgns_elements(args.input)
    adj = build_adjacency_from_elements(elements, n_nodes)
    save_adjacency_csv(adj, args.output)
