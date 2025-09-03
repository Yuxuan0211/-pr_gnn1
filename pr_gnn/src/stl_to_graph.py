# src/stl_to_graph.py
import numpy as np
import pandas as pd
import stl
import os
from typing import Tuple

def read_stl_vertices_and_faces(stl_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 STL 文件中读取顶点和面片数据。
    支持 ASCII 和 Binary 格式。
    """
    try:
        mesh = stl.Mesh.from_file(stl_file)
        vertices = mesh.vertices
        faces = mesh.faces
        return vertices, faces
    except Exception as e:
        print(f"❌ 无法读取 STL 文件: {e}")
        raise

def build_adjacency_from_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    根据面片构建邻接矩阵（仅考虑共享边的顶点）。
    """
    # 去重顶点，分配唯一索引
    unique_vertices, indices = np.unique(vertices, axis=0, return_inverse=True)
    num_vertices = len(unique_vertices)
    adj = np.zeros((num_vertices, num_vertices), dtype=np.int8)

    # 遍历每个面片，建立邻接关系
    for face in faces:
        # 每个面片由三个顶点组成
        v0, v1, v2 = face[0], face[1], face[2]
        # 获取顶点在去重后的索引
        idx0 = indices[np.where(np.all(vertices == v0, axis=1))[0][0]]
        idx1 = indices[np.where(np.all(vertices == v1, axis=1))[0][0]]
        idx2 = indices[np.where(np.all(vertices == v2, axis=1))[0][0]]

        # 建立邻接关系（每个顶点与另外两个顶点相连）
        adj[idx0, idx1] = adj[idx1, idx0] = 1
        adj[idx0, idx2] = adj[idx2, idx0] = 1
        adj[idx1, idx2] = adj[idx2, idx1] = 1

    return adj, unique_vertices

def save_adjacency_csv(adj_matrix: np.ndarray, output_path: str):
    df = pd.DataFrame(adj_matrix)
    df.to_csv(output_path, index=False, header=False)
    print(f"✅ 邻接矩阵已保存至: {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert STL mesh to adjacency matrix CSV")
    parser.add_argument("--input", type=str, required=True, help="Path to .stl file")
    parser.add_argument("--output", type=str, default="data/processed/adjacency.csv", help="Output CSV path")
    args = parser.parse_args()

    vertices, faces = read_stl_vertices_and_faces(args.input)
    adj_matrix, _ = build_adjacency_from_faces(vertices, faces)
    save_adjacency_csv(adj_matrix, args.output)

if __name__ == "__main__":
    main()
