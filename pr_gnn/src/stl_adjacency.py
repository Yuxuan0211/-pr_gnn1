import numpy as np
import trimesh
import sys
import time
import os
from joblib import Parallel, delayed

def verify_same_dir_file(file_name):
    """验证文件是否在脚本当前目录，返回完整相对路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"❌ 文件 {file_name} 未找到！\n"
            f"   请将 {file_name} 放在脚本所在目录：{script_dir}"
        )
    return file_path

def fast_read_stl(stl_file_name="export.stl"):
    """读取同目录下的export.stl，提取三角形索引和顶点数"""
    start_time = time.time()
    try:
        stl_path = verify_same_dir_file(stl_file_name)
        mesh = trimesh.load(
            stl_path,
            force='mesh',
            skip_materials=True,
            skip_textures=True,
            validate=False
        )
        faces = mesh.faces.astype(np.uint32)
        n_nodes = int(faces.max()) + 1 if faces.size > 0 else 0
        cost_time = time.time() - start_time
        print(f"✅ STL读取完成（同目录）")
        print(f"   耗时：{cost_time:.2f}s | 顶点数：{n_nodes:,} | 三角形数：{len(faces):,}")
        return faces, n_nodes
    except Exception as e:
        raise RuntimeError(f"❌ STL读取失败：{str(e)}") from e

def generate_edges(faces):
    """向量化生成所有边（单线程，无并行开销）"""
    start_time = time.time()
    if faces.size == 0:
        return np.array([], dtype=np.uint32).reshape(0, 2)
    all_edges = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], axis=0)
    all_edges_sorted = np.sort(all_edges, axis=1)
    cost_time = time.time() - start_time
    print(f"✅ 边生成完成")
    print(f"   耗时：{cost_time:.2f}s | 总边数：{len(all_edges_sorted):,}")
    return all_edges_sorted

def parallel_deduplicate_edges(edges, n_jobs=-1):
    """多核并行去重边（核心加速步骤）"""
    start_time = time.time()
    n_total = len(edges)
    if n_total < 100000:
        unique_edges = np.unique(edges, axis=0)
        cost_time = time.time() - start_time
        print(f"✅ 边去重完成（单线程）")
        print(f"   耗时：{cost_time:.2f}s | 去重后边数：{len(unique_edges):,}")
        return unique_edges
    
    n_cores = os.cpu_count() if n_jobs == -1 else min(n_jobs, os.cpu_count())
    print(f"✅ 启动多核去重（{n_cores}个核心）")
    
    chunk_size = n_total // n_cores
    edge_chunks = [edges[i*chunk_size : (i+1)*chunk_size] for i in range(n_cores)]
    if n_total % n_cores != 0:
        edge_chunks[-1] = np.concatenate([edge_chunks[-1], edges[n_cores*chunk_size:]], axis=0)
    
    def deduplicate_chunk(chunk):
        return np.unique(chunk, axis=0)
    
    parallel_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(deduplicate_chunk)(chunk) for chunk in edge_chunks
    )
    
    # 合并分片并全局去重
    merged_edges = np.concatenate(parallel_results, axis=0)
    unique_edges = np.unique(merged_edges, axis=0)
    
    cost_time = time.time() - start_time
    print(f"✅ 边去重完成（多核）")
    print(f"   耗时：{cost_time:.2f}s | 去重后边数：{len(unique_edges):,}")
    return unique_edges

def save_csv_to_same_dir(unique_edges, csv_file_name="sparse_adj_export.csv", save_symmetric=False):
    """将稀疏CSV保存到脚本同目录"""
    start_time = time.time()
    n_edges = len(unique_edges)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_file_name)
    
    # 处理空边情况
    if n_edges == 0:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("row_index,col_index,value\n")
        print(f"✅ 空CSV已保存（同目录）")
        print(f"   路径：{csv_path}")
        return
    
    # 构建CSV数据（向量化生成，提速）
    value_col = np.ones((n_edges, 1), dtype=np.uint8)
    csv_data = np.hstack([unique_edges, value_col])
    
    # 按需生成对称边（如(0,1)和(1,0)）
    if save_symmetric:
        symmetric_edges = np.hstack([unique_edges[:, [1, 0]], value_col])
        csv_data = np.concatenate([csv_data, symmetric_edges], axis=0)
    
    # 快速写入CSV（np.savetxt比Pandas快2~3倍）
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("row_index,col_index,value\n")
        np.savetxt(f, csv_data, fmt='%d', delimiter=',')
    
    # 计算文件大小（精确值，而非估算）
    file_size = os.path.getsize(csv_path) / 1024 / 1024  # 转为MB
    cost_time = time.time() - start_time
    print(f"✅ CSV保存完成（同目录）")
    print(f"   耗时：{cost_time:.2f}s | 路径：{csv_path} | 大小：{file_size:.2f}MB")

def main():
    print("="*50)
    print("📌 STL转稀疏CSV（同目录版，多核加速）")
    print("="*50)
    total_start = time.time()
    
    try:
        # 1. 读取同目录export.stl
        faces, n_nodes = fast_read_stl()
        if n_nodes == 0:
            print("❌ STL文件无顶点数据，退出")
            return
        
        # 2. 生成所有边
        all_edges = generate_edges(faces)
        
        # 3. 多核并行去重边
        unique_edges = parallel_deduplicate_edges(all_edges, n_jobs=-1)
        
        # 4. 保存CSV到同目录
        save_csv_to_same_dir(unique_edges, csv_file_name="sparse_adj_export.csv")
        
        # 总耗时统计
        total_cost = time.time() - total_start
        print("="*50)
        print(f"🎉 转换完成！总耗时：{total_cost:.2f}s")
        print(f"   👉 STL源文件：同目录/export.stl")
        print(f"   👉 CSV结果文件：同目录/sparse_adj_export.csv")
        print("="*50)
    
    except Exception as e:
        print(f"\n❌ 转换失败：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 已删除np.set_num_threads(1)，适配旧版本NumPy
    main()
    