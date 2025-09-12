# src/dataset.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from sklearn.preprocessing import StandardScaler
import os
import hashlib
import time

class FlowDataset:
    def __init__(self, adj_csv: str, feature_csv: str, config):
        self.adj_csv = adj_csv
        self.feature_csv = feature_csv
        self.cache_dir = os.path.join('data', 'processed', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        # 提供默认配置
        self.config = {
            'training': {
                'optimizer': 'AdamW',
                'weight_decay': 0.01,
                'lr': 0.001,
                'min_lr': 1e-6,
                'lr_schedule': 'cosine_with_warmup',
                'warmup_epochs': 10,
                'cosine_epochs': 100,
                'mixed_precision': True,
                'neighbor_sampling': True,
                'num_neighbors': 25,
                'num_layers': 2,
                'early_stopping_patience': 20,
                'min_epochs': 50
            },
            'region_normalization': {
                'enabled': False,
                'num_regions': 5
            },
            'free_stream': {
                'V': 100.0,
                'P': 101325.0,
                'rho': 1.225,
                'h': 300000.0
            }
        }
        # 合并用户提供的配置
        if isinstance(config, dict):
            self.config.update(config)
        
        # 初始化scaler，设置handle_nan='raise'以捕获无效值
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def normalize_features_by_region(self, x, region_mask):
        """按区域归一化特征（每个区域单独计算均值和标准差）"""
        x_norm = x.clone()
        region_means = {}
        region_stds = {}
        
        for region_id in range(self.config['region_normalization']['num_regions']):
            region_mask_bool = (region_mask == region_id)
            if region_mask_bool.sum() == 0:
                continue
                
            # 计算该区域的均值和标准差
            region_x = x[region_mask_bool]
            mean = region_x.mean(dim=0, keepdim=True)
            std = region_x.std(dim=0, keepdim=True) + 1e-8  # 避免除零
            
            # 归一化
            x_norm[region_mask_bool] = (region_x - mean) / std
            region_means[region_id] = mean
            region_stds[region_id] = std
            
        return x_norm, region_means, region_stds

    def _get_cache_path(self) -> str:
        """生成缓存文件路径"""
        adj_hash = hashlib.md5(self.adj_csv.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{adj_hash}.pt")
    
    def _load_from_cache(self) -> tuple:
        """从缓存加载邻接矩阵和region_mask"""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return None
        
        # 检查缓存是否过期（源文件比缓存新）
        if os.path.getmtime(self.adj_csv) > os.path.getmtime(cache_path):
            return None
            
        try:
            cache_data = torch.load(cache_path)
            return cache_data['adj_sparse'], cache_data.get('region_mask', None)
        except:
            return None
    
    def _save_to_cache(self, adj_sparse: SparseTensor, region_mask: torch.Tensor = None):
        """保存邻接矩阵和region_mask到缓存"""
        cache_path = self._get_cache_path()
        cache_data = {
            'adj_sparse': adj_sparse,
            'region_mask': region_mask,
            'timestamp': time.time()
        }
        torch.save(cache_data, cache_path)
    
    def load_data(self, batch_size: int = 300) -> Data:
        # 尝试从缓存加载
        cache_result = self._load_from_cache()
        if cache_result is not None:
            adj_sparse, region_mask = cache_result
            edge_index = adj_sparse.to_torch_sparse_coo_tensor().coalesce().indices()
        else:
            # 读取邻接矩阵（支持多种分隔符）
            try:
                adj_df = pd.read_csv(self.adj_csv, header=None, sep=None, engine='python').values
            except Exception as e:
                raise ValueError(f"无法解析邻接矩阵CSV文件 {self.adj_csv}: {str(e)}")
            
            # 转换为SparseTensor并缓存
            edge_index = np.where(adj_df == 1)
            edge_index = torch.tensor(np.array([edge_index[0], edge_index[1]]), dtype=torch.long)
            adj_sparse = SparseTensor.from_edge_index(edge_index)
            self._save_to_cache(adj_sparse)

        # 处理分批加载
        if isinstance(self.feature_csv, list) and batch_size is not None:
            # 随机选择batch_size个流场
            selected_indices = np.random.choice(len(self.feature_csv), batch_size, replace=False)
            selected_files = [self.feature_csv[i] for i in selected_indices]
            self.feature_csv = selected_files

            # 多马赫数数据处理
            if isinstance(self.feature_csv, list):
                multi_mach_data = []
                for csv_file in self.feature_csv:
                    feat_df = pd.read_csv(
                        csv_file,
                        sep=None,
                        engine='python',
                        header='infer',
                        skip_blank_lines=True,
                        comment='#',
                        na_values=['NaN', 'nan', 'inf', '-inf']
                    ).apply(pd.to_numeric, errors='coerce')
                    
                    # 处理无效值
                    feat_df = feat_df.fillna(feat_df.mean()).replace([np.inf, -np.inf], np.nan).fillna(feat_df.mean())
                    if feat_df.isna().any().any():
                        raise ValueError(f"文件 {csv_file} 预处理后仍存在无效值")
                    
                    # 创建数据对象
                    q_inf = self.config['free_stream']
                    x_in = np.hstack([
                        feat_df.iloc[:, 1:4].values,
                        np.tile([q_inf['V'], q_inf['P'], q_inf['rho'], q_inf['h']], (len(feat_df), 1))
                    ])
                    data = Data(
                        x=torch.tensor(x_in, dtype=torch.float),
                        y=torch.tensor(feat_df.iloc[:, 5:15].values, dtype=torch.float),
                        edge_index=edge_index,
                        num_nodes=len(feat_df)
                    )
                    multi_mach_data.append(data)
            
            # 合并多马赫数数据（保持网格结构）
            if len(multi_mach_data) == 0:
                raise ValueError("未加载任何有效的马赫数数据")
                
            # 使用第一个数据作为基础
            base_data = multi_mach_data[0]
            if len(multi_mach_data) > 1:
                # 叠加其他马赫数数据
                base_data.multi_mach_y = torch.stack([d.y for d in multi_mach_data], dim=1)
                print(f"✅ 已加载 {len(multi_mach_data)} 个马赫数数据，网格节点数 {base_data.num_nodes}")
            return base_data, self.scaler_x, self.scaler_y
        else:
            # 单文件模式
            try:
                # 读取CSV文件
                feat_df = pd.read_csv(
                    self.feature_csv,
                    sep=None,
                    engine='python',
                    header='infer',
                    skip_blank_lines=True,
                    comment='#',
                    na_values=['NaN', 'nan', 'inf', '-inf']
                )
            
                # 数据验证和处理
                if feat_df.empty:
                    raise ValueError("CSV文件为空或没有有效数据")
            
                # 转换为数值并处理无效值
                feat_df = feat_df.apply(pd.to_numeric, errors='coerce')
                
                # 填充或删除无效值
                if feat_df.isna().any().any():
                    print(f"警告: 发现 {feat_df.isna().sum().sum()} 个无效值，将使用均值填充")
                    feat_df = feat_df.fillna(feat_df.mean())
                
                # 确保没有无限大值
                if np.isinf(feat_df.values).any():
                    print("警告: 发现无限大值，将替换为最大/最小值")
                    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
                    feat_df = feat_df.fillna(feat_df.mean())
                
                # 最终验证
                if feat_df.isna().any().any():
                    raise ValueError("数据预处理后仍存在无效值，请检查原始数据质量")
            
            except Exception as e:
                raise ValueError(f"无法解析特征数据CSV文件 {self.feature_csv}: {str(e)}")
        
        # 按固定列顺序处理（跳过Node Number列）
        coords = feat_df.iloc[:, 1:4].values  # 第2-4列是X,Y,Z坐标
        q_inf = self.config['free_stream']
        x_in = np.hstack([coords, np.tile([q_inf['V'], q_inf['P'], q_inf['rho'], q_inf['h']], (len(feat_df), 1))])

        # 按固定列顺序处理物理量（根据用户提供的完整CSV格式）
        y_out = feat_df.iloc[:, 5:15].values  # 从第6列(Density)开始，取10列物理量

        # 转换为张量
        x = torch.tensor(x_in, dtype=torch.float)
        y = torch.tensor(y_out, dtype=torch.float)

        # 创建图数据对象
        data = Data(x=x, y=y, edge_index=edge_index)
        data.num_nodes = len(feat_df)

        # 区域归一化
        try:
            # 获取区域归一化配置，提供默认值
            region_config = self.config.get('region_normalization', {
                'enabled': False,
                'num_regions': 5
            })
            
            # 尝试从缓存获取region_mask
            if cache_result is not None and cache_result[1] is not None:
                region_mask = cache_result[1]
            elif region_config.get('enabled', False):
                region_mask = assign_regions(data, self.config)
                # 更新缓存包含region_mask
                self._save_to_cache(adj_sparse, region_mask)
        
            if region_config.get('enabled', False):
                region_mask = assign_regions(data, self.config)
                x_norm, region_means, region_stds = self.normalize_features_by_region(x, region_mask)
                data.x = x_norm
                data.region_means = region_means
                data.region_stds = region_stds
                data.region_mask = region_mask
        except Exception as e:
            error_msg = f"""区域归一化配置错误: {str(e)}
当前配置内容: {self.config.get('region_normalization', {})}
请检查配置文件是否包含正确的'region_normalization'配置项
示例配置:
region_normalization:
  enabled: True
  num_regions: 5"""
            raise ValueError(error_msg) from e
        else:
            # 标准归一化
            x_in = self.scaler_x.fit_transform(x_in)
            y_out = self.scaler_y.fit_transform(y_out)
            data.x = torch.tensor(x_in, dtype=torch.float)
            data.y = torch.tensor(y_out, dtype=torch.float)

        return data, self.scaler_x, self.scaler_y
