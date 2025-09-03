# src/dataset.py
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os

class FlowDataset:
    def __init__(self, adj_csv: str, feature_csv: str, config):
        self.adj_csv = adj_csv
        self.feature_csv = feature_csv
        self.config = config
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def load_data(self) -> Data:
        # 读取邻接矩阵（支持多种分隔符）
        try:
            adj_df = pd.read_csv(self.adj_csv, header=None, sep=None, engine='python').values
        except Exception as e:
            raise ValueError(f"无法解析邻接矩阵CSV文件 {self.adj_csv}: {str(e)}")
        edge_index = np.where(adj_df == 1)
        edge_index = torch.tensor(np.array([edge_index[0], edge_index[1]]), dtype=torch.long)

        # 读取节点数据（带数据验证的CSV解析）
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
        # 列顺序：Node Number, X, Y, Z, Density, Eddy Viscosity, Pressure, Static Enthalpy,
        # Temperature, Velocity u, Velocity v, Velocity w, Velocity.Curl X, Velocity.Curl Y, Velocity.Curl Z
        y_out = feat_df.iloc[:, 5:15].values  # 从第6列(Density)开始，取10列物理量

        x_in = self.scaler_x.fit_transform(x_in)
        y_out = self.scaler_y.fit_transform(y_out)

        x = torch.tensor(x_in, dtype=torch.float)
        y = torch.tensor(y_out, dtype=torch.float)

        data = Data(x=x, y=y, edge_index=edge_index)
        data.num_nodes = len(feat_df)
        return data, self.scaler_x, self.scaler_y
