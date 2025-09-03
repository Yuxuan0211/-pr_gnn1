# src/trainer.py
import torch
from torch.optim import Adam
from tqdm import tqdm
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from pr_gnn.src.physics_loss import PhysicsLoss
from pr_gnn.src.assign_regions import get_regional_masks

class PRGNNTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = Adam(model.parameters(), lr=config['lr'])
        self.physics_loss = PhysicsLoss(config)
        self.device = config['device']

    def regional_pretrain(self, data):
        data = data.to(self.device)
        self.model.train()
        masks = get_regional_masks(data.y)
        # 将mask字典转换为区域编号
        region_mask = torch.zeros(len(data.y), dtype=torch.long, device=self.device)
        region_mask[masks["shock"]] = 0
        region_mask[masks["boundary"]] = 1
        region_mask[masks["wake"]] = 2
        region_mask[masks["inviscid"]] = 3
        region_mask[masks["freestream"]] = 4

        # 根据节点数动态调整训练轮数
        total_nodes = len(data.y)
        base_epochs = self.config.get('pre_epochs', 100)
        scale_factor = min(1.0, 10000 / total_nodes)  # 节点数越多，轮数越少
        adjusted_epochs = max(10, int(base_epochs * scale_factor))

        for r in range(5):
            print(f"--- 预训练阶段: 区域 {r} ---")
            mask = (region_mask == r)
            if not mask.any():
                continue
            for epoch in range(adjusted_epochs):
                self.optimizer.zero_grad()
                pred, _ = self.model(data.x, data.edge_index)
                total_loss, _ = self.physics_loss(pred[mask], data.y[mask], region_mask[mask])
                total_loss.backward()
                self.optimizer.step()
                if epoch % 50 == 0:
                    print(f"Region {r}, Epoch {epoch}, Loss: {total_loss.item():.6f}")
            print(f"区域 {r} 预训练完成。")

    def global_finetune(self, data, epochs):
        data = data.to(self.device)
        self.model.train()
        masks = get_regional_masks(data.y)
        region_mask = torch.zeros(len(data.y), dtype=torch.long, device=self.device)
        region_mask[masks["shock"]] = 0
        region_mask[masks["boundary"]] = 1
        region_mask[masks["wake"]] = 2
        region_mask[masks["inviscid"]] = 3
        region_mask[masks["freestream"]] = 4

        # 根据节点数动态调整batch size
        total_nodes = len(data.y)
        if total_nodes > 10000:
            batch_size = 1024
        elif total_nodes > 5000:
            batch_size = 512
        else:
            batch_size = 256

        for epoch in tqdm(range(epochs)):
            # 分批处理大数据集
            for i in range(0, total_nodes, batch_size):
                batch_mask = slice(i, min(i + batch_size, total_nodes))
            self.optimizer.zero_grad()
            pred, _ = self.model(data.x, data.edge_index)
            total_loss, loss_dict = self.physics_loss(pred, data.y, region_mask)
            total_loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                print(f"微调 Epoch {epoch}, Total Loss: {total_loss.item():.6f}, Sup: {loss_dict['L_supervised']:.6f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存至: {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"模型已加载: {path}")
