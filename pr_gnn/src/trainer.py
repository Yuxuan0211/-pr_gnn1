# src/trainer.py
import torch
from torch.optim import Adam
try:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except ImportError:
    print("⚠️  无法导入ReduceLROnPlateau，将使用简化版实现")
    class ReduceLROnPlateau:
        def __init__(self, optimizer, mode='min', factor=0.1, patience=10, 
                    min_lr=0, verbose=False):
            self.optimizer = optimizer
            self.mode = mode
            self.factor = factor
            self.patience = patience
            self.min_lr = min_lr
            self.verbose = verbose
            self.best = float('inf') if mode == 'min' else -float('inf')
            self.num_bad_epochs = 0
            
        def step(self, metrics):
            if self.mode == 'min':
                is_better = metrics < self.best
            else:
                is_better = metrics > self.best
                
            if is_better:
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
                
            if self.num_bad_epochs >= self.patience:
                self._reduce_lr()
                self.num_bad_epochs = 0
                
        def _reduce_lr(self):
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                if old_lr - new_lr > 1e-8:
                    param_group['lr'] = new_lr
                    if self.verbose:
                        print(f'将学习率从 {old_lr:.2e} 降低到 {new_lr:.2e}')

from tqdm import tqdm
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from pr_gnn.src.physics_loss import PhysicsLoss
from pr_gnn.src.assign_regions import get_regional_masks

class SimpleData(torch.Tensor):
    def __new__(cls, y, pos=None):
        # 使用y作为基础张量
        instance = super().__new__(cls, y)
        instance.pos = pos
        return instance
    
    def __init__(self, y, pos=None):
        super().__init__()
        # 确保y是张量
        if not isinstance(y, torch.Tensor):
            self.data = torch.as_tensor(y)
        else:
            self.data = y
        self.pos = pos
    
    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

class PRGNNTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器初始化
        self.optimizer = Adam(
            model.parameters(),
            lr=config.get('lr', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.get('lr_factor', 0.5),
            patience=config.get('lr_patience', 10),
            min_lr=config.get('min_lr', 1e-6),
            verbose=True
        )
        
        # 物理损失计算器
        self.physics_loss = PhysicsLoss(config)
        
        # 训练状态记录
        self.train_state = {
            'train_loss_history': [],
            'val_loss_history': [],
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'converge_count': 0,
            'current_epoch': 0,
            'lr_history': []
        }
        
        # 区域掩码缓存
        self.region_mask_cache = None

    def _get_region_mask(self, data) -> torch.Tensor:
        """获取区域掩码（带缓存机制）"""
        if self.region_mask_cache is None or len(self.region_mask_cache) != len(data.y):
            masks = get_regional_masks(data.y)
            region_mask = torch.zeros(len(data.y), dtype=torch.long, device=self.device)
            region_mask[masks["shock"]] = 0
            region_mask[masks["boundary"]] = 1
            region_mask[masks["wake"]] = 2
            region_mask[masks["inviscid"]] = 3
            region_mask[masks["freestream"]] = 4
            self.region_mask_cache = region_mask
        return self.region_mask_cache

    def regional_pretrain(self, data, val_data=None):
        data = data.to(self.device)
        if val_data is not None:
            val_data = val_data.to(self.device)
        
        self.model.train()
        region_mask = self._get_region_mask(data)

        # 根据节点数动态调整训练轮数
        total_nodes = len(data.y)
        base_epochs = self.config.get('pre_epochs', 100)
        scale_factor = min(1.0, 10000 / total_nodes)
        adjusted_epochs = max(10, int(base_epochs * scale_factor))
        print(f"📊 区域预训练配置：总节点数{total_nodes}，调整后轮数{adjusted_epochs}")

        # 按区域进行预训练
        for region_id in range(5):
            print(f"\n=== 预训练阶段: 区域 {region_id} ===")
            region_mask_bool = (region_mask == region_id)
            
            # 跳过无节点的区域
            if not region_mask_bool.any():
                print(f"⚠️  区域 {region_id} 无节点，跳过该区域")
                continue
            
            # 重置该区域的收敛计数器
            self.train_state['converge_count'] = 0
            
            for epoch in range(adjusted_epochs):
                self.train_state['current_epoch'] += 1
                current_lr = self.optimizer.param_groups[0]['lr']
                self.train_state['lr_history'].append(current_lr)

                # 训练步骤
                self.optimizer.zero_grad()
                pred, _ = self.model(data.x, data.edge_index)
                # 将布尔掩码转换为整数索引
                region_indices = torch.where(region_mask_bool)[0]
                
                # 处理pos数据
                pos = None
                if hasattr(data, 'pos') and data.pos is not None:
                    pos = data.pos[region_indices]
                
                # 直接传递y张量而不是封装对象
                temp_data = SimpleData(
                    y=data.y[region_indices],
                    pos=pos
                )
                
                total_loss, loss_dict = self.physics_loss(
                    pred[region_indices],
                    temp_data,
                    region_mask[region_indices]
                )
                total_loss.backward()
                self.optimizer.step()

                # 记录训练损失
                self.train_state['train_loss_history'].append(total_loss.item())

                # 验证集评估（如果有）
                val_loss_dict = {}
                if val_data is not None:
                    val_loss_dict = self._evaluate(val_data)
                    self.train_state['val_loss_history'].append(val_loss_dict['val_total_loss'])
                    # 更新学习率调度器
                    self.scheduler.step(val_loss_dict['val_total_loss'])

                # 打印日志（每5个epoch）
                if epoch % 5 == 0:
                    log_msg = (f"区域 {region_id} | Epoch {epoch:3d}/{adjusted_epochs} | LR: {current_lr:.6f} | "
                               f"Train Loss: {total_loss.item():.6f} | "
                               f"Sup Loss: {loss_dict['L_supervised']:.6f}")
                    if val_data is not None:
                        log_msg += f" | Val Loss: {val_loss_dict['val_total_loss']:.6f}"
                    print(log_msg)

                # 收敛检验（如果有验证集）
                if val_data is not None:
                    is_converged, converge_msg = self._check_convergence()
                    print(f"🔍 收敛状态: {converge_msg}")
                    
                    # 保存最佳模型
                    if self.train_state['current_epoch'] == self.train_state['best_epoch']:
                        self.save_model(f"models/best_pretrain_region_{region_id}.pth")
                    
                    # 如果收敛，提前停止该区域训练
                    if is_converged:
                        print(f"🎉 区域 {region_id} 预训练提前收敛，停止训练")
                        break

            print(f"=== 区域 {region_id} 预训练完成 ===")

        # 返回训练历史
        return {
            'train_loss': self.train_state['train_loss_history'],
            'val_loss': self.train_state['val_loss_history'],
            'lr_history': self.train_state['lr_history']
        }

    def global_finetune(self, data, epochs, val_data=None):
        data = data.to(self.device)
        if val_data is not None:
            val_data = val_data.to(self.device)
        
        self.model.train()
        region_mask = self._get_region_mask(data)
        total_nodes = len(data.y)
        
        # 动态调整batch size
        if total_nodes > 10000:
            batch_size = 1024
        elif total_nodes > 5000:
            batch_size = 512
        else:
            batch_size = 256
        print(f"📊 全局微调配置：总节点数{total_nodes}，batch size{batch_size}，最大轮数{epochs}")

        # 梯度累积参数
        grad_accum_steps = self.config.get('grad_accum_steps', 1)
        if grad_accum_steps > 1:
            print(f"⚠️  启用梯度累积，累积步数：{grad_accum_steps}")

        # 微调训练循环
        for epoch in tqdm(range(epochs), desc="全局微调"):
            self.train_state['current_epoch'] += 1
            current_lr = self.optimizer.param_groups[0]['lr']
            self.train_state['lr_history'].append(current_lr)
            
            total_train_loss = 0.0
            batch_count = 0

            # 分批处理大数据集
            for i in range(0, total_nodes, batch_size):
                # 计算当前批次的掩码
                batch_end = min(i + batch_size, total_nodes)
                batch_mask = slice(i, batch_end)
                
                # 批次训练（梯度累积）
                self.optimizer.zero_grad() if batch_count % grad_accum_steps == 0 else None
                
                pred, _ = self.model(data.x, data.edge_index)
                # 只计算当前批次的损失
                # 处理pos数据
                pos = None
                if hasattr(data, 'pos') and data.pos is not None:
                    pos = data.pos[batch_mask]
                
                temp_data = SimpleData(
                    y=data.y[batch_mask],
                    pos=pos
                )
                
                batch_loss, _ = self.physics_loss(
                    pred[batch_mask],
                    temp_data,
                    region_mask[batch_mask]
                )
                
                # 梯度累积：损失除以累积步数
                batch_loss = batch_loss / grad_accum_steps
                batch_loss.backward()
                
                total_train_loss += batch_loss.item() * grad_accum_steps
                batch_count += 1

                # 累积到指定步数后更新参数
                if batch_count % grad_accum_steps == 0 or batch_count == (total_nodes // batch_size + 1):
                    self.optimizer.step()

            # 计算平均训练损失
            avg_train_loss = total_train_loss / batch_count
            self.train_state['train_loss_history'].append(avg_train_loss)

            # 验证集评估（如果有）
            val_loss_dict = {}
            if val_data is not None:
                val_loss_dict = self._evaluate(val_data)
                self.train_state['val_loss_history'].append(val_loss_dict['val_total_loss'])
                # 更新学习率调度器
                self.scheduler.step(val_loss_dict['val_total_loss'])

            # 打印日志（每10个epoch）
            if epoch % 10 == 0:
                log_msg = (f"微调 Epoch {epoch:3d}/{epochs} | LR: {current_lr:.6f} | "
                           f"Avg Train Loss: {avg_train_loss:.6f}")
                if val_data is not None:
                    log_msg += (f" | Val Total Loss: {val_loss_dict['val_total_loss']:.6f} | "
                               f"Val Sup Loss: {val_loss_dict['L_supervised']:.6f} | "
                               f"Val Phys Loss: {val_loss_dict['L_thermo'] + val_loss_dict['L_vorticity']:.6f}")
                print(log_msg)

            # 收敛检验（如果有验证集）
            if val_data is not None:
                is_converged, converge_msg = self._check_convergence()
                print(f"🔍 收敛状态: {converge_msg}")
                
                # 保存最佳模型
                if self.train_state['current_epoch'] == self.train_state['best_epoch']:
                    self.save_model("models/best_finetune.pth")
                    self.save_train_state("models/best_finetune_state.pth")
                
                # 如果收敛，提前停止微调
                if is_converged:
                    print(f"🎉 全局微调提前收敛，停止训练（总训练轮数：{self.train_state['current_epoch']}）")
                    break

        print("=== 全局微调完成 ===")

        # 返回训练历史
        return {
            'train_loss': self.train_state['train_loss_history'],
            'val_loss': self.train_state['val_loss_history'],
            'lr_history': self.train_state['lr_history']
        }

    def _check_convergence(self) -> Tuple[bool, str]:
        """
        收敛检验逻辑
        返回：(是否收敛, 收敛状态信息)
        """
        # 检查是否达到最小训练轮数
        if self.train_state['current_epoch'] < self.config.get('min_epochs', 50):
            return False, f"未达到最小训练轮数（当前{self.train_state['current_epoch']}/{self.config.get('min_epochs', 50)}）"
        
        # 获取监控指标的历史记录
        if not self.train_state['val_loss_history']:
            return False, "无验证损失记录，无法判断收敛"
        
        current_metric = self.train_state['val_loss_history'][-1]
        best_metric = self.train_state['best_val_loss']
        
        # 判断是否有改进
        improvement = best_metric - current_metric
        min_delta = self.config.get('converge_min_delta', 1e-6)
        
        if improvement > min_delta:
            # 有显著改进：更新最佳状态
            self.train_state['best_val_loss'] = current_metric
            self.train_state['best_epoch'] = self.train_state['current_epoch']
            self.train_state['converge_count'] = 0
            return False, f"指标改进{improvement:.6f}，更新最佳状态（epoch {self.train_state['best_epoch']}）"
        else:
            # 无显著改进：累计计数
            self.train_state['converge_count'] += 1
            if self.train_state['converge_count'] >= self.config.get('converge_patience', 15):
                # 达到容忍上限，判定收敛
                return True, (f"连续{self.train_state['converge_count']}个epoch无显著改进（阈值{min_delta}），"
                              f"训练收敛（最佳epoch: {self.train_state['best_epoch']}, 最佳验证损失: {best_metric:.6f}）")
            else:
                return False, f"无显著改进，累计计数{self.train_state['converge_count']}/{self.config.get('converge_patience', 15)}"

    def _evaluate(self, data) -> Dict[str, float]:
        """
        验证集评估（无梯度计算）
        返回：各损失指标字典
        """
        self.model.eval()
        data = data.to(self.device)
        region_mask = self._get_region_mask(data)
        
        with torch.no_grad():
            pred, _ = self.model(data.x, data.edge_index)
            
            # 创建临时数据对象以匹配physics_loss接口
            temp_data = SimpleData(
                y=data.y,
                pos=data.pos if hasattr(data, 'pos') else None
            )
            
            total_loss, loss_dict = self.physics_loss(
                pred,
                temp_data,
                region_mask
            )
        
        # 补充总损失到返回字典
        loss_dict['val_total_loss'] = total_loss.item()
        self.model.train()  # 恢复训练模式
        return loss_dict

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存至: {path}")

    def save_train_state(self, path: str) -> None:
        """保存训练状态（用于中断后恢复）"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_state': self.train_state,
            'config': self.config
        }
        # 确保保存目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(save_dict, path)
        print(f"✅ 训练状态已保存至: {path}")

    def load_train_state(self, path: str) -> None:
        """加载训练状态（从中断处恢复）"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_state = checkpoint['train_state']
        self.config = checkpoint['config']
        print(f"✅ 训练状态已加载（当前epoch: {self.train_state['current_epoch']}, 最佳验证损失: {self.train_state['best_val_loss']:.6f}）")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"模型已加载: {path}")
