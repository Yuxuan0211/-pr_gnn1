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

import math
from tqdm import tqdm
import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from pr_gnn.src.physics_loss import PhysicsLoss
from pr_gnn.src.assign_regions import get_regional_masks

class SimpleData:
    def __init__(self, y, pos=None):
        # 确保y是张量
        if isinstance(y, torch.Tensor):
            self.y = y
        elif isinstance(y, (tuple, list)):
            print(f"⚠️  Warning: Converting tuple/list to tensor (length: {len(y)})")
            try:
                self.y = torch.stack(y) if len(y) > 0 else torch.tensor([])
            except Exception as e:
                raise ValueError(f"无法将tuple/list转换为张量: {str(e)}")
        else:
            try:
                self.y = torch.as_tensor(y)
                if not isinstance(self.y, torch.Tensor):
                    raise ValueError(f"转换失败，结果类型: {type(self.y)}")
            except Exception as e:
                raise ValueError(f"无法将输入转换为张量，类型: {type(y)}. 错误: {str(e)}")
        
        # 确保y是二维张量 (num_nodes, num_features)
        if len(self.y.shape) == 1:
            self.y = self.y.unsqueeze(1)
        elif len(self.y.shape) != 2:
            raise ValueError(f"y的形状应为2D (num_nodes, num_features)，实际为: {self.y.shape}")
            
        self.pos = pos if pos is not None else None
    
    def __getattr__(self, name):
        # 将属性访问转发到y张量
        if name in ['size', 'shape', 'device', 'dtype']:
            return getattr(self.y, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __iter__(self):
        # 防止被当作元组处理
        raise TypeError(f"'{type(self).__name__}' object is not iterable")
    
    def __array__(self):
        # 支持numpy转换
        return self.y.numpy()
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # 支持torch函数调用
        if kwargs is None:
            kwargs = {}
        return func(self.y, *args, **kwargs)

class PRGNNTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 混合精度训练初始化
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['training']['mixed_precision'])
        
        # 优化器初始化 (AdamW with weight decay)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器 (预热 + 余弦衰减)
        warmup_epochs = config['training']['warmup_epochs']
        cosine_epochs = config['training']['cosine_epochs']
        total_steps = warmup_epochs + cosine_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_epochs:
                return float(current_step) / float(max(1, warmup_epochs))
            progress = float(current_step - warmup_epochs) / float(max(1, cosine_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda,
            last_epoch=-1
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
            region_mask_bool = (region_mask == region_id)
            region_nodes = region_mask_bool.sum().item()
            
            if region_nodes == 0:
                print(f"⏩ 区域 {region_id} 无节点，自动进入下一区域")
                continue
                
            print(f"\n=== 开始预训练区域 {region_id} ===")
            print(f"📊 区域节点数: {region_nodes}")
            print(f"⏳ 训练轮数: {adjusted_epochs}")
            
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
                
                # 每50轮记录一次进度
                if (epoch + 1) % 50 == 0 or epoch == adjusted_epochs - 1:
                    log_msg = f"[区域{region_id}] 轮次 {epoch + 1}/{adjusted_epochs} 损失: {total_loss.item():.4f}"
                    if val_data is not None:
                        val_loss = self._evaluate(val_data)['val_total_loss']
                        log_msg += f" | 验证损失: {val_loss:.4f}"
                    print(log_msg)
                
                # 记录训练损失
                self.train_state['train_loss_history'].append(total_loss.item())
                
                # 验证集评估并更新学习率（如果有）
                if val_data is not None:
                    val_loss_dict = self._evaluate(val_data)
                    self.train_state['val_loss_history'].append(val_loss_dict['val_total_loss'])
                    self.scheduler.step(val_loss_dict['val_total_loss'])

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

            print(f"✅ 区域 {region_id} 预训练完成")

        # 返回训练历史
        return {
            'train_loss': self.train_state['train_loss_history'],
            'val_loss': self.train_state['val_loss_history'],
            'lr_history': self.train_state['lr_history']
        }

    def global_finetune(self, data, epochs, val_data=None, batch_size=None):
        # 处理多流场数据
        if hasattr(data, 'multi_mach_y') and batch_size is not None:
            # 随机选择batch_size个流场
            selected_indices = torch.randperm(data.multi_mach_y.size(1))[:batch_size]
            data.y = data.multi_mach_y[:, selected_indices].mean(dim=1)  # 使用均值作为当前批次的y
            print(f"🔀 使用流场批次训练 (批次大小: {batch_size})")
        
        data = data.to(self.device)
        if val_data is not None:
            val_data = val_data.to(self.device)
        
        self.model.train()
        region_mask = self._get_region_mask(data)
        total_nodes = len(data.y)
        
        # 初始化train_loader
        train_loader = None
        
        # 邻居采样配置
        if self.config['training']['neighbor_sampling']:
            num_neighbors = [self.config['training']['num_neighbors']] * self.config['training']['num_layers']
            batch_size = min(2048, total_nodes)  # 邻居采样时使用较小的batch size
            
            train_loader = NeighborSampler(
                data.edge_index, 
                node_idx=None, 
                sizes=num_neighbors,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            print(f"📊 全局微调配置：总节点数{total_nodes}，邻居采样batch size={batch_size}，邻居数={num_neighbors}")
        else:
<<<<<<< HEAD
            # 动态调整batch size
            min_batch_size = 64
            max_batch_size = 2048
            target_batches = 100  # 目标批次数
            
            # 智能计算batch size
            batch_size = min(
                max_batch_size,
                max(min_batch_size, total_nodes // target_batches)
            )
            
            # 创建虚拟train_loader以保持代码结构一致
            train_loader = [(batch_size, torch.arange(total_nodes), [])]  # 使用元组模拟NeighborSampler输出
            
            # 确保不超过显存限制
            if torch.cuda.is_available():
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                safe_batch_size = min(batch_size, free_mem // (1024 * 1024 * 10))  # 估算10MB/样本
                if safe_batch_size < batch_size:
                    print(f"⚠️  显存限制，batch size从{batch_size}调整为{safe_batch_size}")
                    batch_size = safe_batch_size
            
=======
            # 使用配置中的batch size
            batch_size = min(self.config['training']['batch_size'], total_nodes)
>>>>>>> 04127fc6411003847074387eb6378920c0ab225b
            print(f"📊 全局微调配置：总节点数{total_nodes}，batch size={batch_size}，最大轮数={epochs}")

        # 梯度累积参数
        grad_accum_steps = self.config.get('grad_accum_steps', 1)
        if grad_accum_steps > 1:
            print(f"⚠️  启用梯度累积，累积步数：{grad_accum_steps}")

        # 微调训练循环
        with tqdm(range(epochs), desc="全局微调", unit="epoch") as pbar:
            for epoch in pbar:
                # 更新进度条描述
                pbar.set_postfix({
                    'batch': f"{batch_size}/{total_nodes}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}"
                })
                
                if self.config['training']['neighbor_sampling']:
                    # 邻居采样训练
                    total_train_loss = 0.0
                batch_count = 0
                
                for batch_size, n_id, adjs in train_loader:
                    adjs = [adj.to(self.device) for adj in adjs]
                    self.optimizer.zero_grad() if batch_count % grad_accum_steps == 0 else None
                    
                    # 混合精度训练
                    with torch.cuda.amp.autocast(enabled=self.config['training']['mixed_precision']):
                        # 将adjs转换为SparseTensor格式
                        if len(adjs) > 0 and hasattr(adjs[0], 'edge_index'):
                            # 确保只使用当前批次的节点
                            batch_nodes = n_id[:batch_size]
                            adj = SparseTensor(
                                row=adjs[0].edge_index[0],
                                col=adjs[0].edge_index[1],
                                sparse_sizes=(len(batch_nodes), len(batch_nodes))
                            )
                            out = self.model(data.x[batch_nodes], adj)
                        else:
                            # 如果没有有效的邻接信息，使用全连接
                            batch_nodes = n_id[:batch_size]
                            adj = SparseTensor(
                                row=torch.arange(len(batch_nodes), device=self.device),
                                col=torch.arange(len(batch_nodes), device=self.device),
                                sparse_sizes=(len(batch_nodes), len(batch_nodes))
                            )
                            out = self.model(data.x[batch_nodes], adj)
                        
                        # 处理pos数据
                        pos = None
                        if hasattr(data, 'pos') and data.pos is not None:
                            pos = data.pos[n_id]
                        
                # 调试日志
                print(f"data.y type before SimpleData: {type(data.y)}")
                if hasattr(data.y, 'shape'):
                    print(f"data.y shape: {data.y.shape}")
                elif isinstance(data.y, (tuple, list)):
                    print(f"data.y length: {len(data.y)}")
                
                # 确保y是tensor
                y_tensor = torch.as_tensor(data.y[n_id]) if not isinstance(data.y[n_id], torch.Tensor) else data.y[n_id]
                print(f"y_tensor type: {type(y_tensor)}, shape: {y_tensor.shape}")
                
                temp_data = SimpleData(
                    y=y_tensor,
                    pos=pos
                )
                
                batch_loss, _ = self.physics_loss(
                    out,
                    temp_data,
                    region_mask[n_id]
                )
                
                batch_loss = batch_loss / grad_accum_steps
                self.scaler.scale(batch_loss).backward()
                total_train_loss += batch_loss.item() * grad_accum_steps
                batch_count += 1

                # 累积到指定步数后更新参数
                if batch_count % grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                
                # 计算平均训练损失
                avg_train_loss = total_train_loss / batch_count
                self.train_state['train_loss_history'].append(avg_train_loss)
            else:
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
                # 每50轮保存一次模型和状态
                if epoch % 50 == 0 or epoch == epochs - 1:
                    log_msg = (f"微调 Epoch {epoch:3d}/{epochs} | LR: {current_lr:.6f} | "
                           f"Avg Train Loss: {avg_train_loss:.6f}")
                    if val_data is not None:
                        log_msg += (f" | Val Total Loss: {val_loss_dict['val_total_loss']:.6f}")
                    print(log_msg)
                    
                    # 保存检查点
                    self.save_model(f"models/checkpoint_epoch_{epoch}.pth")
                    self.save_train_state(f"models/checkpoint_state_{epoch}.pth")

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
                return {
                    'train_loss': self.train_state['train_loss_history'],
                    'val_loss': self.train_state['val_loss_history'],
                    'lr_history': self.train_state['lr_history']
                }

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
