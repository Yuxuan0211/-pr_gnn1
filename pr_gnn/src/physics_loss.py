# src/physics_loss.py
import torch
import torch.nn.functional as F
from pr_gnn.src.itemize_loss import (
    loss_supervised,
    loss_thermo,
    loss_vorticity,
    loss_energy,
    loss_noslip,
    loss_wake,
    loss_inviscid,
    loss_freestream
)

class PhysicsLoss:
    def __init__(self, config):
        self.config = config
        from ..config import get_config_value
        self.get_config = get_config_value
        
    def __call__(self, pred, true, region_mask):
        # 构造mask字典
        masks = {
            "boundary": (region_mask == 0),
            "inviscid": (region_mask == 1),
            "wake": (region_mask == 2),
            "freestream": (region_mask == 4)
        }
        
        # 获取配置参数
        gamma = self.get_config('gamma', 1.4)
        free_stream = self.get_config('free_stream', 
                                   {'V': 100, 'P': 101325, 'rho': 1.225, 'h': 300e3})
        
        # 设置默认权重值
        default_weights = {
            'w_thermo': 1.0,
            'w_vorticity': 10.0,
            'w_noslip': 5.0,
            'w_wake': 1.0,
            'w_energy': 1.0,
            'w_inviscid': 0.5,
            'w_freestream': 10.0,
            'lambda_phys': 1.0
        }
        
        # 合并默认值和配置值
        weights = {**default_weights, **self.config}
        
        # 计算各项loss
        L_sup = loss_supervised(pred, true)
        L_thermo = loss_thermo(pred, gamma) * weights["w_thermo"]
        # 处理涡量计算的位置信息
        pos = true.pos if hasattr(true, 'pos') else None
        if pos is None:
            L_vort = torch.tensor(0.0, device=pred.device)
        else:
            L_vort = loss_vorticity(pred, true, pos) * weights["w_vorticity"]
        L_noslip = loss_noslip(pred, masks["boundary"]) * weights["w_noslip"]
        L_wake = loss_wake(pred, masks["wake"]) * weights["w_wake"]
        L_energy = loss_energy(pred, free_stream, 
                             masks["inviscid"], masks["freestream"]) * weights["w_energy"]
        L_inviscid = loss_inviscid(pred, masks["inviscid"]) * weights["w_inviscid"]
        L_freestream = loss_freestream(pred, free_stream, 
                                     masks["freestream"]) * weights["w_freestream"]
        
        # 总物理约束Loss与总Loss
        L_physics = L_thermo + L_vort + L_noslip + L_wake + L_energy + L_inviscid + L_freestream
        lambda_phys = weights.get("lambda_phys", 1.0)
        L_total = L_sup + lambda_phys * L_physics
        
        # Loss日志
        loss_dict = {
            "L_total": L_total.item(),
            "L_supervised": L_sup.item(),
            "L_thermo": L_thermo.item(),
            "L_vorticity": L_vort.item(),
            "L_noslip": L_noslip.item(),
            "L_wake": L_wake.item(),
            "L_energy": L_energy.item(),
            "L_inviscid": L_inviscid.item(),
            "L_freestream": L_freestream.item()
        }
        
        return L_total, loss_dict
