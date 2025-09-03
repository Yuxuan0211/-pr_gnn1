import torch
import torch.nn.functional as F

def loss_supervised(Q_pred: torch.Tensor, Q_true: torch.Tensor) -> torch.Tensor:
    """《分区物理0.17.pdf》2.5节监督损失：L_supervised = 1/N × sum(||Q_pred - Q_true||²)"""
    return F.mse_loss(Q_pred, Q_true, reduction="mean")


def loss_thermo(Q_pred: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    """《分区物理0.17.pdf》2.5.1节热力学一致性损失：L_thermo = 1/N × sum((P - (γ-1)/γ·ρh)²)"""
    P_pred = Q_pred[:, 6]
    rho_pred = Q_pred[:, 7]
    h_pred = Q_pred[:, 9]
    residual = P_pred - ((gamma - 1) / gamma) * rho_pred * h_pred
    return (residual ** 2).mean()


def loss_vorticity(Q_pred: torch.Tensor, Q_true: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """《分区物理0.17.pdf》2.5.3节涡量一致性损失：L_vorticity = 1/N × sum(||ω_pred - ∇×V_pred||²)"""
    # 1. 提取预测涡量与真值涡量（真值来自15个量中的Velocity.Curl）
    omega_pred = Q_pred[:, 3:6]
    omega_true = Q_true[:, 3:6]
    
    # 2. 按两文件2.5.3节计算理论涡量（∇×V_pred，自动微分法）
    V_pred = Q_pred[:, 0:3]  # 预测速度（来自15个量中的Velocity u/v/w）
    jacobian = torch.autograd.functional.jacobian(  # 雅可比矩阵计算（两文件优化方法）
        func=lambda x: V_pred,
        inputs=pos,
        create_graph=True  # 训练时保留梯度，推理时设为False
    )
    # 涡量公式（两文件2.5.3节式5）
    omega_theory = torch.zeros_like(omega_pred)
    omega_theory[:, 0] = jacobian[:, 2, 1] - jacobian[:, 1, 2]  # ωx = ∂Vz/∂y - ∂Vy/∂z
    omega_theory[:, 1] = jacobian[:, 0, 2] - jacobian[:, 2, 0]  # ωy = ∂Vx/∂z - ∂Vz/∂x
    omega_theory[:, 2] = jacobian[:, 1, 0] - jacobian[:, 0, 1]  # ωz = ∂Vy/∂x - ∂Vx/∂y
    
    # 3. 双重约束：匹配真值+满足微分关系（两文件2.5.3节核心要求）
    return 0.5 * F.mse_loss(omega_pred, omega_true) + 0.5 * F.mse_loss(omega_pred, omega_theory)


def loss_energy(Q_pred: torch.Tensor, Q_inf: torch.Tensor, mask_inviscid: torch.Tensor, mask_freestream: torch.Tensor) -> torch.Tensor:
    """《分区物理0.17.pdf》2.5.2节总能量守恒损失：仅无粘区+来流区生效"""
    mask = mask_inviscid | mask_freestream
    if mask.sum() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    
    # 计算预测总焓与来流总焓
    Vx, Vy, Vz = Q_pred[mask, 0], Q_pred[mask, 1], Q_pred[mask, 2]
    h_pred = Q_pred[mask, 9]
    h_total_pred = h_pred + 0.5 * (Vx**2 + Vy**2 + Vz**2)
    
    # 处理Q_inf的不同数据结构情况
    if isinstance(Q_inf, dict):
        h_inf = Q_inf.get('h', 300e3)
        Vx_inf = Q_inf.get('V', 100)
        Vy_inf = Q_inf.get('V', 100)  # 假设与Vx相同
        Vz_inf = Q_inf.get('V', 100)  # 假设与Vx相同
    elif hasattr(Q_inf, '__len__'):
        h_inf = Q_inf[9] if len(Q_inf) > 9 else 300e3
        Vx_inf = Q_inf[0] if len(Q_inf) > 0 else 100
        Vy_inf = Q_inf[1] if len(Q_inf) > 1 else 100
        Vz_inf = Q_inf[2] if len(Q_inf) > 2 else 100
    else:
        h_inf = 300e3
        Vx_inf = Vy_inf = Vz_inf = 100
        
    h_total_inf = h_inf + 0.5 * (Vx_inf**2 + Vy_inf**2 + Vz_inf**2)
    
    return ((h_total_pred - h_total_inf) ** 2).mean()


def loss_noslip(Q_pred: torch.Tensor, mask_boundary: torch.Tensor) -> torch.Tensor:
    """《分区物理0.17.pdf》2.5.4节边界层无滑移损失：L_noslip = 1/|V1| × sum(||V_pred||²)"""
    if mask_boundary.sum() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    
    Vx, Vy, Vz = Q_pred[mask_boundary, 0], Q_pred[mask_boundary, 1], Q_pred[mask_boundary, 2]
    return (Vx**2 + Vy**2 + Vz**2).mean()


def loss_wake(Q_pred: torch.Tensor, mask_wake: torch.Tensor) -> torch.Tensor:
    """《分区物理0.17.pdf》2.5.4节尾流损失：L_wake = 1/|V2| × sum(||V_pred||² + ReLU(C_turb - μt_pred))"""
    if mask_wake.sum() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    
    Vx, Vy, Vz = Q_pred[mask_wake, 0], Q_pred[mask_wake, 1], Q_pred[mask_wake, 2]
    mu_t_pred = Q_pred[mask_wake, 8]
    C_turb = 0.008  # 两文件2.5.4节默认阈值
    return (Vx**2 + Vy**2 + Vz**2 + F.relu(C_turb - mu_t_pred)).mean()


def loss_inviscid(Q_pred: torch.Tensor, mask_inviscid: torch.Tensor) -> torch.Tensor:
    """《分区物理0.17.pdf》2.5.4节无粘区损失：L_inviscid = 1/|V3| × sum(μt_pred²)"""
    if mask_inviscid.sum() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    
    mu_t_pred = Q_pred[mask_inviscid, 8]
    return (mu_t_pred ** 2).mean()


def loss_freestream(Q_pred: torch.Tensor, Q_inf: torch.Tensor, mask_freestream: torch.Tensor) -> torch.Tensor:
    """《分区物理0.17.pdf》2.5.4节来流区损失：L_freestream = 1/|V4| × sum(||Q_pred - Q∞||²)"""
    if mask_freestream.sum() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    
    return F.mse_loss(Q_pred[mask_freestream], Q_inf.repeat(mask_freestream.sum(), 1), reduction="mean")