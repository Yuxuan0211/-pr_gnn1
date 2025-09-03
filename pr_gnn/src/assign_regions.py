import torch
import torch.nn.functional as F
from pr_gnn.config import get_config_value

def get_regional_masks(Q_true: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    按《分区物理0.17.pdf》表1、《分区物理0.18.pdf》2.3节划分5区域，仅用15个量映射的Q向量
    :param Q_true: 真值Q向量 (N_nodes, 10)
    :return: 区域掩码字典（布尔型，True表示节点属于该区域）
    """
    # 获取配置参数
    C1 = get_config_value('C1', 1.5)
    C2 = get_config_value('C2', 0.01)
    C3 = get_config_value('C3', 0.3)
    C4 = get_config_value('C4', 0.005)
    C5 = get_config_value('C5', 0.5)
    C6 = get_config_value('C6', 0.001)
    C7 = get_config_value('C7', 0.1)
    # 从Q向量提取15个量映射的关键物理量
    Vx, Vy, Vz = Q_true[:, 0], Q_true[:, 1], Q_true[:, 2]
    P, rho, mu_t = Q_true[:, 6], Q_true[:, 7], Q_true[:, 8]
    V_mag = torch.sqrt(Vx**2 + Vy**2 + Vz**2)  # 速度大小（表1划分必需）
    
    # 计算来流条件Q∞（表1依赖，用"低μt+高速度"节点均值，两文件2.3节方法）
    temp_freestream_mask = (mu_t < C6) & (V_mag > 0.9 * V_mag.max())
    Q_inf = Q_true[temp_freestream_mask].mean(dim=0) if temp_freestream_mask.sum() > 0 else Q_true.mean(dim=0)
    Vx_inf, P_inf, rho_inf = Q_inf[0], Q_inf[6], Q_inf[7]
    V_inf_mag = torch.sqrt(Q_inf[0]**2 + Q_inf[1]**2 + Q_inf[2]**2)
    
    # 1. 自由来流区（区域4，表1）：μt< C6 + 物理量匹配来流
    mask_freestream = (mu_t < C6) & \
                      (torch.abs(P - P_inf)/P_inf < C7) & \
                      (torch.abs(rho - rho_inf)/rho_inf < C7) & \
                      (torch.abs(V_mag - V_inf_mag)/V_inf_mag < C7)
    
    # 2. 激波区（区域0，表1）：P> C1×P∞ 或 ρ> C1×ρ∞（排除来流区）
    mask_shock = ((P > C1 * P_inf) | (rho > C1 * rho_inf)) & ~mask_freestream
    
    # 3. 边界层区（区域1，表1）：μt> C2 + 速度< C3×V∞（排除其他区）
    mask_boundary = (mu_t > C2) & (V_mag < C3 * V_inf_mag) & \
                    ~mask_shock & ~mask_freestream
    
    # 4. 尾流区（区域2，表1）：μt> C4 + Vx< C5×Vx∞（排除其他区）
    mask_wake = (mu_t > C4) & (Vx < C5 * Vx_inf) & \
                ~mask_shock & ~mask_boundary & ~mask_freestream
    
    # 5. 无粘区（区域3，表1）：μt< C6（排除所有特殊区）
    mask_inviscid = (mu_t < C6) & \
                    ~mask_shock & ~mask_boundary & ~mask_wake & ~mask_freestream
    
    return {
        "shock": mask_shock,       # 激波区（区域0）
        "boundary": mask_boundary, # 边界层区（区域1）
        "wake": mask_wake,         # 尾流区（区域2）
        "inviscid": mask_inviscid, # 无粘区（区域3）
        "freestream": mask_freestream # 来流区（区域4）
    }