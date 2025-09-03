import torch
import torch.nn.functional as F
import numpy as np


# -------------------------- 1. 全局常量配置（需根据工况调整） --------------------------
CONFIG = {
    # 物理常数
    "gamma": 1.4,          # 比热比（空气默认1.4）
    "R": 287.0,            # 气体常数（空气默认287 J/(kg·K)）
    "cp": 1005.0,          # 定压比热（空气默认1005 J/(kg·K)）
    
    # 区域划分阈值（论文表1，需根据流场特性调参）
    "C1": 1.2,             # 激波区：P/ρ > C1*来流值
    "C2": 0.01,            # 边界层区：湍流粘度 > C2 (Pa·s)
    "C3": 0.3,             # 边界层区：速度 < C3*来流速度
    "C4": 0.005,           # 尾流区：湍流粘度 > C4 (Pa·s)
    "C5": 0.5,             # 尾流区：x方向速度 < C5*来流速度
    "C6": 0.001,           # 外部无粘区：湍流粘度 < C6 (Pa·s)
    "C7": 0.05,            # 自由来流区：物理量与来流值偏差 < C7（相对误差）
    
    # 湍流损失阈值
    "C_turb": 0.008,       # 尾流区：湍流粘度最低阈值 (Pa·s)
    
    # Loss权重（论文2.5/2.6，需训练调优）
    "lambda_phys": 0.5,    # 物理约束总权重
    "w_thermo": 1.0,       # 热力学损失权重
    "w_vorticity": 1.0,    # 涡量损失权重
    "w_noslip": 2.0,       # 无滑移损失权重
    "w_wake": 1.5,         # 尾流损失权重
    "w_energy": 1.0,       # 能量守恒损失权重
    "w_inviscid": 1.0,     # 无粘区损失权重
    "w_freestream": 2.0    # 自由来流区损失权重
}


# -------------------------- 2. 数据预处理工具（适配用户输入） --------------------------
def parse_node_data(node_data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将用户提供的节点数据解析为模型输入格式
    :param node_data: 节点数据矩阵，shape=(N_nodes, 12)，列顺序为：
                     [Node Number, X, Y, Z, Density, Eddy Viscosity, 
                      Pressure, Static Enthalpy, Temperature, Velocity u, Velocity v, Velocity w]
    :return:
        pos: 节点坐标，shape=(N_nodes, 3)，requires_grad=True（用于涡量计算）
        Q_true: 物理量真值，shape=(N_nodes, 10)（对应论文Q向量）
        Q_pred: 模型预测物理量（示例中用真值加噪声模拟，实际替换为模型输出）
    """
    # 提取坐标（需开启梯度，用于计算速度偏导数）
    pos = torch.tensor(node_data[:, 1:4], dtype=torch.float32, requires_grad=True)
    
    # 解析物理量真值（映射到论文10维Q向量）
    Q_true = torch.zeros((node_data.shape[0], 10), dtype=torch.float32)
    Q_true[:, 0] = node_data[:, 9]    # Vx = Velocity u
    Q_true[:, 1] = node_data[:, 10]   # Vy = Velocity v
    Q_true[:, 2] = node_data[:, 11]   # Vz = Velocity w
    Q_true[:, 6] = node_data[:, 6]    # P = Pressure
    Q_true[:, 7] = node_data[:, 4]    # ρ = Density
    Q_true[:, 8] = node_data[:, 5]    # μt = Eddy Viscosity
    Q_true[:, 9] = node_data[:, 7]    # h = Static Enthalpy
    
    # 计算涡量真值（ω = ∇×V，替换为模型预测时删除此步）
    omega_true = compute_curl(pos, Q_true[:, :3])  # shape=(N_nodes, 3)
    Q_true[:, 3:6] = omega_true
    
    # 模拟模型预测（实际使用时替换为PR-GNN输出）
    Q_pred = Q_true + torch.randn_like(Q_true) * 0.01  # 加小噪声模拟预测误差
    
    return pos, Q_true, Q_pred


def get_free_stream_condition(Q_true: torch.Tensor, mask_freestream: torch.Tensor) -> torch.Tensor:
    """
    计算来流条件Q_inf（自由来流区物理量均值）
    :param Q_true: 物理量真值，shape=(N_nodes, 10)
    :param mask_freestream: 自由来流区掩码，shape=(N_nodes,)（布尔型）
    :return: Q_inf: 来流物理量，shape=(10,)
    """
    if mask_freestream.sum() == 0:
        raise ValueError("自由来流区无节点，请检查区域划分阈值！")
    return Q_true[mask_freestream].mean(dim=0)


# -------------------------- 3. 流场区域划分（论文表1） --------------------------
def compute_regional_masks(Q_true: torch.Tensor, Q_inf: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    根据物理量真值划分5个区域，生成节点掩码（布尔型，True表示属于该区域）
    :param Q_true: 物理量真值，shape=(N_nodes, 10)
    :param Q_inf: 来流条件，shape=(10,)
    :return: masks: 区域掩码字典，keys=["shock", "boundary", "wake", "inviscid", "freestream"]
    """
    # 提取关键物理量（真值）
    Vx, Vy, Vz = Q_true[:, 0], Q_true[:, 1], Q_true[:, 2]
    P, rho, mu_t = Q_true[:, 6], Q_true[:, 7], Q_true[:, 8]
    V_mag = torch.sqrt(Vx**2 + Vy**2 + Vz**2)  # 速度大小
    Vx_inf = Q_inf[0]                          # 来流x方向速度
    P_inf, rho_inf = Q_inf[6], Q_inf[7]        # 来流压力、密度
    
    # 1. 自由来流区（区域4）：μt≈0 + 物理量接近来流值
    mask_freestream = (mu_t < CONFIG["C6"]) & \
                      (torch.abs(P - P_inf) / P_inf < CONFIG["C7"]) & \
                      (torch.abs(rho - rho_inf) / rho_inf < CONFIG["C7"]) & \
                      (torch.abs(V_mag - torch.sqrt(Q_inf[0]**2 + Q_inf[1]**2 + Q_inf[2]**2)) / 
                       torch.sqrt(Q_inf[0]**2 + Q_inf[1]**2 + Q_inf[2]**2) < CONFIG["C7"])
    
    # 2. 激波区（区域0）：P> C1*P_inf 或 ρ> C1*ρ_inf（排除自由来流区）
    mask_shock = ((P > CONFIG["C1"] * P_inf) | (rho > CONFIG["C1"] * rho_inf)) & ~mask_freestream
    
    # 3. 边界层区（区域1）：μt> C2 + 速度< C3*V_inf（排除已划分区域）
    mask_boundary = (mu_t > CONFIG["C2"]) & (V_mag < CONFIG["C3"] * V_mag) & \
                    ~mask_shock & ~mask_freestream
    
    # 4. 尾流区（区域2）：μt> C4 + Vx< C5*Vx_inf（排除已划分区域）
    mask_wake = (mu_t > CONFIG["C4"]) & (Vx < CONFIG["C5"] * Vx_inf) & \
                ~mask_shock & ~mask_boundary & ~mask_freestream
    
    # 5. 外部无粘区（区域3）：μt< C6（排除所有已划分区域）
    mask_inviscid = (mu_t < CONFIG["C6"]) & \
                    ~mask_shock & ~mask_boundary & ~mask_wake & ~mask_freestream
    
    # 确保区域互不相交且覆盖所有节点
    all_masks = torch.stack([mask_shock, mask_boundary, mask_wake, mask_inviscid, mask_freestream], dim=1)
    assert (all_masks.sum(dim=1) == 1).all(), "区域划分存在重叠或遗漏！"
    
    return {
        "shock": mask_shock,
        "boundary": mask_boundary,
        "wake": mask_wake,
        "inviscid": mask_inviscid,
        "freestream": mask_freestream
    }


# -------------------------- 4. 核心Loss计算（论文2.5） --------------------------
def compute_supervised_loss(Q_pred: torch.Tensor, Q_true: torch.Tensor) -> torch.Tensor:
    """
    监督损失（MSE）：论文公式 L_supervised = 1/N * sum(||Q_pred - Q_true||²)
    :param Q_pred: 预测物理量，shape=(N_nodes, 10)
    :param Q_true: 真值物理量，shape=(N_nodes, 10)
    :return: 监督损失（标量）
    """
    return F.mse_loss(Q_pred, Q_true, reduction="mean")


def compute_thermo_loss(Q_pred: torch.Tensor) -> torch.Tensor:
    """
    热力学一致性损失（论文2.5.1）：L_thermo = 1/N * sum( (P - (γ-1)/γ * ρh)² )
    :param Q_pred: 预测物理量，shape=(N_nodes, 10)
    :return: 热力学损失（标量）
    """
    P_pred = Q_pred[:, 6]    # 预测静压
    rho_pred = Q_pred[:, 7]  # 预测密度
    h_pred = Q_pred[:, 9]    # 预测静焓
    gamma = CONFIG["gamma"]
    
    # 热力学残差
    residual = P_pred - ((gamma - 1) / gamma) * rho_pred * h_pred
    return (residual ** 2).mean()


def compute_energy_loss(Q_pred: torch.Tensor, Q_inf: torch.Tensor, mask_inviscid: torch.Tensor, mask_freestream: torch.Tensor) -> torch.Tensor:
    """
    总能量守恒损失（论文2.5.2）：仅应用于外部无粘区+自由来流区
    L_energy = 1/|V3,4| * sum( (h + 0.5V² - h_total_inf)² )
    :param Q_pred: 预测物理量，shape=(N_nodes, 10)
    :param Q_inf: 来流条件，shape=(10,)
    :param mask_inviscid: 外部无粘区掩码，shape=(N_nodes,)
    :param mask_freestream: 自由来流区掩码，shape=(N_nodes,)
    :return: 能量守恒损失（标量）
    """
    # 合并无粘区+自由来流区掩码
    mask = mask_inviscid | mask_freestream
    if mask.sum() == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=Q_pred.device)
    
    # 预测总焓
    Vx, Vy, Vz = Q_pred[mask, 0], Q_pred[mask, 1], Q_pred[mask, 2]
    h_pred = Q_pred[mask, 9]
    h_total_pred = h_pred + 0.5 * (Vx**2 + Vy**2 + Vz**2)
    
    # 来流总焓
    h_inf = Q_inf[9]
    Vx_inf, Vy_inf, Vz_inf = Q_inf[0], Q_inf[1], Q_inf[2]
    h_total_inf = h_inf + 0.5 * (Vx_inf**2 + Vy_inf**2 + Vz_inf**2)
    
    # 计算损失
    residual = h_total_pred - h_total_inf
    return (residual ** 2).mean()


def compute_curl(pos: torch.Tensor, V_pred: torch.Tensor) -> torch.Tensor:
    """
    计算速度场的旋度（∇×V），用于涡量一致性损失（论文2.5.3）
    :param pos: 节点坐标，shape=(N_nodes, 3)，requires_grad=True
    :param V_pred: 预测速度场，shape=(N_nodes, 3)（Vx, Vy, Vz）
    :return: curl_V: 旋度（理论涡量），shape=(N_nodes, 3)（ωx, ωy, ωz）
    """
    N_nodes = pos.shape[0]
    curl_V = torch.zeros_like(V_pred)
    
    # 计算速度对坐标的雅可比矩阵 J[i, j, k] = ∂V_pred[i,j]/∂pos[i,k]（i=节点，j=速度分量，k=坐标分量）
    jacobian = torch.autograd.functional.jacobian(
        func=lambda x: V_pred @ torch.eye(3, device=x.device)  # 保持速度维度不变
        if x.dim() == 1 else V_pred,
        inputs=pos,
        create_graph=False  # 推理时无需创建计算图，训练时可设为True
    )
    
    # 提取偏导数并计算旋度（论文公式5）
    # ωx = ∂Vz/∂y - ∂Vy/∂z = J[:,2,1] - J[:,1,2]
    curl_V[:, 0] = jacobian[:, 2, 1] - jacobian[:, 1, 2]
    # ωy = ∂Vx/∂z - ∂Vz/∂x = J[:,0,2] - J[:,2,0]
    curl_V[:, 1] = jacobian[:, 0, 2] - jacobian[:, 2, 0]
    # ωz = ∂Vy/∂x - ∂Vx/∂y = J[:,1,0] - J[:,0,1]
    curl_V[:, 2] = jacobian[:, 1, 0] - jacobian[:, 0, 1]
    
    return curl_V


def compute_vorticity_loss(Q_pred: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    涡量一致性损失（论文2.5.3）：L_vorticity = 1/N * sum(||ω_pred - ∇×V_pred||²)
    :param Q_pred: 预测物理量，shape=(N_nodes, 10)
    :param pos: 节点坐标，shape=(N_nodes, 3)（requires_grad=True）
    :return: 涡量损失（标量）
    """
    # 预测涡量与理论涡量（旋度）
    omega_pred = Q_pred[:, 3:6]  # 模型预测的ωx, ωy, ωz
    omega_theory = compute_curl(pos, Q_pred[:, :3])  # 速度旋度计算的理论涡量
    
    # MSE损失
    return F.mse_loss(omega_pred, omega_theory, reduction="mean")


def compute_regional_losses(Q_pred: torch.Tensor, Q_inf: torch.Tensor, masks: dict) -> dict[str, torch.Tensor]:
    """
    分区损失计算（论文2.5.4）：无滑移、尾流、无粘区、自由来流区损失
    :param Q_pred: 预测物理量，shape=(N_nodes, 10)
    :param Q_inf: 来流条件，shape=(10,)
    :param masks: 区域掩码字典
    :return: 分区损失字典
    """
    regional_losses = {}
    Vx, Vy, Vz = Q_pred[:, 0], Q_pred[:, 1], Q_pred[:, 2]
    mu_t_pred = Q_pred[:, 8]
    V_mag_sq = Vx**2 + Vy**2 + Vz**2  # 速度平方（避免重复计算）
    
    # 1. 边界层区：无滑移损失 L_noslip（速度≈0）
    mask_boundary = masks["boundary"]
    if mask_boundary.sum() > 0:
        regional_losses["noslip"] = (V_mag_sq[mask_boundary]).mean()
    else:
        regional_losses["noslip"] = torch.tensor(0.0, device=Q_pred.device)
    
    # 2. 尾流区：低速+高湍流粘度损失 L_wake
    mask_wake = masks["wake"]
    if mask_wake.sum() > 0:
        wake_loss = V_mag_sq[mask_wake] + F.relu(CONFIG["C_turb"] - mu_t_pred[mask_wake])
        regional_losses["wake"] = wake_loss.mean()
    else:
        regional_losses["wake"] = torch.tensor(0.0, device=Q_pred.device)
    
    # 3. 外部无粘区：低湍流粘度损失 L_inviscid（μt≈0）
    mask_inviscid = masks["inviscid"]
    if mask_inviscid.sum() > 0:
        regional_losses["inviscid"] = (mu_t_pred[mask_inviscid] ** 2).mean()
    else:
        regional_losses["inviscid"] = torch.tensor(0.0, device=Q_pred.device)
    
    # 4. 自由来流区：物理量匹配来流损失 L_freestream
    mask_freestream = masks["freestream"]
    if mask_freestream.sum() > 0:
        regional_losses["freestream"] = F.mse_loss(Q_pred[mask_freestream], Q_inf.repeat(mask_freestream.sum(), 1), reduction="mean")
    else:
        regional_losses["freestream"] = torch.tensor(0.0, device=Q_pred.device)
    
    return regional_losses


# -------------------------- 5. 总Loss组合（论文2.5核心公式） --------------------------
def compute_total_loss(pos: torch.Tensor, Q_pred: torch.Tensor, Q_true: torch.Tensor, masks: dict) -> tuple[torch.Tensor, dict]:
    """
    计算总Loss：L_total = L_supervised + λ_phys * L_physics
    其中 L_physics = 通用物理约束损失 + 分区物理约束损失
    :param pos: 节点坐标，shape=(N_nodes, 3)
    :param Q_pred: 预测物理量，shape=(N_nodes, 10)
    :param Q_true: 真值物理量，shape=(N_nodes, 10)
    :param masks: 区域掩码字典
    :return:
        total_loss: 总损失（标量）
        loss_dict: 各分项损失字典（用于日志打印）
    """
    # 1. 计算来流条件
    Q_inf = get_free_stream_condition(Q_true, masks["freestream"])
    
    # 2. 分项损失计算
    # （1）监督损失
    L_supervised = compute_supervised_loss(Q_pred, Q_true)
    
    # （2）通用物理约束损失（全局生效）
    L_thermo = CONFIG["w_thermo"] * compute_thermo_loss(Q_pred)
    L_vorticity = CONFIG["w_vorticity"] * compute_vorticity_loss(Q_pred, pos)
    L_general_phys = L_thermo + L_vorticity
    
    # （3）分区物理约束损失（区域生效）
    regional_losses = compute_regional_losses(Q_pred, Q_inf, masks)
    L_regional_phys = (CONFIG["w_noslip"] * regional_losses["noslip"] +
                       CONFIG["w_wake"] * regional_losses["wake"] +
                       CONFIG["w_energy"] * compute_energy_loss(Q_pred, Q_inf, masks["inviscid"], masks["freestream"]) +
                       CONFIG["w_inviscid"] * regional_losses["inviscid"] +
                       CONFIG["w_freestream"] * regional_losses["freestream"])
    
    # （4）总物理约束损失
    L_physics = L_general_phys + L_regional_phys
    
    # （5）总Loss
    total_loss = L_supervised + CONFIG["lambda_phys"] * L_physics
    
    # 整理损失字典（用于调试和可视化）
    loss_dict = {
        "total_loss": total_loss.item(),
        "supervised_loss": L_supervised.item(),
        "thermo_loss": L_thermo.item(),
        "vorticity_loss": L_vorticity.item(),
        "noslip_loss": regional_losses["noslip"].item(),
        "wake_loss": regional_losses["wake"].item(),
        "energy_loss": compute_energy_loss(Q_pred, Q_inf, masks["inviscid"], masks["freestream"]).item(),
        "inviscid_loss": regional_losses["inviscid"].item(),
        "freestream_loss": regional_losses["freestream"].item()
    }
    
    return total_loss, loss_dict


# -------------------------- 6. 测试示例 --------------------------
if __name__ == "__main__":
    # 1. 构造模拟节点数据（100个节点，符合用户输入格式）
    N_nodes = 100
    node_data = np.random.rand(N_nodes, 12)  # 随机生成示例数据
    # 手动调整部分节点为自由来流区（确保区域划分有效）
    node_data[0:20, 5] = 0.0005  # 湍流粘度低（< C6=0.001）
    node_data[0:20, 6] = 101325  # 来流压力（标准大气压）
    node_data[0:20, 7] = 35000   # 来流焓
    node_data[0:20, 9:12] = 300  # 来流速度（300 m/s）
    
    # 2. 数据解析
    pos, Q_true, Q_pred = parse_node_data(node_data)
    
    # 3. 区域划分
    Q_inf_init = get_free_stream_condition(Q_true, (Q_true[:, 8] < CONFIG["C6"]) & (Q_true[:, 6] > 100000))  # 临时来流条件用于划分
    masks = compute_regional_masks(Q_true, Q_inf_init)
    
    # 4. 计算总Loss
    total_loss, loss_dict = compute_total_loss(pos, Q_pred, Q_true, masks)
    
    # 5. 打印结果
    print("="*50)
    print("各分项损失（单位：根据物理量推导，无统一量纲）：")
    for key, value in loss_dict.items():
        print(f"{key:20s}: {value:.6f}")
    print("="*50)
    print(f"总Loss: {total_loss.item():.6f}")