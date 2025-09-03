def compute_total_loss(pos: torch.Tensor, Q_pred: torch.Tensor, Q_true: torch.Tensor, masks: dict) -> tuple[torch.Tensor, dict]:
    """
    计算《分区物理0.17.pdf》2.5节、《分区物理0.18.pdf》2.5节总损失：
    L_total = L_supervised + λ_phys × (L_thermo + L_vorticity + L_noslip + L_wake + L_energy + L_inviscid + L_freestream)
    """
    # 1. 计算来流条件Q∞（两文件2.3节）
    Q_inf = Q_true[masks["freestream"]].mean(dim=0) if masks["freestream"].sum() > 0 else Q_true.mean(dim=0)
    
    # 2. 分项Loss计算
    L_sup = loss_supervised(Q_pred, Q_true)
    L_thermo = loss_thermo(Q_pred) * CONFIG["w_thermo"]
    L_vort = loss_vorticity(Q_pred, Q_true, pos) * CONFIG["w_vorticity"]
    L_noslip = loss_noslip(Q_pred, masks["boundary"]) * CONFIG["w_noslip"]
    L_wake = loss_wake(Q_pred, masks["wake"]) * CONFIG["w_wake"]
    L_energy = loss_energy(Q_pred, Q_inf, masks["inviscid"], masks["freestream"]) * CONFIG["w_energy"]
    L_inviscid = loss_inviscid(Q_pred, masks["inviscid"]) * CONFIG["w_inviscid"]
    L_freestream = loss_freestream(Q_pred, Q_inf, masks["freestream"]) * CONFIG["w_freestream"]
    
    # 3. 总物理约束Loss与总Loss（两文件2.5节公式）
    L_physics = L_thermo + L_vort + L_noslip + L_wake + L_energy + L_inviscid + L_freestream
    L_total = L_sup + CONFIG["lambda_phys"] * L_physics
    
    # 4. Loss日志（标注两文件对应章节）
    loss_dict = {
        "L_total": L_total.item(),                  # 总损失（2.5节）
        "L_supervised": L_sup.item(),               # 监督损失（2.5节）
        "L_thermo": L_thermo.item(),                # 热力学损失（2.5.1节）
        "L_vorticity": L_vort.item(),               # 涡量损失（2.5.3节）
        "L_noslip": L_noslip.item(),                # 无滑移损失（2.5.4节）
        "L_wake": L_wake.item(),                    # 尾流损失（2.5.4节）
        "L_energy": L_energy.item(),                # 能量损失（2.5.2节）
        "L_inviscid": L_inviscid.item(),            # 无粘区损失（2.5.4节）
        "L_freestream": L_freestream.item()         # 来流区损失（2.5.4节）
    }
    
    return L_total, loss_dict