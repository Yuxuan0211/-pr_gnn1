预训练的核心是让模型先学习流场的共性物理规律（如流体连续性方程、动量方程）和节点区域特征（如边界、尾流、来流区的差异），再通过微调适配具体马赫数，避免直接训练时模型 “从零学起”。
在原有区域预训练前，增加一轮无监督预训练，让模型先掌握流体的基本物理约束，减少后续监督训练的收敛难度：
针对 “不同来流马赫数” 需求，预训练阶段引入多马赫数数据（而非单马赫数），让模型学习马赫数对流场的影响规律：
预训练数据加载：加载 3-10 组不同马赫数的数据，按批次混合输入模型，让预训练过程覆盖多工况：根据区域节点数量调整轮次（节点多的区域多训练，节点少的区域少训练）：
根据区域节点数量调整轮次（节点多的区域多训练，节点少的区域少训练）：对物理约束强的区域（如边界层、尾流）增加损失权重，让模型优先学习关键区域特征：
接下来是微调阶段
使用 邻居采样（Neighbor Sampling） 技术（PyTorch Geometric 原生支持），每轮只对部分节点的邻居进行采样，而非处理全图邻居：
混合精度训练：用 FP16 精度计算（GPU 显存占用减少 50%，速度提升 20-30%），PyTorch 原生支持：用 PyTorch Geometric 的SparseTensor存储邻接矩阵（内存占用减少 80%，消息传递更快）：
替换优化器：用AdamW（带权重衰减）替代Adam，减少过拟合，收敛更稳定：学习率预热 + 余弦衰减：替代原有ReduceLROnPlateau，初期用小学习率预热（避免梯度爆炸），后期余弦衰减（精细调整）：
当验证损失连续多轮不下降时停止训练，节省时间：
不同物理量（如速度、压力、密度）的尺度差异极大（如速度 1e3、压力 1e5），会导致梯度震荡，用按区域归一化（不同区域的特征尺度不同）优化：
归一化方法# dataset.py 中新增区域归一化
def normalize_features_by_region(self, x, region_mask):
    """按区域归一化特征（每个区域单独计算均值和标准差）"""
    x_norm = x.clone()
    for region_id in range(5):
        region_mask_bool = (region_mask == region_id)
        if region_mask_bool.sum() == 0:
            continue
        # 计算该区域的均值和标准差
        region_x = x[region_mask_bool]
        mean = region_x.mean(dim=0, keepdim=True)
        std = region_x.std(dim=0, keepdim=True) + 1e-8  # 避免除零
        # 归一化
        x_norm[region_mask_bool] = (region_x - mean) / std
    return x_norm, mean, std  # 保存均值/标准差，用于后续反归一化预测结果