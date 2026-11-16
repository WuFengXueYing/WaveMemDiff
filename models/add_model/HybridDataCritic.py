import torch
import torch.nn.functional as F

class HybridDataCritic:

    """
    该类的核心功能是融合低频和高频两种权重计算方式，为输入数据生成最终的权重评分（用于筛选或加权数据，比如在混合数据训练中过滤离群样本）。
    该类通过双维度权重计算实现离群样本过滤：
    低频权重：基于注意力匹配得分的全局统计分布，过滤 “整体匹配度低” 的样本；
    高频权重：基于多分辨率特征的批次内余弦共识，过滤 “特征分布异常” 的样本；
    最终权重融合两者结果，输出 0~1 的评分（得分越高，样本越 “正常”，训练时可加权保留；得分越低，样本越异常，可弱化 / 过滤）。

        args:warmup_epochs=10,预热轮数，前 10 轮不执行过滤，仅建立数据分布基准
            sigma_factor=1.5,标准差系数，用于计算离群点过滤阈值（μ - 1.5σ）
            momentum=0.9,移动平均的动量系数，用于更新全局均值 / 标准差
    """
    def __init__(self, warmup_epochs=10, sigma_factor=1.5, momentum=0.9, device='cuda'):
        self.warmup_epochs = warmup_epochs
        self.sigma_factor = sigma_factor
        self.momentum = momentum
        self.device = device
        self.running_mu = None  # 始化全局运行均值（记录低频得分的整体均值）。
        self.running_std = None # 初始化全局运行标准差（记录低频得分的整体标准差）

    #att_weights注意力权重
    def get_weights(self, xA, xD, att_weights, epoch):
        # att_weights 现在已经是完美对齐的 [B, Total_P, N_items]
        B = xA.size(0)

        # --- (1) 低频：基于记忆匹配的移动平均法 ---
        # 找到每个 Patch 的最大匹配得分，然后对所有 Patch 求平均
        scores, _ = torch.max(att_weights, dim=-1)  # [B, Total_P]
        low_freq_scores = scores.mean(dim=1)        # [B]

        # 训练轮数 < 预热轮数时，不过滤样本（权重全为 1），仅更新全局均值 / 标准差：
        if epoch < self.warmup_epochs:
            low_weights = torch.ones(B).to(self.device)
            # 预热期建立基准
            batch_mu = low_freq_scores.mean().detach()
            batch_std = low_freq_scores.std().detach()
            # 若全局均值未初始化，直接赋值为当前批次的均值 / 标准差。
            if self.running_mu is None:
                self.running_mu, self.running_std = batch_mu, batch_std
            #  若已初始化，用动量移动平均更新全局均值 / 标准差（公式：新值 = 动量 × 旧值 + (1 - 动量)× 当前批次值），保证统计量的平滑性。
            else:
                self.running_mu = self.momentum * self.running_mu + (1 - self.momentum) * batch_mu
                self.running_std = self.momentum * self.running_std + (1 - self.momentum) * batch_std
        else:
            # 正式期：过滤离群点
            threshold = self.running_mu - self.sigma_factor * self.running_std
            # 离群样本（得分远低于阈值）权重趋近于 0，正常样本权重趋近于 1。
            low_weights = torch.sigmoid(10.0 * (low_freq_scores - threshold) / (self.running_std + 1e-6))

        # --- (2) 高频：多分辨余弦共识法 ---初始化高频权重为全 1 张量（后续逐层乘性更新），高频权重的核心逻辑是基于多分辨率特征的余弦共识，筛选特征分布异常的离群样本。
        high_weights = torch.ones(B).to(self.device)

        for d_layer in xD:
            # d_layer: [B, C, L] -> 展平为 [B, C*L]
            d_flat = d_layer.view(B, -1)

            # L2归一化 + 矩阵乘法 = 余弦相似度 [B, B]
            d_norm = F.normalize(d_flat, p=2, dim=1)
            # 计算批次内样本间的余弦相似度矩阵，形状为[B, B]（sim_matrix [i][j] 表示第 i 个样本与第 j 个样本的特征相似度）。
            sim_matrix = torch.matmul(d_norm, d_norm.t())

            # 对相似度矩阵按行求均值，得到每个样本的 “共识得分”（形状 [B]），代表该样本与批次内所有样本的平均相似度（得分越低，越可能是离群点）。
            consensus = sim_matrix.mean(dim=1)

            # Min-Max 归一化提取离群点
            # 对共识得分做 Min-Max 归一化（映射到 [0,1]）
            layer_w = (consensus - consensus.min()) / (consensus.max() - consensus.min() + 1e-6)
            high_weights *= layer_w

        # --- (3) 权重融合 ---
        final_weights = low_weights * high_weights
        return final_weights.detach()