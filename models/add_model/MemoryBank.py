import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    def __init__(self, n_items,                                                                                 item_dim):
        super(MemoryBank, self).__init__()
        self.n_items = n_items  # 记忆条目的数量 (如 500)
        self.item_dim = item_dim  # 记忆向量的维度 (对应 d_model)

        # 定义记忆库矩阵 [N, D]
        self.memory = nn.Parameter(torch.randn(n_items, item_dim))
        nn.init.kaiming_uniform_(self.memory)

    def forward(self, x):
        """
        x: [Batch * Patch, d_model] 或 [Batch, Patch, d_model]
        """
        # 1. 计算输入向量与记忆库中所有条目的余弦相似度
        # x_norm: [B*P, D], mem_norm: [N, D]
        x_reshaped = x.view(-1, self.item_dim)

        # 计算相似度得分 (Query-Key mechanism)

        logit = torch.matmul(x_reshaped, self.memory.t())  # [B*P, N]
        att_weight = F.softmax(logit, dim=1)  # 得到分配权重

        # 2. 选取 Top-K 记忆或直接加权求和（建议使用 Soft-addressing）
        # 这里可以选择引入稀疏性约束，防止模型使用过多的记忆条目
        read_out = torch.matmul(att_weight, self.memory)  # [B*P, D]

        # 3. 还原形状
        read_out = read_out.view(x.shape)

        return read_out, att_weight