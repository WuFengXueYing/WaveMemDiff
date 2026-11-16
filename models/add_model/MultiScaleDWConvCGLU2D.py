import torch
import torch.nn as nn
import torch.nn.functional as F


# 新增：多尺度2D深度卷积（适配 [B, D, C, N] → Conv2d on (C, N)）
class MultiScaleDWConv2D(nn.Module):
    def __init__(self, dim):  # dim = D (in_channels)
        super().__init__()
        # self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=1, padding=0, bias=True, groups=dim)
        # kernel_size=(3,3): 3 on C (通道), 3 on N (patch)；可改 (5,3) 偏通道多尺度
        self.dwconv_small = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=1, padding=1, bias=True, groups=dim)
        # self.dwconv_large = nn.Conv2d(dim, dim, kernel_size=(5, 3), stride=1, padding=(2, 1), bias=True,
        #                               groups=dim)  # 5 on C, 3 on N
        self.dwconv_large = nn.Conv2d(dim, dim, kernel_size=(5, 5), stride=1, padding=2, bias=True,
                                      groups=dim)  # 5 on C, 3 on N
    def forward(self, x):
        # x: [B, D, C, N]
        # x_1= self.dwconv1(x)
        x_small = self.dwconv_small(x)
        x_large = self.dwconv_large(x)
        x = (x_small + x_large) / 2  # 平均融合；或加学权重 nn.Parameter(torch.ones(2)/2)
        return x


# # 新增：2D通道自适应（SEBlock，用2D全局平均池化）
class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=4):  # channels = D
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 池化 (C,N) → 1x1
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, D, C, N]
        B, D, C, N = x.shape
        x_pool = self.global_avg_pool(x).squeeze(-1).squeeze(-1)  # [B, D] (avg over C,N)
        x_se = self.sigmoid(self.fc2(self.relu(self.fc1(x_pool)))).unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        return x * x_se  # 广播到 [B, D, C, N]
#
#
# # 新增：2D CGLU模块（核心，替换原1D）
class CGLU2D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # hidden_features = int(2 * hidden_features / 3)  # 可选压缩；或直接用 hidden_features * 2

        # fc1: 先投影 [B, D, C, N] → [B, hidden*2, C, N] (保持空间)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1, bias=True)  # 1x1 Conv 等价 Linear on D
        self.dwconv = MultiScaleDWConv2D(hidden_features)
        self.se = SEBlock2D(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)  # 1x1 输出投影
        self.drop = nn.Dropout2d(drop)  # 2D dropout (可选 nn.Dropout)

    def forward(self, x):
        # x: [B, D, C, N]  # 直接输入，无需reshape！
        # B, D, C, N = x.shape
        x_proj = self.fc1(x)  # [B, hidden*2, C, N]
        x, v = torch.chunk(x_proj, 2, dim=1)  # [B, hidden, C, N] each (沿通道D chunk)

        x_conv = self.dwconv(x)
        x_gated = self.act(x_conv) * v  # 门控：点乘，保持 [B, hidden, C, N]
        x_gated = self.se(x_gated)
        x_gated = self.drop(x_gated)
        x_out = self.fc2(x_gated)  # [B, D, C, N]
        x_out = self.drop(x_out)
        return x_out


