import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
    正弦位置编码类（SinusoidalPositionEmbeddings）
    作用：为扩散模型的 “时间步（t）” 生成位置编码，让模型感知不同加噪 / 去噪阶段的时间特征（扩散模型核心：时间步是关键条件）。
"""
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # 决定位置编码的输出维度。

    def forward(self, time):
        device = time.device  # 获取时间张量的设备（CPU/GPU）
        half_dim = self.dim // 2  # 拆分维度：一半用于sin，一半用于cos
        embeddings = math.log(10000) / (half_dim - 1)  # 位置编码的缩放系数（固定公式）
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)  # 生成指数衰减的序列
        embeddings = time[:, None] * embeddings[None, :]  # 时间步与编码序列相乘（广播：[B,1] * [1,half_dim] → [B,half_dim]）
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  # 拼接sin/cos编码 → [B,dim]
        return embeddings


class SimpleDenoiser1D(nn.Module):
    """一个极其轻量级的 1D 去噪网络
    作用：扩散模型的核心 “去噪器”，输入带噪数据 + 时间步，输出预测的噪声（训练阶段）或直接去噪（推理阶段）；
    适配 1 维序列数据（如时序异常检测的特征序列）
    """

    def __init__(self, seq_len, channels):
        super().__init__()
        self.time_mlp = nn.Sequential(
            #  时间步编码的MLP：将64维位置编码→128维→128维（GELU激活）
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )

        # # 1D卷积层：将输入通道→64→64→原通道（保持序列长度不变，padding=1）
        self.conv1 = nn.Conv1d(channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv_out = nn.Conv1d(64, channels, kernel_size=3, padding=1)
        # 时间编码投影层：将128维时间编码→64维，适配卷积层通道数
        self.time_embed_proj = nn.Linear(128, 64)

    def forward(self, x, time):
        # x: [B, L, C] → 转为 [B, C, L] 适应Conv1D（PyTorch的Conv1D输入格式：[批次, 通道, 序列长度]）
        x = x.transpose(1, 2)

        t = self.time_mlp(time)  # [B, 128]：时间步编码
        t = self.time_embed_proj(t)  # [B, 64]：投影到卷积通道维度

        x = F.gelu(self.conv1(x))  # 卷积1 + GELU激活 → [B,64,L]
        # 注入时间步信息：时间编码广播到 [B,64,L]，与卷积特征相加（特征融合）
        x = x + t.unsqueeze(-1)
        x = F.gelu(self.conv2(x))  # 卷积2 + GELU激活 → [B,64,L]

        out = self.conv_out(x)  # 输出卷积 → [B,C,L]
        # 转回 [B, L, C]（恢复原始输入格式）
        return out.transpose(1, 2)


class PurifierDiffusion(nn.Module):

    """
    作用：实现 DDPM（Denoising Diffusion Probabilistic Models）的核心逻辑，
    包含 “前向加噪”“反向去噪”“训练损失计算”“数据净化” 四大功能；在异常检测中，用于 “给脏数据加噪覆盖异常→反向去噪恢复正常”。
    """

    def __init__(self, seq_len, channels, timesteps=500):
        super().__init__()
        self.denoiser = SimpleDenoiser1D(seq_len, channels)  # 绑定去噪器
        self.timesteps = timesteps  # 扩散总步数（默认500）

        # 预计算DDPM核心参数（扩散过程的β/α/累积α等，固定公式）
        betas = torch.linspace(1e-4, 0.02, timesteps)  # β_t：前向加噪的噪声系数，从1e-4线性增加到0.02
        alphas = 1. - betas  # α_t = 1 - β_t：每步保留的干净数据比例
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # α_t累积乘积：α_1*α_2*...*α_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # 前一步的累积α（补0步为1.0）

        # 注册为buffer（不参与训练，但随模型保存/加载，且移至对应设备）
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))  # √α_cumprod：前向加噪的干净数据系数
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))  # √(1-α_cumprod)：前向加噪的噪声系数
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))  # √(1/α_t)：反向去噪的均值系数
        self.register_buffer('coeff_noise', betas / torch.sqrt(1. - alphas_cumprod))  # β_t/√(1-α_cumprod)：反向去噪的噪声系数
        self.register_buffer('posterior_variance',
                             betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))  # 后验方差：反向去噪的噪声方差

    def get_loss(self, x_0):
        """用干净数据训练预测噪声"""
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device).long()
        noise = torch.randn_like(x_0)

        x_t = (
                self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x_0 +
                self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
        )
        predicted_noise = self.denoiser(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def purify(self, x_dirty, noise_ratio=0.3):
        """
        核心净化逻辑：加一定量噪声覆盖异常突变，再用干净记忆去噪
        noise_ratio: 控制加噪深度，0.3 表示破坏 30% 并重构
        """
        B = x_dirty.shape[0]
        device = x_dirty.device
        t_start = int(self.timesteps * noise_ratio)

        # 1. 前向加噪 (盖住脏特征)
        t_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(x_dirty)
        x_t = (
                self.sqrt_alphas_cumprod[t_tensor].view(-1, 1, 1) * x_dirty +
                self.sqrt_one_minus_alphas_cumprod[t_tensor].view(-1, 1, 1) * noise
        )

        # 2. 反向去噪 (拉回正常流形)
        for i in reversed(range(t_start)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            pred_noise = self.denoiser(x_t, t)

            mean = self.sqrt_recip_alphas[t].view(-1, 1, 1) * (
                    x_t - self.coeff_noise[t].view(-1, 1, 1) * pred_noise
            )
            if i > 0:
                var = self.posterior_variance[t].view(-1, 1, 1)
                x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
            else:
                x_t = mean

        return x_t