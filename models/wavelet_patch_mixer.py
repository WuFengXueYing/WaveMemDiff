import torch.nn as nn
import torch
import numpy as np
from einops import rearrange



from models.add_model.MemoryBank import MemoryBank
from models.add_model.MultiScaleDWConvCGLU2D import CGLU2D
from utils.RevIN import RevIN
from models.decomposition import Decomposition

class WPMixerCore(nn.Module):
    def __init__(self,
                 input_length = [],
                 pred_length = [],
                 wavelet_name = [],
                 level = [],
                 batch_size = [],
                 channel = [],
                 d_model = [],
                 dropout = [],
                 embedding_dropout = [],
                 tfactor = [],
                 dfactor = [],
                 device = [],
                 patch_len = [],
                 patch_stride = [],
                 no_decomposition = [],
                 use_amp = [],
                 use_kan=[],
                 prototype_num = [],
                 diffusion_step = []
                 ):

        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp
        #=======================================================
        self.use_kan = use_kan
        self.prototype_num = prototype_num
        self.diffusion_step = diffusion_step
        self.Decomposition_model = Decomposition(input_length = self.input_length,
                                        pred_length = self.pred_length,
                                        wavelet_name = self.wavelet_name,
                                        level = self.level,
                                        batch_size = self.batch_size,
                                        channel = self.channel,
                                        d_model = self.d_model,
                                        tfactor = self.tfactor,
                                        dfactor = self.dfactor,
                                        device = self.device,
                                        no_decomposition = self.no_decomposition,
                                        use_amp = self.use_amp)

        self.input_w_dim = self.Decomposition_model.input_w_dim # list of the length of the input coefficient series after decompose
        # self.pred_w_dim = self.Decomposition_model.pred_w_dim # list of the length of the predicted coefficient series（把预测长度扔进dwt中然后看看返回的最高层的低频系数和每层的高频系数个数是多少，根据这个长度，通过head进行预测映射的时候就把原来的输入seq_len对应的小波系数长度映射到预测pred_len对应的小波系数长度。再逆小波恢复到原本时域时才能是预测长度而不是输入长度）

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # (m+1) number of resolutionBranch
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq = self.input_w_dim[i],
                                                           # pred_seq = self.pred_w_dim[i],
                                                           batch_size = self.batch_size,
                                                           device=self.device,
                                                           channel = self.channel,
                                                           d_model = self.d_model,
                                                           dropout = self.dropout,
                                                           embedding_dropout = self.embedding_dropout,
                                                           tfactor = self.tfactor,
                                                           dfactor = self.dfactor,
                                                           patch_len = self.patch_len,
                                                           patch_stride = self.patch_stride,
                                                           prototype_num = self.prototype_num
                                                                ) for i in range(len(self.input_w_dim))])

        self.revin = RevIN(self.channel, eps=1e-5, affine = True, subtract_last = False)


    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : Look back window: [Batch, look_back_length, channel]

        Returns
        -------
        xT : Prediction time series: [Batch, prediction_length, output_channel]
        '''

        x = self.revin(xL, 'norm')
        x = x.transpose(1, 2) # [batch, channel, look_back_length]

        # xA: approximation coefficient series,
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series

        xA, xD = self.Decomposition_model.transform(x) # XA: [B, C, L m]  xD: [[B, C, Lm]、[B, C, Lm-1]、...[B, C, L1]
        # if self.use_kan:
        #     xB = xA
        #     xA = xA.transpose(1, 2)
        #     xA = rearrange(xA, "B L D -> (B D) L")
        #     xA_smooth = self.kan_ad(xA)
        #     xA_smooth = rearrange(xA_smooth, "(B D) L -> B L D", B=x.size(0))
        #     xA_smooth = xA_smooth.transpose(1, 2)
        #     xA_smooth = xA_smooth + xB
        #     yA = self.resolutionBranch[0](xA_smooth)
        # else:
        yA, att_A = self.resolutionBranch[0](xA)
        yD = []
        all_att = [att_A]  # 收集所有分支的权重
        for i in range(len(xD)):
            yD_i, att_D_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)
            all_att.append(att_D_i)

        all_att = torch.cat(all_att, dim=1)

        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        # 把代码从长期预测改成了异常检测后，不需要再截断最后pred_len的预测长度了，直接返回原始序列的长度
        # y = y[:, -self.pred_length:, :] # decomposition output is always even, but pred length can be odd(因为小波每层分解的下采样都是除以2，由于奇数长度要对齐，经常会在边界做 padding / 截断，所以系数长度、重构后的长度都会倾向于变成“偶数”或者“2 的幂相关的长度”。小波那条路子出来的序列长度是“偶数”（如果），但你的 pred_length 可以是奇数，所以最后的输出可以通过截断来获得指定的pred_len长度)
        xT = self.revin(y, 'denorm')

        return xT, all_att, xA, xD



class ResolutionBranch(nn.Module):
    def __init__(self,
                 input_seq = [],
                 pred_seq = [],
                 batch_size = [],
                 device = [],
                 channel = [],
                 d_model = [],
                 dropout = [],
                 embedding_dropout = [],
                 tfactor = [],
                 dfactor = [],
                 patch_len = [],
                 patch_stride = [],
                 prototype_num = []):
        super(ResolutionBranch, self).__init__()
        self.input_seq = input_seq # input_w_dim[i] 第i层系数序列长度
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.device = device
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.prototype_num = prototype_num
        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)
        self.patch_norm = nn.BatchNorm2d(self.channel)
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model) # shared among all channels
        self.mixer1 = Mixer(input_seq = self.patch_num,
                            out_seq = self.patch_num,
                            batch_size = self.batch_size,
                            channel = self.channel,
                            d_model = self.d_model,
                            dropout = self.dropout,
                            tfactor = self.tfactor,
                            dfactor = self.dfactor)
        # self.mixer2 = Mixer(input_seq = self.patch_num,
        #                     out_seq = self.patch_num,
        #                     batch_size = self.batch_size,
        #                     channel = self.channel,
        #                     d_model = self.d_model,
        #                     dropout = self.dropout,
        #                     tfactor = self.tfactor,
        #                     dfactor = self.dfactor)
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout)
        self.head = nn.Sequential(nn.Flatten(start_dim = -2 , end_dim = -1),# [B, C, patch_num * d_model]
                                  # nn.Linear(self.patch_num * self.d_model, self.pred_seq)
                                  nn.Linear(self.patch_num * self.d_model, self.input_seq),  # [B, C, Li]
                                  nn.Dropout(self.dropout)
                                  )

        self.revin = RevIN(self.channel)
        self.memory_bank = MemoryBank(n_items=prototype_num, item_dim=self.d_model)

    def forward(self, x):
        '''
        Parameters
        ----------
        x : input coefficient series: [Batch, channel, length_of_coefficient_series]

        Returns
        -------
        out : predicted coefficient series: [Batch, channel, length_of_pred_coeff_series]
        '''
        x = x.transpose(1, 2)
        x = self.revin(x, 'norm')
        x = x.transpose(1, 2)

        # 对每条系数序列分块 [B, C, Li]->[B, C, Ni(第i层的patch_num), P(patch_len)] (Li = Ni * P)
        x_patch = self.do_patching(x)
        # 进行2D批归一化(在通道维度)
        x_patch = self.patch_norm(x_patch)
        # 进行嵌入，[B, C, Ni, P]->[B, C, Ni, D] (D = d_model)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))
        out = self.mixer1(x_emb)  # [B, C, Ni, D]
        # res = out
        # out = res + self.mixer2(out) # [B, C, Ni, D]
        out = self.norm(out)


        # 在进入 self.head 之前，通过记忆库进行“特征过滤”
        # x shape: [Batch, Channel, Patch_num, d_model]
        B, C, N, D = out.shape
        out_flat = out.view(-1, D)

        # 通过记忆库重构特征
        out_mem, att_weights = self.memory_bank(out_flat)
        # att_weights = att_weights.view(B, N, -1)
        att_weights = att_weights.view(B, C*N, -1) # 26-4-10 改

        # 将过滤后的特征传给 Head 进行最后的信号重构
        # out = out.view(B, C, N, D)
        out = out_mem.view(B, C, N, D)


        # 对于预测任务 这里每条系数序列长度Li和输入的Li已经不同了，变成了pred_len所对应的系数序列长度，而不是seq_len对应的系数序列长度。对于异常检测任务则还是对应seq_len的系数序列长度
        out = self.head(out) # [B, C, Li]
        out = out.transpose(1, 2)
        out = self.revin(out, 'denorm')
        out = out.transpose(1, 2) #[B, C, Li]
        return out, att_weights
    '''
        x : input coefficient series: [Batch, channel, length_of_coefficient_series]
    '''
    def do_patching(self, x):
        x_end = x[:, :, -1:] # [B, C, 1]
        # 将最后一个元素重复 patch_stride 次，生成填充序列，确保序列长度足够进行分块操作。这样确保即使原序列长度不够，经过填充后也足以容纳至少一个完整的 patch
        x_padding = x_end.repeat(1, 1, self.patch_stride) # [B, C, patch_stride]
        x_new = torch.cat((x, x_padding), dim = -1)  # [B, C, length_of_coefficient_series + patch_stride]
        x_patch = x_new.unfold(dimension = -1, size = self.patch_len, step = self.patch_stride) # [B, C, patch_num, patch_len]
        return x_patch # [B, C, patch_num, patch_len]


class Mixer(nn.Module):
    def __init__(self,
                 input_seq = [],
                 out_seq = [],
                 batch_size = [],
                 channel = [],
                 d_model = [],
                 dropout = [],
                 tfactor = [],
                 dfactor = []):
        super(Mixer, self).__init__()
        self.input_seq = input_seq # patch_num
        self.pred_seq = out_seq  # patch_num
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor # expansion factor for patch mixer
        self.dfactor = dfactor # expansion factor for embedding mixer

        self.tMixer = TimeMixer(input_seq=self.input_seq,
                                batch_size=self.batch_size,
                                channel=self.channel,
                                pred_seq=self.pred_seq,
                                dropout=self.dropout,
                                factor=self.tfactor,
                                d_model=self.d_model)



        # self.tMixer = TokenMixer(input_seq=self.input_seq,
        #                         batch_size=self.batch_size,
        #                         channel=self.channel,
        #                         pred_seq=self.pred_seq,
        #                         dropout=self.dropout,
        #                         factor=self.tfactor,
        #                         d_model=self.d_model)
        self.dropoutLayer = nn.Dropout(self.dropout)
        self.norm1 = nn.BatchNorm2d(self.channel)
        self.norm2 = nn.BatchNorm2d(self.channel)

        self.embeddingMixer = nn.Sequential(nn.Linear(self.d_model, self.d_model * self.dfactor),
                                            nn.GELU(),
                                            nn.Dropout(self.dropout),
                                            nn.Linear(self.d_model * self.dfactor, self.d_model))


    def forward(self, x):
        '''
        Parameters
        ----------
        x : input: [Batch, Channel, Patch_number, d_model]

        Returns
        -------
        x: output: [Batch, Channel, Patch_number, d_model]

        '''
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # [B, D, C, N]
        x = self.dropoutLayer(self.tMixer(x)) # [B, D, C, N]
        x = x.permute(0, 2, 3, 1) # [B, C, N, D]
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x)) # [B, C, N, D]
        return x



class TimeMixer(nn.Module):
    def __init__(self, input_seq = [], batch_size = [], channel = [], pred_seq = [], dropout = [], factor = [], d_model = []):
        super(TimeMixer, self).__init__()
        self.input_seq = input_seq  # N = patch_num
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq  # N (same as input for non-pred)
        self.dropout = dropout
        self.factor = factor  # 用于CGLU hidden expansion
        self.d_model = d_model  # D

        # 替换原layers为CGLU1D（增强跨patch混合）
        hidden_features = int(self.d_model * self.factor)
        # self.cglu = CGLU1D(in_features=self.d_model,
        #                    hidden_features=hidden_features,
        #                    out_features=self.d_model,
        #                    drop=self.dropout)
        self.cglu = CGLU2D(in_features=self.d_model,
                           hidden_features=hidden_features,
                           out_features=self.d_model,
                           drop=self.dropout)


    def forward(self, x):
        # # x: [B, D, C, N]  # 匹配实际输入
        # B, D, C, N = x.shape
        # # Permute to [B, C, N, D] 以匹配 patching 视角
        # x_perm = x.permute(0, 2, 3, 1)  # [B, C, N, D]
        # # Reshape: [B, C, N, D] -> [B*C, N, D]
        # x_resh = x_perm.reshape(B * C, N, D)
        # # CGLU混合
        # out_resh = self.cglu(x_resh)
        # # Reshape back: [B*C, N, D] -> [B, C, N, D]
        # out_perm = out_resh.reshape(B, C, N, D)
        # # Permute back to [B, D, C, N]
        # out = out_perm.permute(0, 3, 1, 2)
        out = self.cglu(x)  # 输出同形
        return out


class TokenMixer(nn.Module):
    def __init__(self, input_seq = [], batch_size = [], channel = [], pred_seq = [], dropout = [], factor = [], d_model = []):
        super(TokenMixer, self).__init__()
        self.input_seq = input_seq  # patch_num
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq # patch_num
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.layers = nn.Sequential(nn.Linear(self.input_seq, self.pred_seq * self.factor),
                                   nn.GELU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.pred_seq * self.factor, self.pred_seq)
                                   )


    '''
         [Batch, d_model, Channel, Patch_number]
    '''
    def forward(self, x):
        x = x.transpose(1, 2) # [B, C, D, Ni]
        x = self.layers(x)
        x = x.transpose(1, 2) # [B, D, C, Ni]
        return x

