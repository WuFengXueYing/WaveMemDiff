import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from pytorch_wavelets import DWTForward, DWTInverse
from utils.RevIN import RevIN

class Decomposition(nn.Module):
    def __init__(self,
                 input_length = [],
                 pred_length = [],
                 wavelet_name = [],
                 level = [],
                 batch_size = [],
                 channel = [],
                 d_model = [],
                 tfactor = [],
                 dfactor = [],
                 device = [],
                 no_decomposition = [],
                 use_amp = []):
        super(Decomposition, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.device = device
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.eps = 1e-5

        self.dwt = DWT1DForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp)
        self.idwt = DWT1DInverse(wave = self.wavelet_name, use_amp = self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DInverse(wave = self.wavelet_name, use_amp = self.use_amp)
        # self.dwt = DWTForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DForward(wave = self.wavelet_name, J = self.level, use_amp = self.use_amp)
        # self.idwt = DWTInverse(wave = self.wavelet_name, use_amp = self.use_amp).cuda() if self.device.type == 'cuda' else DWT1DInverse(wave = self.wavelet_name, use_amp = self.use_amp)

        self.input_w_dim = self._dummy_forward(self.input_length) if not self.no_decomposition else [self.input_length] # length of the input seq after decompose,为了知道 resolution branch 的 input_seq 所以要提前算长度
        # self.input_w_dim = self._dummy_forward_2D(self.input_length) if not self.no_decomposition else [self.input_length] # length of the input seq after decompose,为了知道 resolution branch 的 input_seq 所以要提前算长度

        # 做异常检测，注释掉预测长度
        # self.pred_w_dim = self._dummy_forward(self.pred_length) if not self.no_decomposition else [self.pred_length] # required length of the pred seq after decom

        self.tfactor = tfactor
        self.dfactor = dfactor
        #################################
        self.affine = False
        self.rev_ins_normalization = False
        #################################

        if self.affine:
            self._init_params()
        if self.rev_ins_normalization:
            self.revin = nn.ModuleList([RevIN(self.channel) for i in range(self.level + 1)])

    def transform(self, x):
        # input: x shape: batch, channel, seq
        if not self.no_decomposition:
            yl, yh = self._wavelet_decompose(x)
        else:
            yl, yh = x, [] # no decompose: returning the same value in yl
        return yl, yh

    def inv_transform(self, yl, yh):
        if not self.no_decomposition:
            # 根据一个低频系数和多个高频系数进行逆小波重构；
            x = self._wavelet_reverse_decompose(yl, yh)
        else:
            x = yl # no decompose: returning the same value in x
        return x

    '''
        构造一个“假的”输入信号，长度 = 你设定的 input_length 或 pred_length；

        真正跑一遍 DWT，看每一层分解后各个系数序列的长度是多少；

        把这些长度保存下来，给后面每个 ResolutionBranch 用，来决定 head 的输出维度。
    '''
    def _dummy_forward(self, input_length):
        dummy_x = torch.ones((self.batch_size, self.channel, input_length)).to(self.device)
        yl, yh = self.dwt(dummy_x)
        l = []
        l.append(yl.shape[-1])
        for i in range(len(yh)):
            l.append(yh[i].shape[-1])
        return l

    def _dummy_forward_2D(self, input_length):
        dummy_x = torch.ones((self.batch_size, self.channel, input_length, 1)).to(self.device)
        yl, yh = self.dwt(dummy_x)
        l = []
        l.append(yl.shape[-1])
        for i in range(len(yh)):
            l.append(yh[i].shape[-1])
        return l

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones((self.level + 1, self.channel)))
        self.affine_bias = nn.Parameter(torch.zeros((self.level + 1, self.channel)))


    '''
        return yl(一个低通滤波器的系数，形状[B,C,L']), yh(多个高通滤波器的系数列表，每个形状[B,C,Lm],Lm是可变长度，基于每层的分解系数长度（个数）)
    '''
    def _wavelet_decompose(self, x):
        # input: x shape: batch, channel, seq

        yl, yh = self.dwt(x)

        if self.affine:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = yl * self.affine_weight[0]
            yl = yl + self.affine_bias[0]
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = yh_ * self.affine_weight[i + 1]
                yh_ = yh_ + self.affine_bias[i + 1]
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq

        if self.rev_ins_normalization:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = self.revin[0](yl, 'norm')
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = self.revin[i + 1](yh_, 'norm')
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq
        return yl, yh

    def _wavelet_reverse_decompose(self, yl, yh):
        if self.affine:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = yl - self.affine_bias[0]
            yl = yl / (self.affine_weight[0] + self.eps)
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = yh_ - self.affine_bias[i + 1]
                yh_ = yh_ / (self.affine_weight[i + 1] + self.eps)
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq

        if self.rev_ins_normalization:
            yl = yl.transpose(1, 2) # batch, seq, channel
            yl = self.revin[0](yl, 'denorm')
            yl = yl.transpose(1, 2) # batch, channel, seq
            for i in range(self.level):
                yh_ = yh[i].transpose(1, 2)  # batch, seq, channel
                yh_ = self.revin[i + 1](yh_, 'denorm')
                yh[i] = yh_.transpose(1, 2) # batch, channel, seq

        x = self.idwt((yl, yh))
        return x # shape: batch, channel, seq
