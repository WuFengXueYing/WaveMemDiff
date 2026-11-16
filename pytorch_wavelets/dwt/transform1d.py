import torch.nn as nn
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch


class DWT1DForward(nn.Module):
    """ Performs a 1d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero', use_amp = False):
        super().__init__()
        self.use_amp = use_amp
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0, h1 = wave.dec_lo, wave.dec_hi
        else:
            assert len(wave) == 2
            h0, h1 = wave[0], wave[1]

        # Prepare the filters - this makes them into column filters
        filts = lowlevel.prep_filt_afb1d(h0, h1)
        self.register_buffer('h0', filts[0])
        self.register_buffer('h1', filts[1])
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, L_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients.
        """
        # 断言检查输入张量必须是三维的，维度分别为批量大小(N)、通道数(C)和长度(L)。
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        # 初始化一个空列表，用于存储每一层的高频系数（带通系数）。
        highs = []
        # 将输入张量赋值给 x0，作为第一层的输入
        x0 = x
        # 将填充模式字符串转换为整数表示，供底层函数使用
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform 循环执行J层的小波分解，J是分解层数。
        for j in range(self.J):
            # 应用一维交替滤波器组(AFB1D)进行一层小波变换，返回低频部分 x0 和高频部分 x1。
            #  x0: Tensor of shape (N, C, L') - lowpass
            #  x1: Tensor of shape (N, C, L') - highpass
            #  L'代表每层系数的长度(每层需要用L'个低频和L'个高频系数来代表上一层的输入)
            x0, x1 = lowlevel.AFB1D.apply(x0, self.h0, self.h1, mode, self.use_amp)
            # 将每次分解出来的高频系数 x1 添加到highs列表中
            highs.append(x1)
        # 回最终的低频系数 x0 和所有层级的高频系数列表 highs。
        return x0, highs


class DWT1DInverse(nn.Module):
    """ Performs a 1d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """
    def __init__(self, wave='db1', mode='zero', use_amp = False):
        super().__init__()
        self.use_amp = use_amp
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0, g1 = wave.rec_lo, wave.rec_hi
        else:
            assert len(wave) == 2
            g0, g1 = wave[0], wave[1]

        # Prepare the filters
        filts = lowlevel.prep_filt_sfb1d(g0, g1)
        self.register_buffer('g0', filts[0])
        self.register_buffer('g1', filts[1])
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, should
              match the format returned by DWT1DForward.

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, L_{in})`

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        # 解包输入系数元组，x0 是最低频的低通系数，highs 是各层高频系数的列表。
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        mode = lowlevel.mode_to_int(self.mode)
        # Do a multilevel inverse transform
        # 逆序遍历高频系数列表，从最粗尺度到最细尺度进行重构。
        for x1 in highs[::-1]:
            # 如果某层高频系数为None，则用零张量替代，实现缺失系数的处理
            if x1 is None:
                x1 = torch.zeros_like(x0)

            # 'Unpad' added signal
            # 如果低频系数长度大于高频系数长度，则去除低频系数末尾的一个元素，确保两者长度匹配。(去填充)
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., :-1]
            #   在小波逆变换中，当低频系数 x0 的长度比高频系数 x1 长1个元素时，通过去除 x0 的最后一个元素来使两者长度对齐，确保可以正确进行合成滤波器组 SFB1D 操作
            x0 = lowlevel.SFB1D.apply(x0, x1, self.g0, self.g1, mode, self.use_amp)
        return x0
