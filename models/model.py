import torch.nn as nn

from models.wavelet_patch_mixer import WPMixerCore


class WPMixer(nn.Module):
    def __init__(self,
                 c_in=[],
                 c_out=[],
                 seq_len=[],
                 out_len=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 device=[],
                 batch_size=[],
                 tfactor=[],
                 dfactor=[],
                 wavelet=[],
                 level=[],
                 patch_len=[],
                 stride=[],
                 no_decomposition=[],
                 use_amp=[],
                 use_kan=[],
                 kan_order = [],
                 prototype_num = [],
                 diffusion_step = []
    ):
        super(WPMixer, self).__init__()
        self.pred_len = out_len
        self.channel_in = c_in
        self.channel_out = c_out
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.batch_size = batch_size  # not required now
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.wavelet = wavelet
        self.level = level
        # patch predictior
        self.actual_seq_len = seq_len
        self.no_decomposition = no_decomposition
        self.use_amp = use_amp
        self.device = device
        # add model
        self.use_kan = use_kan
        self.kan_order = kan_order
        self.prototype_num = prototype_num
        self.diffusion_step = diffusion_step
        self.wpmixerCore = WPMixerCore(input_length=self.actual_seq_len,
                                       pred_length=self.pred_len,
                                       wavelet_name=self.wavelet,
                                       level=self.level,
                                       batch_size=self.batch_size,
                                       channel=self.channel_in,
                                       d_model=self.d_model,
                                       dropout=self.dropout,
                                       embedding_dropout=self.embedding_dropout,
                                       tfactor=self.tfactor,
                                       dfactor=self.dfactor,
                                       device=self.device,
                                       patch_len=self.patch_len,
                                       patch_stride=self.stride,
                                       no_decomposition=self.no_decomposition,
                                       use_amp=self.use_amp,
                                       use_kan = self.use_kan,
                                       prototype_num = self.prototype_num,
                                       diffusion_step = self.diffusion_step
                                       )
        self.decomposition = self.wpmixerCore.Decomposition_model


    def forward(self, x):
        # Normalization
        # means = x.mean(1, keepdim=True).detach()
        # x = x - means
        # stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x /= stdev


        pred, att_weights,xA, xD = self.wpmixerCore(x)
        pred = pred[:, :, -self.channel_out:]

        # De-Normalization
        # dec_out = pred * (stdev[:, 0].unsqueeze(1).repeat(1, pred.shape[1], 1))
        # dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, pred.shape[1], 1))

        return pred, att_weights, xA, xD
        # return dec_out, att_weights, xA, xD