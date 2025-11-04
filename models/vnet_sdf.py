# models/vnet_sdf.py
import torch
from torch import nn

"""
VNet với 2 head:
- Head SDF: 1 kênh, Tanh -> [-1, 1]
- Head Seg: n_classes kênh (logits)
"""

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_in, n_out, normalization='none'):
        super().__init__()
        ops = []
        for i in range(n_stages):
            ch_in = n_in if i == 0 else n_out
            ops.append(nn.Conv3d(ch_in, n_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_out))
            elif normalization != 'none':
                raise ValueError(f"Unknown norm: {normalization}")
            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
        self.post = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.post(self.conv(x))


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=2, normalization='none'):
        super().__init__()
        ops = [nn.Conv3d(n_in, n_out, stride, padding=0, stride=stride)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_out))
        elif normalization != 'none':
            raise ValueError(f"Unknown norm: {normalization}")
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=2, normalization='none'):
        super().__init__()
        ops = [nn.ConvTranspose3d(n_in, n_out, stride, padding=0, stride=stride)]
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_out))
        elif normalization != 'none':
            raise ValueError(f"Unknown norm: {normalization}")
        ops.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*ops)

    def forward(self, x):
        return self.deconv(x)


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16,
                 normalization='none', has_dropout=True):
        super().__init__()
        self.has_dropout = has_dropout
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        # Encoder
        self.block1 = ConvBlock(1, n_channels, n_filters, normalization)
        self.down1  = DownsamplingConvBlock(n_filters, 2*n_filters, normalization=normalization)

        self.block2 = ConvBlock(2, 2*n_filters, 2*n_filters, normalization)
        self.down2  = DownsamplingConvBlock(2*n_filters, 4*n_filters, normalization=normalization)

        self.block3 = ConvBlock(3, 4*n_filters, 4*n_filters, normalization)
        self.down3  = DownsamplingConvBlock(4*n_filters, 8*n_filters, normalization=normalization)

        self.block4 = ConvBlock(3, 8*n_filters, 8*n_filters, normalization)
        self.down4  = DownsamplingConvBlock(8*n_filters, 16*n_filters, normalization=normalization)

        self.block5 = ConvBlock(3, 16*n_filters, 16*n_filters, normalization)

        # Decoder
        self.up5 = UpsamplingDeconvBlock(16*n_filters, 8*n_filters, normalization=normalization)
        self.block6 = ConvBlock(3, 8*n_filters, 8*n_filters, normalization)

        self.up6 = UpsamplingDeconvBlock(8*n_filters, 4*n_filters, normalization=normalization)
        self.block7 = ConvBlock(3, 4*n_filters, 4*n_filters, normalization)

        self.up7 = UpsamplingDeconvBlock(4*n_filters, 2*n_filters, normalization=normalization)
        self.block8 = ConvBlock(2, 2*n_filters, 2*n_filters, normalization)

        self.up8 = UpsamplingDeconvBlock(2*n_filters, n_filters, normalization=normalization)
        self.block9 = ConvBlock(1, n_filters, n_filters, normalization)

        # Two heads
        self.out_sdf = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0)          # 1 channel
        self.out_seg = nn.Conv3d(n_filters, n_classes, kernel_size=1, padding=0)   # n_classes
        self.tanh = nn.Tanh()

    def encoder(self, x):
        x1 = self.block1(x);  x1d = self.down1(x1)
        x2 = self.block2(x1d); x2d = self.down2(x2)
        x3 = self.block3(x2d); x3d = self.down3(x3)
        x4 = self.block4(x3d); x4d = self.down4(x4)
        x5 = self.block5(x4d)
        if self.has_dropout: x5 = self.dropout(x5)
        return x1, x2, x3, x4, x5

    def decoder(self, feats):
        x1, x2, x3, x4, x5 = feats
        u5 = self.up5(x5) + x4
        x6 = self.block6(u5)
        u6 = self.up6(x6) + x3
        x7 = self.block7(u6)
        u7 = self.up7(x7) + x2
        x8 = self.block8(u7)
        u8 = self.up8(x8) + x1
        x9 = self.block9(u8)
        if self.has_dropout: x9 = self.dropout(x9)
        sdf_pred  = self.tanh(self.out_sdf(x9))   # (B,1,D,H,W) in [-1,1]
        seg_logits= self.out_seg(x9)              # (B,C,D,H,W) logits
        return sdf_pred, seg_logits

    def forward(self, x, turnoff_drop=False):
        old = self.has_dropout
        if turnoff_drop: self.has_dropout = False
        feats = self.encoder(x)
        sdf_pred, seg_logits = self.decoder(feats)
        if turnoff_drop: self.has_dropout = old
        return sdf_pred, seg_logits


if __name__ == '__main__':
    from thop import profile, clever_format
    model = VNet(n_channels=1, n_classes=2)
    inputs = torch.randn(2, 1, 144, 144, 112)
    sdf, seg = model(inputs)
    print("sdf:", sdf.shape, "seg:", seg.shape)   # Expect (B,1,...) and (B,2,...)
    flops, params = profile(model, inputs=(inputs,))
    macs, params = clever_format([flops, params], "%.3f")
    print("FLOPs/Params:", macs, params)
