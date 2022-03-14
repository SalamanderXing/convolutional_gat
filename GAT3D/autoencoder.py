import torch as t
from torch import nn
import torch.nn.functional as F
import ipdb

## Encoder
def create_encoder_single_conv(in_chs, out_chs, kernel):
    assert kernel % 2 == 1
    return nn.Sequential(
        nn.Conv2d(
            in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2
        ),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(inplace=True),
    )


class EncoderInceptionModuleSignle(nn.Module):
    def __init__(self, channels):
        assert channels % 2 == 0
        super().__init__()
        # put bottle-neck layers before convolution
        bn_ch = channels // 2
        self.bottleneck = create_encoder_single_conv(channels, bn_ch, 1)
        # bn -> Conv1, 3, 5
        self.conv1 = create_encoder_single_conv(bn_ch, channels, 1)
        self.conv3 = create_encoder_single_conv(bn_ch, channels, 3)
        self.conv5 = create_encoder_single_conv(bn_ch, channels, 5)
        self.conv7 = create_encoder_single_conv(bn_ch, channels, 7)
        # pool-proj(no-bottle neck)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        # Original inception is concatenation, but use simple addition instead
        bn = self.bottleneck(x)
        out = (
            self.conv1(bn)
            + self.conv3(bn)
            + self.conv5(bn)
            + self.conv7(bn)
            + self.pool3(x)
            + self.pool5(x)
        )
        return out


class EncoderModule(nn.Module):
    def __init__(self, chs, repeat_num, use_inception):
        super().__init__()
        if use_inception:
            layers = [
                EncoderInceptionModuleSignle(chs) for i in range(repeat_num)
            ]
        else:
            layers = [
                create_encoder_single_conv(chs, chs, 3)
                for i in range(repeat_num)
            ]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, use_inception=True, repeat_per_module=1):
        super().__init__()
        # stages
        self.upch1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.stage1 = EncoderModule(32, repeat_per_module, use_inception)
        self.upch2 = self._create_downsampling_module(32, 4)
        self.stage2 = EncoderModule(64, repeat_per_module, use_inception)
        self.upch3 = self._create_downsampling_module(64, 4)
        self.stage3 = EncoderModule(128, repeat_per_module, use_inception)
        self.upch4 = self._create_downsampling_module(128, 2)
        self.stage4 = EncoderModule(256, repeat_per_module, use_inception)

    def _create_downsampling_module(self, input_channels, pooling_kenel):
        return nn.Sequential(
            nn.AvgPool2d(pooling_kenel),
            nn.Conv2d(input_channels, input_channels * 2, kernel_size=1),
            nn.BatchNorm2d(input_channels * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.stage1(self.upch1(x))
        out = self.stage2(self.upch2(out))
        out = self.stage3(self.upch3(out))
        out = self.stage4(self.upch4(out))
        out = F.avg_pool2d(out, 2)  # Global Average pooling
        return out  # out.view(-1, 256)

    ## Decoder


def create_decoder_single_conv(in_chs, out_chs, kernel):
    assert kernel % 2 == 1
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2
        ),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(inplace=True),
    )


class DecoderInceptionModuleSingle(nn.Module):
    def __init__(self, channels):
        assert channels % 2 == 0
        super().__init__()
        # put bottle-neck layers before convolution
        bn_ch = channels // 4
        self.bottleneck = create_decoder_single_conv(channels, bn_ch, 1)
        # bn -> Conv1, 3, 5
        self.conv1 = create_decoder_single_conv(bn_ch, channels, 1)
        self.conv3 = create_decoder_single_conv(bn_ch, channels, 3)
        self.conv5 = create_decoder_single_conv(bn_ch, channels, 5)
        self.conv7 = create_decoder_single_conv(bn_ch, channels, 7)
        # pool-proj(no-bottle neck)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        # Original inception is concatenation, but use simple addition instead
        bn = self.bottleneck(x)
        out = (
            self.conv1(bn)
            + self.conv3(bn)
            + self.conv5(bn)
            + self.conv7(bn)
            + self.pool3(x)
            + self.pool5(x)
        )
        return out


class DecoderModule(nn.Module):
    def __init__(self, chs, repeat_num, use_inception):
        super().__init__()
        if use_inception:
            layers = [
                DecoderInceptionModuleSingle(chs) for i in range(repeat_num)
            ]
        else:
            layers = [
                create_decoder_single_conv(chs, chs, 3)
                for i in range(repeat_num)
            ]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class Decoder(nn.Module):
    def __init__(
        self, out_channels=6, use_inception=True, repeat_per_module=1
    ):
        super().__init__()
        # stages
        self.stage1 = DecoderModule(256, repeat_per_module, use_inception)
        self.downch1 = self._create_upsampling_module(256, 2)
        self.stage2 = DecoderModule(128, repeat_per_module, use_inception)
        self.downch2 = self._create_upsampling_module(128, 4)
        self.stage3 = DecoderModule(64, repeat_per_module, use_inception)
        self.downch3 = self._create_upsampling_module(64, 4)
        self.stage4 = DecoderModule(32, repeat_per_module, use_inception)
        self.last = nn.ConvTranspose2d(32, out_channels, kernel_size=1)

    def _create_upsampling_module(self, input_channels, pooling_kenel):
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                input_channels // 2,
                kernel_size=pooling_kenel,
                stride=pooling_kenel,
            ),
            nn.BatchNorm2d(input_channels // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = F.upsample(x.reshape(x.shape[0], -1, 1, 1), scale_factor=8)
        out = self.downch1(self.stage1(out))
        out = self.downch2(self.stage2(out))
        out = self.downch3(self.stage3(out))
        out = self.stage4(out)
        return t.sigmoid(self.last(out))
