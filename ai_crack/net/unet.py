# -*- coding:utf-8 -*-

''''''

from torch.functional import F
import monai.networks.nets as mnn
from torch import nn

class OutputTransition(nn.Module):
    def __init__(self, outChans, nll = False):
        super().__init__()

        self.outChans = outChans

        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # make channels the last axis
        if len(x.shape) == 5:
            out = x.permute(0, 2, 3, 4, 1).contiguous()
        elif len(x.shape) == 4:
            out = x.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError('wrong input shape')

        # flatten
        shape = out.shape
        out = out.view(out.numel() // self.outChans, self.outChans)
        out = self.softmax(out, dim = 1)

        if len(shape) == 5:
            out = out.view(shape).permute(0, 4, 1, 2, 3)
        elif len(shape) == 4:
            out = out.view(shape).permute(0, 3, 1, 2)

        return out

class UNet(mnn.UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_transition = OutputTransition(outChans = self.out_channels)

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out = self.output_transition(out)
        return out


UNet2D_128 = lambda in_channel, out_channel: UNet(
                spatial_dims = 2,
                in_channels = in_channel,
                out_channels = out_channel,
                channels = (16, 32, 64, 128),
                strides = (2, 2, 2),
                num_res_units = 2,
            )

UNet2D_256 = lambda in_channel, out_channel: UNet(
                spatial_dims = 2,
                in_channels = in_channel,
                out_channels = out_channel,
                channels = (32, 64, 128, 256),
                strides = (2, 2, 2),
                num_res_units = 2,
            )

if __name__ == '__main__':
    from torchinfo import summary

    model = UNet2D_128(2, 2)
    print(model.out_channels)
    summary(model, input_size=(1, 1, 256, 256))

