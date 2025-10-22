# -*- coding:utf-8 -*-

''''''

import typing
import torch

from ..tensor_utils import Slice, pad_image_2d, get_max_slice_size
from ..factory import REGISTER

class MultiPlaneVnet(torch.nn.Module):
    ''''''

    def __init__(self, vnet):
        super().__init__()

        self.vnet = vnet

    def forward(self, x_tuple: typing.Tuple[torch.Tensor, typing.Sequence[typing.Sequence[Slice]]]):
        x, slice_objs = x_tuple

        assert x.shape[0] == len(slice_objs), f'Wrong input shape: {x.shape} vs {len(slice_objs)}'

        if not hasattr(self, 'target_size'):
            self.target_size = get_max_slice_size(x)

        probs = self.vnet(x)

        slice_outs = []

        for i in range(x.shape[0]):
            slices = []

            for slice_objs_per_voi in slice_objs[i]:
                slice_ = slice_objs_per_voi.get_data(probs[i])

                slice_ = pad_image_2d(slice_, self.target_size)

                slices.append(slice_)

            slice_outs.append(torch.stack(slices))

        slice_outs = torch.cat(slice_outs).contiguous()

        return slice_outs

    def load_state_dict(self, state_dict, strict = True, assign = False):
        try:
            return self.vnet.load_state_dict(state_dict, strict = strict, assign = assign)

        except:
            return super().load_state_dict(state_dict, strict = strict, assign = assign)


REGISTER('net', 'multi_plane_vnet', MultiPlaneVnet)
