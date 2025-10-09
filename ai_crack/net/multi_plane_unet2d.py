# -*- coding:utf-8 -*-

''''''

import typing
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..tensor_utils import (
    get_max_slice_size,
    normalize_tensor,
    pad_image_2d,
    unpad_image_2d,
    stack_slice_with_plane,
    Slice
)


class MultiPlaneUNet2D(nn.Module):
    def __init__(self, unet2d: nn.Module, unet2d_batch_size: int = 48, fusion_tactic: str = 'mean'):
        super().__init__()

        self.unet = unet2d

        self.slice_batch_size = unet2d_batch_size

        self.unet.train()

        self.fusion_tactic = fusion_tactic

    def forward(self, x_tuple: typing.Tuple[torch.Tensor, typing.Sequence[Slice]]):
        x, slice_objs = x_tuple

        assert x.shape[0] == 1, 'Only support single batch input'

        assert x.shape[1] == 1, 'Only support single channel input'

        assert 1 == len(slice_objs), 'Only support single batch input'

        slice_objs: typing.Sequence[Slice] = slice_objs[0]

        if not hasattr(self, 'target_size'):
            self.target_size = get_max_slice_size(x)

        xy_probs, xz_probs, yz_probs = self.process_multi_plane(x)

        xy_probs_ = xy_probs.permute(1, 0, 2, 3)

        xz_probs_ = xz_probs.permute(1, 0, 2, 3)

        yz_probs_ = yz_probs.permute(1, 0, 2, 3)

        slice_outs = []

        for slice_obj in slice_objs:
            if slice_obj.plane == 'xy':
                slice_ = slice_obj.get_data(xy_probs_)

            elif slice_obj.plane == 'xz':
                slice_ = slice_obj.get_data(xz_probs_)

            elif slice_obj.plane == 'yz':
                slice_ = slice_obj.get_data(yz_probs_)

            slice_ = pad_image_2d(slice_, self.target_size)

            slice_outs.append(slice_)

        slice_outs = torch.stack(slice_outs).contiguous()

        prob_volume = self.fuse_results(xy_probs, xz_probs, yz_probs).permute(1, 0, 2, 3)

        slice_outs_fusion = []

        for slice_obj in slice_objs:
            slice_ = slice_obj.get_data(prob_volume)

            slice_ = pad_image_2d(slice_, self.target_size)

            slice_outs_fusion.append(slice_)

        slice_outs_fusion = torch.stack(slice_outs_fusion).contiguous()

        return [slice_outs, slice_outs_fusion]

    def load_state_dict(self, state_dict, strict = True, assign = False):
        try:
            return self.unet.load_state_dict(state_dict, strict = strict, assign = assign)

        except:
            return super().load_state_dict(state_dict, strict = strict, assign = assign)

    def process_multi_plane(self, image3d: torch.Tensor) -> torch.Tensor:
        ''''''

        batch, _, depth, height, width = image3d.shape

        image3d = image3d.reshape(-1, height, width).contiguous()

        xy_probs = self.process_xy_plane(image3d)

        xz_probs = self.process_xz_plane(image3d)

        yz_probs = self.process_yz_plane(image3d)

        return xy_probs, xz_probs, yz_probs

    def process_xy_plane(self, volume: torch.Tensor) -> torch.Tensor:
        """处理XY平面切片"""

        depth, height, width = volume.shape

        # [Z, num_classes, Y, X]
        prob_volume = torch.zeros(
            (depth, self.unet.out_channels, height, width),
            device = volume.device,
            dtype = torch.float32
        )

        batch_size = self.slice_batch_size

        for start_idx in range(0, depth, batch_size):
            end_idx = min(start_idx + batch_size, depth)

            slices = volume[start_idx : end_idx, :, :]

            slices = slices.unsqueeze(1)

            src_size = slices.shape[-2:]

            slices = pad_image_2d(slices, self.target_size)

            if self.unet.in_channels == 2:
                slices = stack_slice_with_plane(slices, 'xy')

            # 模型预测
            # probs = self.unet(slices.contiguous())
            slices = slices.contiguous().requires_grad_(True)

            probs = checkpoint(self.checkpoint_fn, self.unet, slices)

            probs = unpad_image_2d(probs, src_size)

            prob_volume[start_idx : end_idx, ...] = probs

            # torch.cuda.empty_cache()

        return prob_volume.contiguous()

    def process_xz_plane(self, volume: torch.Tensor) -> torch.Tensor:
        """处理XZ平面切片"""

        depth, height, width = volume.shape

        # [Y, num_classes, Z, X]
        prob_volume = torch.zeros(
            (height, self.unet.out_channels, depth, width),
            device = volume.device,
            dtype = torch.float32
        )

        batch_size = self.slice_batch_size

        for start_idx in range(0, height, batch_size):
            end_idx = min(start_idx + batch_size, height)

            slices = volume[:, start_idx : end_idx, :].permute(1, 0, 2)

            slices = slices.unsqueeze(1)

            src_size = slices.shape[-2:]

            slices = pad_image_2d(slices, self.target_size)

            if self.unet.in_channels == 2:
                slices = stack_slice_with_plane(slices, 'xz')

            # 模型预测
            # probs = self.unet(slices.contiguous())

            slices = slices.contiguous().requires_grad_(True)

            probs = checkpoint(self.checkpoint_fn, self.unet, slices)

            probs = unpad_image_2d(probs, src_size)

            prob_volume[start_idx : end_idx, ...] = probs

        return prob_volume.permute(2, 1, 0, 3).contiguous()

    def process_yz_plane(self, volume: torch.Tensor) -> torch.Tensor:
        """处理YZ平面切片"""

        depth, height, width = volume.shape

        # [X, num_classes, Z, Y]
        prob_volume = torch.zeros(
            (width, self.unet.out_channels, depth, height),
            device = volume.device,
            dtype = torch.float32
        )

        batch_size = self.slice_batch_size

        for start_idx in range(0, width, batch_size):
            end_idx = min(start_idx + batch_size, width)

            slices = volume[:, :, start_idx : end_idx].permute(2, 0, 1)

            slices = slices.unsqueeze(1)

            src_size = slices.shape[-2:]

            slices = pad_image_2d(slices, self.target_size)

            if self.unet.in_channels == 2:
                slices = stack_slice_with_plane(slices, 'yz')

            # 模型预测
            # probs = self.usnet(slices.contiguous())

            slices = slices.contiguous().requires_grad_(True)

            probs = checkpoint(self.checkpoint_fn, self.unet, slices)

            probs = unpad_image_2d(probs, src_size)

            prob_volume[start_idx : end_idx, ...] = probs

        # 创建分割结果
        return prob_volume.permute(2, 1, 3, 0).contiguous()

    def fuse_results(self, xy_probs, xz_probs, yz_probs) -> torch.Tensor:
        """融合三种方位的结果"""

        # 应用融合策略
        if self.fusion_tactic == 'vote':
            # 多数投票
            xy_mask = self.featuremap_to_mask(xy_probs)

            xz_mask = self.featuremap_to_mask(xz_probs)

            yz_mask = self.featuremap_to_mask(yz_probs)

            xy_onehot = F.one_hot(xy_mask.to(torch.long), num_classes=self.net_out_channel).permute(0, 3, 1, 2)

            xz_onehot = F.one_hot(xz_mask.to(torch.long), num_classes=self.net_out_channel).permute(0, 3, 1, 2)

            yz_onehot = F.one_hot(yz_mask.to(torch.long), num_classes=self.net_out_channel).permute(0, 3, 1, 2)

            combined = xy_onehot + xz_onehot + yz_onehot

            return combined

        elif self.fusion_tactic == 'mean':
            # 平均概率
            avg_probs = (xy_probs + xz_probs + yz_probs) / 3.0

            return avg_probs

        elif self.fusion_tactic == 'max':
            # 最大概率
            max_probs, _ = torch.stack([xy_probs, xz_probs, yz_probs]).max(dim=0)

            return max_probs

        elif self.fusion_tactic == 'wighted_mean':
            # 加权平均 - 给XY平面更高权重
            weights = torch.tensor([0.5, 0.25, 0.25], device=xy_probs.device)  # xy, xz, yz权重

            weighted_probs = weights[0] * xy_probs + weights[1] * xz_probs + weights[2] * yz_probs

            return weighted_probs

    def featuremap_to_mask(self, featuremap: torch.Tensor) -> torch.Tensor:
        return torch.argmax(featuremap, dim=1).squeeze()

    def checkpoint_fn(self, model, x):
        # x.requires_grad_(True)
        return model(x)
