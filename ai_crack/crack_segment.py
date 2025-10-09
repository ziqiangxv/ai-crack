# -*- coding:utf-8 -*-

''''''

import typing
import torch
import os
import torch.nn.functional as F
from .net import VNet128, VNet256, UNet2D_128, UNet2D_256, MultiPlaneUNet2D
from .tensor_utils import (
    normalize_tensor,
    pad_image_2d,
    get_max_slice_size,
    stack_slice_with_plane,
    unpad_image_2d
)

class CrackSegment3D:
    ''''''

    def __init__(
        self,
        model_path: str,
        net: str,
        net_out_channel: int,
        infer_tactic: str,
        save_dir: str,
        input_image_reader: typing.Callable[[str], torch.Tensor],
        output_mask_dumper: typing.Callable[[torch.Tensor, str], None],
        element_spacing: typing.Optional[typing.Tuple[float, float, float]] = None,
        *,
        window_size: typing.Optional[typing.Tuple[int, int, int]] = None,
        overlap: typing.Optional[typing.Tuple[int, int, int]] = None,
        overlap_fusion_tactic: typing.Optional[str] = 'max',
        device: str = 'cuda:0'):
        ''''''

        assert infer_tactic in ('full', 'slide_window'), f'infer_mode {infer_tactic} not supported'

        if infer_tactic == 'slide_window':
            assert window_size is not None, 'window_size must be specified when infer_mode is slide_window'
            assert overlap is not None, 'overlap must be specified when infer_mode is slide_window'
            assert overlap_fusion_tactic in ('average', 'max'), \
                'overlap_merge_tactic must be specified "average" or "mean" when infer_mode is slide_window'
            assert len(window_size) == 3, 'window_size must be a tuple of length 3'
            assert len(overlap) == 3, 'overlap must be a tuple of length 3'

        if net == 'vnet256':
            net = VNet256(out_channel = net_out_channel).to(device)
        elif net == 'vnet128':
            net = VNet128(out_channel = net_out_channel).to(device)
        else:
            raise ValueError(f'net {net} not supported')

        net.load_state_dict(torch.load(model_path, map_location = device)['model_state_dict'])
        net.eval()

        self.net = net
        self.infer_tactic = infer_tactic
        self.device = device
        self.save_dir = save_dir
        self.window_size = window_size
        self.overlap = overlap
        self.overlap_fusion_tactic = overlap_fusion_tactic
        self.reader = input_image_reader
        self.dumper = output_mask_dumper
        self.element_spacing = element_spacing

    @torch.no_grad()
    def __call__(self, image_path: str) -> torch.Tensor:
        ''''''

        image = self.reader(image_path).to(self.device).unsqueeze(0)

        assert len(image.shape) == 5, "输入图像必须是3D张量 (D, H, W)"
        assert image.shape[0] == 1 and image.shape[1] == 1, "输入图像的第一二个维度必须是1"

        image = normalize_tensor(image)

        if self.infer_tactic == 'full':
            featuremap = self.net(image)

        elif self.infer_tactic == 'slide_window':
            featuremap = self._slide_window_infer(image)

        else:
            raise ValueError(f'infer_tactic {self.infer_tactic} not supported')

        mask = self._featuremap_to_mask(featuremap)

        spacing = self.element_spacing if self.element_spacing is not None else [1.0, 1.0, 1.0]

        self.dumper(mask.cpu(), os.path.join(self.save_dir, os.path.basename(image_path)), spacing)

        return mask


    def _slide_window_infer(self, image: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = image.shape
        win_d, win_h, win_w = self.window_size
        ovl_d, ovl_h, ovl_w = self.overlap
        assert all([win_d <= D, win_h <= H, win_w <= W]), "窗口尺寸不能大于图像尺寸"
        assert all([ovl_d < win_d, ovl_h < win_h, ovl_w < win_w]), "重叠尺寸必须小于窗口尺寸"

        # 计算步长
        step_d = win_d - self.overlap[0]
        step_h = win_h - self.overlap[1]
        step_w = win_w - self.overlap[2]

        result_shape = [B, C + 1, D, H, W]
        result = torch.zeros(result_shape, device = self.device)
        count = torch.zeros(result_shape, device = self.device)

        # 滑窗遍历
        self.net.eval()

        for z in range(0, D, step_d):
            for y in range(0, H, step_h):
                for x in range(0, W , step_w):
                    z_end = z + win_d if z + win_d <= D else D
                    y_end = y + win_h if y + win_h <= H else H
                    x_end = x + win_w if x + win_w <= W else W

                    z_start = z_end - win_d
                    y_start = y_end - win_h
                    x_start = x_end - win_w

                    slices = [slice(None), slice(None), slice(z_start, z_end), slice(y_start, y_end), slice(x_start, x_end)]

                    window = image[slices].to(self.device)

                    window_pred = self.net(window)

                    # 融合策略
                    if self.overlap_fusion_tactic == "average":
                        result[slices] += window_pred

                        count[slices] += 1

                    elif self.overlap_fusion_tactic == "max":
                        result[slices] = torch.max(result[slices], window_pred)

                    else:
                        raise ValueError("overlap_fusion_tactic must be 'average' or 'max'")

        # 平均融合的归一化处理
        if self.overlap_fusion_tactic == "average":
            result = result / count.clamp(min=1)

        return result


    def _featuremap_to_mask(self, featuremap: torch.Tensor) -> torch.Tensor:
        return torch.argmax(featuremap, dim=1).squeeze()

class CrackSegment2D:
    ''''''

    def __init__(
        self,
        model_path: str,
        net: str,
        net_in_channel: int,
        net_out_channel: int,
        plane: str,
        slice_batch_size: int,
        save_dir: str,
        input_image_reader: typing.Callable[[str], torch.Tensor],
        output_mask_dumper: typing.Callable[[torch.Tensor, str], None],
        fusion_tactic: str = 'max',
        element_spacing: typing.Optional[typing.Tuple[float, float, float]] = None,
        device: str = 'cuda:0'):
        ''''''

         # 验证方位参数
        valid_planes = ['xy', 'xz', 'yz', 'xyz']
        if plane not in valid_planes:
            raise ValueError(f"无效方位参数: {plane}. 应为: {valid_planes}")

        assert fusion_tactic in ('mean', 'max', 'vote', 'weighted_mean'), \
            'merge_tactic must be specified "max" or "mean" or "vote" or "weighted_mean"'

        if net == 'unet2d_128':
            net = UNet2D_128(in_channel = net_in_channel, out_channel = net_out_channel).to(device)

            net.load_state_dict(torch.load(model_path, map_location = device)['model_state_dict'])

        elif net == 'unet2d_256':
            net = UNet2D_256(in_channel = net_in_channel, out_channel = net_out_channel).to(device)

            net.load_state_dict(torch.load(model_path, map_location = device)['model_state_dict'])

        elif net == 'multi_plane_unet2d_256':
            net = MultiPlaneUNet2D(UNet2D_256(in_channel = net_in_channel, out_channel = net_out_channel)).to(device)

            net.load_state_dict(torch.load(model_path, map_location = device)['model_state_dict'])

            net = net.unet

        else:
            raise ValueError(f'net {net} not supported')

        net.eval()

        self.net = net
        self.device = device
        self.save_dir = save_dir
        self.reader = input_image_reader
        self.dumper = output_mask_dumper
        self.element_spacing = element_spacing
        self.orientation = plane
        self.fusion_tactic = fusion_tactic
        self.net_out_channel = net_out_channel
        self.net_in_channel = net_in_channel
        self.slice_batch_size = slice_batch_size

    @torch.no_grad()
    def __call__(self, image_path: str) -> torch.Tensor:
        ''''''

        self.image_path =  image_path[:-4]

        image3d = self.reader(image_path).to(self.device).squeeze(0) # [Z, Y, X]
        assert len(image3d.shape) == 3, "输入图像必须是3D张量 (D, H, W)"

        if not hasattr(self, 'target_size'):
            self.target_size = get_max_slice_size(image3d)

        # 归一化图像
        image3d = normalize_tensor(image3d)

        # 处理不同方位
        if self.orientation == 'xy':
            prob_volume = self._process_xy_plane(image3d)

        elif self.orientation == 'xz':
            prob_volume = self._process_xz_plane(image3d)

        elif self.orientation == 'yz':
            prob_volume = self._process_yz_plane(image3d)

        else:  # 'xyz'
            xy_probs = self._process_xy_plane(image3d)
            xz_probs = self._process_xz_plane(image3d)
            yz_probs = self._process_yz_plane(image3d)
            prob_volume = self._fuse_results(xy_probs, xz_probs, yz_probs)

        mask = self._featuremap_to_mask(prob_volume)

        self.dumper(mask.cpu(), os.path.join(self.save_dir, os.path.basename(image_path)), self.element_spacing)

        return mask

    def _process_xy_plane(self, volume: torch.Tensor) -> torch.Tensor:
        """处理XY平面切片"""

        depth, height, width = volume.shape

        # [Z, num_classes, Y, X]
        prob_volume = torch.zeros((depth, self.net_out_channel, height, width), device=self.device, dtype=torch.float32)
        batch_size = self.slice_batch_size

        for start_idx in range(0, depth, batch_size):
            end_idx = min(start_idx + batch_size, depth)

            slices = volume[start_idx : end_idx, :, :]
            slices = slices.unsqueeze(1)
            src_size = slices.shape[-2:]

            slices = pad_image_2d(slices, self.target_size)

            if self.net_in_channel == 2:
                slices = stack_slice_with_plane(slices, 'xy')

            # 模型预测
            probs = self.net(slices.contiguous())
            probs = unpad_image_2d(probs, src_size)
            prob_volume[start_idx : end_idx, ...] = probs

        return prob_volume.contiguous()

    def _process_xz_plane(self, volume: torch.Tensor) -> torch.Tensor:
        """处理XZ平面切片"""

        depth, height, width = volume.shape

        # [Y, num_classes, Z, X]
        prob_volume = torch.zeros((height, self.net_out_channel, depth, width), device=self.device, dtype=torch.float32)
        batch_size = self.slice_batch_size

        for start_idx in range(0, height, batch_size):
            end_idx = min(start_idx + batch_size, height)

            slices = volume[:, start_idx : end_idx, :].permute(1, 0, 2)
            slices = slices.unsqueeze(1)
            src_size = slices.shape[-2:]

            slices = pad_image_2d(slices, self.target_size)

            if self.net_in_channel == 2:
                slices = stack_slice_with_plane(slices, 'xz')

            # 模型预测
            probs = self.net(slices.contiguous())
            probs = unpad_image_2d(probs, src_size)
            prob_volume[start_idx : end_idx, ...] = probs

        return prob_volume.permute(2, 1, 0, 3).contiguous()

    def _process_yz_plane(self, volume: torch.Tensor) -> torch.Tensor:
        """处理YZ平面切片"""

        depth, height, width = volume.shape

        # [X, num_classes, Z, Y]
        prob_volume = torch.zeros((width, self.net_out_channel, depth, height), device=self.device, dtype=torch.float32)
        batch_size = self.slice_batch_size

        for start_idx in range(0, width, batch_size):
            end_idx = min(start_idx + batch_size, width)

            slices = volume[:, :, start_idx : end_idx].permute(2, 0, 1)
            slices = slices.unsqueeze(1)
            src_size = slices.shape[-2:]

            slices = pad_image_2d(slices, self.target_size)

            if self.net_in_channel == 2:
                slices = stack_slice_with_plane(slices, 'yz')

            # 模型预测
            probs = self.net(slices.contiguous())
            probs = unpad_image_2d(probs, src_size)
            prob_volume[start_idx : end_idx, ...] = probs

        # 创建分割结果
        return prob_volume.permute(2, 1, 3, 0).contiguous()

    def _fuse_results(self, xy_probs, xz_probs, yz_probs) -> torch.Tensor:
        """融合三种方位的结果"""

        # 应用融合策略
        if self.fusion_tactic == 'vote':
            # 多数投票
            xy_mask = self._featuremap_to_mask(xy_probs)
            xz_mask = self._featuremap_to_mask(xz_probs)
            yz_mask = self._featuremap_to_mask(yz_probs)

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

        elif self.fusion_tactic == 'wighted_mean':  # WEIGHTED_AVERAGE
            # 加权平均 - 给XY平面更高权重
            weights = torch.tensor([0.5, 0.25, 0.25], device=xy_probs.device)  # xy, xz, yz权重
            weighted_probs = (
                weights[0] * xy_probs +
                weights[1] * xz_probs +
                weights[2] * yz_probs
            )
            return weighted_probs

    def _featuremap_to_mask(self, featuremap: torch.Tensor) -> torch.Tensor:
        return torch.argmax(featuremap, dim=1).squeeze()
