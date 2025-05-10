# -*- coding:utf-8 -*-

''''''

import typing
import torch
import os
from .net import VNet128, VNet256

class CrackSegment:
    ''''''

    def __init__(
        self,
        model_path: str,
        net: str,
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
            net = VNet256().to(device)

        elif net == 'vnet128':
            net = VNet128().to(device)

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

        assert len(image.shape) == 5, "输入图像必须是5D张量 (D, H, W)"
        assert image.shape[0] == 1 and image.shape[1] == 1, "输入图像的第一二个维度必须是1"

        image = (image - image.min()) / (image.max() - image.min())

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

