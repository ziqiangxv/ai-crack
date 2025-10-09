# -*- coding:utf-8 -*-

''''''

import numpy as np
import typing
import torch
import torch.nn.functional as F

class Slice(object):
    ''''''

    def __init__(self, plane: str, index: int, roi: typing.Tuple[int, int, int, int] = None):
        ''''''

        assert plane.lower() in ('xy', 'axial', 'yz', 'coronal', 'xz','sagittal'), f'Wrong plane: {plane}'

        self.plane = plane
        self.index = index
        self.roi = roi # [left, top, right, bottom]

    def get_data(self, _3d_data: torch.Tensor) -> typing.Union[np.ndarray, torch.Tensor]:
        ''''''

        if self.plane.lower() in ('xy', 'axial'):
            slice_data = _3d_data[..., self.index, :, :]

        elif self.plane.lower() in ('xz', 'coronal'):
            slice_data = _3d_data[..., :, self.index, :]

        elif self.plane.lower() in ('yz', 'sagittal'):
            slice_data = _3d_data[..., :, :, self.index]

        else:
            raise Exception(f'Wrong plane: {self.plane}')

        if self.roi is not None:
            mask = torch.ones_like(slice_data, dtype = torch.bool)
            mask[..., self.roi[0] : self.roi[2], self.roi[1] : self.roi[3]] = False
            slice_data[mask] = 0

        return slice_data

def stack_slice_with_plane(slice_data: torch.Tensor, plane: str) -> torch.Tensor:
    """
    批量预处理切片以适应模型输入

    参数:
    slices: 2D切片批次 [C, H, W] / [B, C, H, W]
    plane: 平面方向 ('xy', 'xz', 'yz')

    返回:
    torch.Tensor: 预处理后的张量 [B, C, target_H, target_W]
    """

    assert len(slice_data.shape) in (3, 4), 'Wrong input shape'

    # 添加方向通道
    if plane == 'xy':
        direction_value = 0.0
    elif plane == 'xz':
        direction_value = 0.5
    else:  # 'yz'
        direction_value = 1.0

    # 创建方向通道
    direction_channel = torch.full_like(slice_data, direction_value)

    if len(slice_data.shape) == 3:
        # 堆叠通道 [2, H, W]
        stack = torch.cat([slice_data, direction_channel], dim = 0)
    else:
        # 堆叠通道 [B, 2, H, W]
        stack = torch.cat([slice_data, direction_channel], dim = 1)

    return stack

def get_axis_by_plane(plane: str, layout: str = 'BCDHW') -> typing.Tuple[int, ...]:
    ''''''

    layout = layout.upper()
    plane = plane.upper()

    assert layout in ('BCDHW', 'BDHWC', 'DHW', 'CDHW', 'DHWC'), f'Wrong layout: {layout}'
    assert plane in ('XY', 'XZ', 'YZ'), f'Wrong plane: {plane}'

    layout = layout.replace('D', 'Z').replace('H', 'Y').replace('W', 'X')

    pos_dict = {}

    for i, c in enumerate(layout):
        pos_dict[c] = i

    res = []

    for _p in plane:
        res.append(pos_dict[_p])

    return tuple(res)

def get_max_slice_size(image: torch.Tensor) -> typing.Tuple[int, int]:
    planar_dict = {
        'xy': Slice('xy', 0),
        'yz': Slice('yz', 0),
        'xz': Slice('xz', 0),
    }

    size_1, size_2 = [], []

    for _, s in planar_dict.items():
        slice_obj = s.get_data(image)
        size_1.append(slice_obj.shape[-2])
        size_2.append(slice_obj.shape[-1])

    target_size = (np.max(size_1), np.max(size_2))

    return target_size

def resize_image_with_pad_2d(image: torch.Tensor, target_size: typing.Tuple[int, int]) -> torch.Tensor:
    """保持纵横比的填充调整"""

    if image.shape[-2:] == target_size:
        return image

    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    h, w = image.shape[-2:]
    target_h, target_w = target_size

    # 计算缩放比例
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 使用双线性插值调整大小
    resized = F.interpolate(
                image,
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            )

    # 计算填充尺寸
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    # 应用填充 (填充顺序: 左, 右, 上, 下)
    padded = F.pad(
                resized,
                [pad_left, pad_right, pad_top, pad_bottom],
                mode='constant',
                value=0
            )

    # 移除批次和通道维度 [H, W]
    if len(image.shape) == 2:
        return padded.squeeze(0).squeeze(0)
    else:
        return padded.squeeze(0)

def pad_image_2d(image: torch.Tensor, target_size: typing.Tuple[int, int]) -> torch.Tensor:
    '''不保持纵横比的填充调整, 直接在x、y尾部填充0'''

    if image.shape[-2:] == target_size:
        return image

    if len(image.shape) == 2:
        image_ = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image_ = image.unsqueeze(0)
    elif len(image.shape) == 4:
        image_ = image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    h, w = image.shape[-2:]
    target_h, target_w = target_size

    # 计算填充尺寸
    pad_top = 0
    pad_bottom = target_h - h
    pad_left = 0
    pad_right = target_w - w

    # 应用填充 (填充顺序: 左, 右, 上, 下)
    padded = F.pad(
                image_,
                [pad_left, pad_right, pad_top, pad_bottom],
                mode='constant',
                value=0
            )

    # 移除批次和通道维度 [H, W]
    if len(image.shape) == 2:
        return padded.squeeze(0).squeeze(0)
    elif len(image.shape) == 3:
        return padded.squeeze(0)
    else:
        return padded

def unpad_image_2d(image: torch.Tensor, src_size: typing.Tuple[int, int]) -> torch.Tensor:
    '''去除在x、y尾部填充的0'''

    if image.shape[-2:] == src_size:
        return image

    src_h, src_w = src_size
    unpad = torch.zeros((*image.shape[:-2], *src_size), dtype = image.dtype, device = image.device)
    unpad[...] = image[..., 0 : src_h, 0 : src_w]
    return unpad

def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    ''''''

    return (t - t.min()) / (t.max() - t.min())


class TupleData(object):
    def __init__(self, args):
        self.data = args

    def to(self, to_):
        new_data = []

        for d in self.data:
            if isinstance(d, torch.Tensor):
                new_data.append(d.to(to_))

            else:
                new_data.append(d)

        return TupleData(new_data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __setitem__(self, idx, value):
        self.data[idx] = value


if __name__ == '__main__':
    # 设置更全面的打印选项
    torch.set_printoptions(
        linewidth=150,     # 每行最大字符数
        precision=3,       # 浮点数精度
        threshold=10000,   # 显示元素总数阈值（超过则截断）
        edgeitems=3,       # 每维开头和结尾显示的元素数
        sci_mode=False     # 禁用科学计数法
    )

    t = torch.rand(3, 4)
    taget_size = (6, 6)
    t_pad = pad_image_2d(t, taget_size)

    print(t)
    print(t_pad)
