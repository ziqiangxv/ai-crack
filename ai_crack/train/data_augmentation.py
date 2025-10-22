# -*- coding:utf-8 -*-

''''''

import typing
import numpy as np
import torch
from monai import transforms as T
from monai.utils import GridSampleMode, GridSamplePadMode

from ..factory import REGISTER

class ApplyToImage(object):
    def __init__(self, func: typing.Callable[[float], T.transform.Transform], *args, **kwargs) -> None:
        self.trans = func(*args, **kwargs)

    def __call__(self, data: torch.Tensor, label: str) -> typing.Tuple[torch.Tensor, str]:
        if label.lower() == 'image':
            return self.trans(data), label

        elif label.lower() == 'gt':
            return data, label

        else:
            raise ValueError(f'label {label} is not supported')

class ApplyToImageAndGT(object):
    def __init__(self, func: typing.Callable[[float], T.transform.Transform], *args, **kwargs) -> None:
        self.trans = func(*args, **kwargs)

    def __call__(self, data: torch.Tensor, label: str) -> typing.Tuple[torch.Tensor, str]:
        if label.lower() in ('image', 'gt'):
            return self.trans(data), label

        else:
            raise ValueError(f'label {label} is not supported')

RandRotate = lambda *args, **kwargs: ApplyToImageAndGT(T.RandRotate)
RandGaussianNoise = lambda *args, **kwargs: ApplyToImage(T.RandGaussianNoise)
RandGaussianSmooth = lambda *args, **kwargs: ApplyToImage(T.RandGaussianSmooth)

REGISTER('data_augmentation', 'rotate', RandRotate)
REGISTER('data_augmentation', 'gaussian_noise', RandGaussianNoise)
REGISTER('data_augmentation', 'gaussian_smooth', RandGaussianSmooth)


def get_augmentation(
    augmentation_configs: typing.Sequence[typing.Dict[str, typing.Any]]
    ) -> typing.Callable[[torch.Tensor], torch.Tensor]:
    ''''''

    l = []

    for aug in augmentation_configs:
        name = aug['name'].lower()

        if name == 'rotate':
            l.append(
                RandRotate(
                    prob=aug['prob'],
                    range_x = aug['range_x'],
                    range_y = aug['range_y'],
                    range_z = aug['range_z'],
                )
            )

        elif name == 'gaussian_noise':
            l.append(RandGaussianNoise(prob = aug['prob'], mean = aug['mean'], std = aug['std']))

        elif name == 'gaussian_smooth':
            l.append(
                RandGaussianSmooth(
                    prob = aug['prob'],
                    sigma_x = aug['sigma_x'],
                    sigma_y = aug['sigma_y'],
                    sigma_z = aug['sigma_z']
                )
            )

    return T.compose.Compose(l, map_items = False, unpack_items = True)
