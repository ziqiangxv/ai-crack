# -*- coding:utf-8 -*-

''''''

import typing
import os
import torch
from torch.utils.data import Dataset

class _DatasetConfig(object):
    ''''''

    def __init__(self, config_path: str):
        ''''''

        self.config_path = config_path

        self.image_dir = None

        self.gt_dir = None

        self.images = []

        self.gts = []

        self.parse()

    def parse(self):
        ''''''

        kw_image_dir = 'IMAGE_DIR::'
        kw_gt_dir = 'GT_DIR::'
        kw_image = 'IMAGE::'
        kw_gt = 'GT::'

        with open(self.config_path, 'r') as file:
            lines = file.readlines()

            image_exists = False

            for line in lines:
                line = line.strip()

                if line == '' or line.startswith('#'):
                    continue

                if line.startswith(kw_image_dir):
                    self.image_dir = line[len(kw_image_dir) :].strip()

                    assert os.path.exists(self.image_dir), f'{kw_image_dir} {self.image_dir} not exists'

                elif line.startswith(kw_gt_dir):
                    self.gt_dir = line[len(kw_gt_dir) :].strip()

                    assert os.path.exists(self.gt_dir), f'{kw_gt_dir} {self.gt_dir} not exists'

                elif line.startswith(kw_image):
                    assert self.image_dir is not None, 'IMAGE_DIR is not set'

                    assert not image_exists, 'GT is not set'

                    self.images.append(os.path.join(self.image_dir, line[len(kw_image) :].strip()))

                    image_exists = True

                elif line.startswith(kw_gt):
                    assert self.gt_dir is not None, 'GT_DIR is not set'

                    assert image_exists, 'IMAGE is not set'

                    self.gts.append(os.path.join(self.gt_dir, line[len(kw_gt) :].strip()))

                    image_exists = False

                else:
                    raise Exception(f'Wrong keyword: {line}')

        assert len(self.images) == len(self.gts), 'IMAGE and GT count not match'


class Image3DDataset(Dataset):
    ''''''

    def __init__(
        self,
        config_path: str,
        reader: typing.Callable[[str], torch.Tensor],
        transform: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize = True):
        ''''''

        self.config = _DatasetConfig(config_path)
        self.transform = transform
        self.normalize = normalize
        self.reader = reader

    def __len__(self):
        return len(self.config.images)

    def __getitem__(self, idx):
        # 读取 .mhd 图像
        image_path = self.config.images[idx]
        image = self.reader(image_path)

        # 读取对应的标签图像
        label_path = self.config.gts[idx]
        gt = self.reader(label_path)

        # 应用可选的变换
        if self.transform:
            image = self.transform(image)

        if self.normalize:
            image = (image - image.min()) / (image.max() - image.min())

        return image, gt


class SlideWindowImage3DDataset(Dataset):
    ''''''

    def __init__(
        self,
        config_path: str,
        window_size: typing.Tuple[int, int, int],
        overlap: typing.Tuple[int, int, int],
        reader: typing.Callable[[str], torch.Tensor],
        transform: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize = True):
        ''''''

        self.config = _DatasetConfig(config_path)
        self.transform = transform
        self.normalize = normalize
        self.reader = reader

        stride = (
                window_size[0] - overlap[0],
                window_size[1] - overlap[1],
                window_size[2] - overlap[2],
            )

        gt = self.reader(self.config.gts[0])
        image_size = gt.shape[1:]
        del gt

        self.image_num = len(self.config.gts)
        sample_indices = []
        i_, j_, k_ = 0, 0, 0

        for i in range(0, image_size[0] - window_size[0], stride[0]):
            for j in range(0, image_size[1] - window_size[1], stride[1]):
                for k in range(0, image_size[2] - window_size[2], stride[2]):
                    sample_indices.append([(i, j, k), (i + window_size[0], j + window_size[1], k + window_size[2])])
                    i_, j_, k_ = i, j, k

        if (image_size[0] - window_size[0] - i_) / image_size[0] > 0.05 or \
            (image_size[1] - window_size[1] - j_) / image_size[1] > 0.05 or \
            (image_size[2] - window_size[2] - k_) / image_size[2] > 0.05:
            for i in range(image_size[0] - window_size[0], image_size[0], window_size[0]):
                for j in range(image_size[1] - window_size[1], image_size[1], window_size[1]):
                    for k in range(image_size[2] - window_size[2], image_size[2], window_size[2]):
                        sample_indices.append([(i, j, k), (i + window_size[0], j + window_size[1], k + window_size[2])])

        self.sample_indices = sample_indices

    def __len__(self):
        return len(self.sample_indices) * self.image_num

    def __getitem__(self, idx):
        image_idx = idx // len(self.sample_indices)
        sample_idx = idx % len(self.sample_indices)

        image_path = self.config.images[image_idx]
        gt_path = self.config.gts[image_idx]
        image = self.reader(image_path)
        gt = self.reader(gt_path)

        start, end = self.sample_indices[sample_idx]
        image = image[..., start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        gt = gt[..., start[0] : end[0], start[1] : end[1], start[2] : end[2]]

        if self.transform:
            image = self.transform(image)

        if self.normalize:
            image = (image - image.min()) / (image.max() - image.min())

        # img = image.squeeze().cpu().numpy()
        # import SimpleITK as sitk
        # img = sitk.GetImageFromArray(img)
        # sitk.WriteImage(img, '/home/xzq/dev/zuo/tmp/0.mhd')

        # gt_ = gt.squeeze().cpu().numpy()
        # gt_ = sitk.GetImageFromArray(gt_)
        # sitk.WriteImage(gt_, '/home/xzq/dev/zuo/tmp/1.mhd')

        return image, gt


class RandomWindowImage3DDataset(Dataset):
    ''''''

    def __init__(
        self,
        config_path: str,
        window_size: typing.Tuple[int, int, int],
        num_samples_per_image: int,
        reader: typing.Callable[[str], torch.Tensor],
        transform: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize = True):
        ''''''

        self.config = _DatasetConfig(config_path)
        self.transform = transform
        self.normalize = normalize
        self.reader = reader
        self.num_samples_per_image = num_samples_per_image
        self.window_size = window_size

        gt = self.reader(self.config.gts[0])
        self.image_size = gt.shape[1:]
        del gt
        self.image_num = len(self.config.gt)

    def __len__(self):
        return self.image_num * self.num_samples_per_image

    def __getitem__(self, idx):
        image_idx = idx // self.num_samples_per_image

        image_path = self.config.images[image_idx]
        gt_path = self.config.gts[image_idx]
        image = self.reader(image_path)
        gt = self.reader(gt_path)

        if self.transform:
            image = self.transform(image)

        if self.normalize:
            image = (image - image.min()) / (image.max() - image.min())

        z_size, y_size, x_size = self.image.shape
        z_max = z_size - self.window_size[0]
        y_max = y_size - self.window_size[1]
        x_max = x_size - self.window_size[2]

        # 随机选择窗口的起始位置
        z_start = torch.randint(0, z_max)
        y_start = torch.randint(0, y_max)
        x_start = torch.randint(0, x_max)

        # 提取随机窗口
        window_image = image[...,
                             z_start:z_start + self.window_size[0],
                             y_start:y_start + self.window_size[1],
                             x_start:x_start + self.window_size[2]]
        window_gt    = gt[...,
                            z_start:z_start + self.window_size[0],
                            y_start:y_start + self.window_size[1],
                            x_start:x_start + self.window_size[2]]

        return window_image, window_gt

