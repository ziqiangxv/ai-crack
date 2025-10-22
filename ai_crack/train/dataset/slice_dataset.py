# -*- coding:utf-8 -*-

''''''

import typing
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import lru_cache

from .dataset_config import DatasetConfig
from ...tensor_utils import pad_image_2d, get_max_slice_size, normalize_tensor, stack_slice_with_plane, TupleData
from .image3d_dataset import SlideWindowImage3DDataset, RandomWindowImage3DDataset
from .image_dataset_base import ImageDatasetBase
from ...factory import REGISTER


class SliceDataset(ImageDatasetBase):
    '''3D-Image slices and 3D-GT slices which in "xy" or "yz" or "xz" plane from a 3D image.'''

    def __init__(
        self,
        config_path: str,
        reader: typing.Callable[[str], torch.Tensor],
        net_inchannel: int,
        transform: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        device: str = 'cuda:0'):
        ''''''

        super().__init__(reader, transform, normalize, device)

        self.config = DatasetConfig(config_path)

        self.net_inchannel = net_inchannel

        self._slice_nums = [len(item) for item in self.config.slices]

    def __len__(self):
        return np.sum(self._slice_nums)

    def __getitem__(self, idx):
        # 计算图像索引和切片索引
        sum_ = 0
        image_idx = 0
        slice_idx = 0

        for i, _slice_num in enumerate(self._slice_nums):
            if sum_ <= idx < sum_ + _slice_num:
                image_idx = i
                slice_idx = idx - sum_
                sum_ += _slice_num
                break


        image_path = self.config.images[image_idx]
        gt_path = self.config.gts[image_idx]

        image = self.get_data(image_path, 'image')
        gt = self.get_data(gt_path, 'gt')

        if not hasattr(self, '_max_slice_size'):
            self._max_slice_size = get_max_slice_size(image)

        slice_obj = self.config.slices[image_idx][slice_idx]
        slice_image = slice_obj.get_data(image)

        if self.net_inchannel == 2:
            slice_image = stack_slice_with_plane(slice_image, slice_obj.plane)

        slice_gt = slice_obj.get_data(gt)
        slice_image = pad_image_2d(slice_image, self._max_slice_size)
        slice_gt = pad_image_2d(slice_gt, self._max_slice_size)

        return slice_image, slice_gt


class Image3DWithGTSliceDataset(ImageDatasetBase):
    '''3D-Image and 3D-GT and slice methods which in "xy" or "yz" or "xz" plane from a 3D image.'''

    def __init__(
        self,
        config_path: str,
        reader: typing.Callable[[str], torch.Tensor],
        net_inchannel: int,
        transform: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        device: str = 'cuda:0'):
        ''''''

        super().__init__(reader, transform, normalize, device)

        self.config = DatasetConfig(config_path)

        self.net_inchannel = net_inchannel

        self._slice_nums = [len(item) for item in self.config.slices]

    def __len__(self):
        return len(self.config.images)

    def __getitem__(self, idx):
        # 读取 .mhd 图像
        image_path = self.config.images[idx]
        gt_path = self.config.gts[idx]

        image3d = self.get_data(image_path, 'image')
        gt3d = self.get_data(gt_path, 'gt')

        if not hasattr(self, '_max_slice_size'):
            self._max_slice_size = get_max_slice_size(image3d)

        gt_slices = []
        slice_objs = self.config.slices[idx]

        for slice_obj in slice_objs:
            slice_ = slice_obj.get_data(gt3d)
            slice_ = pad_image_2d(slice_, self._max_slice_size)
            gt_slices.append(slice_)

        gt_slices = torch.stack(gt_slices)# 在 depth 维度上堆叠

        return image3d, gt_slices, slice_objs

    def collate_fn(self, batch):
        '''自定义 collate_fn(处理可变长度数据)'''

        images = torch.stack([item[0] for item in batch])
        labels = torch.cat([item[1] for item in batch])
        slice_objs = [item[2] for item in batch]  # 保持 list 形式

        return TupleData((images, slice_objs)), labels


class SlideWindowImage3DWithGTSliceDataset(SlideWindowImage3DDataset):
    ''''''

    def __init__(
        self,
        config_path: str,
        window_size: typing.Tuple[int, int, int],
        window_overlap: typing.Tuple[int, int, int],
        reader: typing.Callable[[str], torch.Tensor],
        transform: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        device: str = 'cuda:0'):
        ''''''

        super().__init__(config_path, window_size, window_overlap, reader, transform, normalize, device)

        vois, slices = [], []

        image_paths, gt_paths = [], []

        for i in range(self.image_num):
            slice_objs = self.config.slices[i]

            _vois = []

            _slices_1 = []

            for voi in self.sample_vois:
                _slices_2 = []

                for slice_obj in slice_objs:
                    slice_obj_copy = None

                    if slice_obj.plane == 'xy' and voi[0][0] <= slice_obj.index <= voi[1][0]:
                        slice_obj_copy = copy.deepcopy(slice_obj)

                        slice_obj_copy.index -= voi[0][0]

                    elif slice_obj.plane == 'xz' and voi[0][1] <= slice_obj.index <= voi[1][1]:
                        slice_obj_copy = copy.deepcopy(slice_obj)

                        slice_obj_copy.index -= voi[0][1]

                    elif slice_obj.plane == 'yz' and voi[0][2] <= slice_obj.index <= voi[1][2]:
                        slice_obj_copy = copy.deepcopy(slice_obj)

                        slice_obj_copy.index -= voi[0][2]

                    if slice_obj_copy is not None:
                        _slices_2.append(slice_obj_copy)

                if len(_slices_2) > 0:
                    _vois.append(voi)

                    _slices_1.append(_slices_2)

            if len(_vois) > 0:
                vois.append(_vois)

                slices.append(_slices_1)

                image_paths.append(self.config.images[i])

                gt_paths.append(self.config.gts[i])

        assert len(vois) == len(slices) == len(image_paths) == len(gt_paths)

        for v, s in zip(vois, slices):
            assert len(v) == len(s)

        self.sample_vois = vois

        self.sample_slices = slices

        self.image_paths = image_paths

        self.gt_paths = gt_paths

        self.voi_nums = [len(_vois) for _vois in vois]

    def __len__(self):
        return np.sum(self.voi_nums)

    def __getitem__(self, idx):
        # 计算图像索引和切片索引
        sum_ = 0
        image_idx = 0
        voi_idx = 0

        for i, voi_num in enumerate(self.voi_nums):
            if sum_ <= idx < sum_ + voi_num:
                image_idx = i
                voi_idx = idx - sum_
                sum_ += voi_num
                break

        image_path = self.image_paths[image_idx]

        gt_path = self.gt_paths[image_idx]

        image3d = self.get_data(image_path, 'image')

        gt3d = self.get_data(gt_path, 'gt')

        start, end = self.sample_vois[image_idx][voi_idx]

        image_voi = image3d[..., start[0] : end[0], start[1] : end[1], start[2] : end[2]]

        gt_voi = gt3d[..., start[0] : end[0], start[1] : end[1], start[2] : end[2]]

        if not hasattr(self, '_max_slice_size'):
            self._max_slice_size = get_max_slice_size(image_voi)

        slice_objs = self.sample_slices[image_idx][voi_idx]

        gt_voi_slices = []

        for slice_obj in slice_objs:
            slice_ = slice_obj.get_data(gt_voi)

            slice_ = pad_image_2d(slice_, self._max_slice_size)

            gt_voi_slices.append(slice_)

        gt_voi_slices = torch.stack(gt_voi_slices)# 在 depth 维度上堆叠




        # i = 0
        # yz_exsits = False
        # for slice_obj in slice_objs:
        #     if slice_obj.plane == 'yz' and (gt_voi_slices[i] > 0).any():
        #         yz_exsits = True
        #         break
        #     i += 1

        # if yz_exsits:
        # import SimpleITK as sitk
        # import os
        # save_dir = '/media/gzz/D/tmp000'
        # os.makedirs(save_dir, exist_ok = True)

        # image_voi_ = sitk.GetImageFromArray(image_voi[0].cpu().numpy())
        # sitk.WriteImage(image_voi_, os.path.join(save_dir, 'image_voi.mhd'))

        # gt_voi_ = sitk.GetImageFromArray(gt_voi[0].cpu().numpy())
        # sitk.WriteImage(gt_voi_, os.path.join(save_dir, 'gt_voi.mhd'))

        # gt_voi_slices_ = sitk.GetImageFromArray(gt_voi_slices.squeeze(dim=1).cpu().numpy())
        # sitk.WriteImage(gt_voi_slices_, os.path.join(save_dir, 'gt_voi_slices.mhd'))

        # print(000)


        # return image_voi, gt_voi_slices, slice_objs
        return image_voi.to(torch.float32), gt_voi_slices, slice_objs

    def collate_fn(self, batch):
        '''自定义 collate_fn(处理可变长度数据)'''

        images = torch.stack([item[0] for item in batch])
        labels = torch.cat([item[1] for item in batch])
        slice_objs = [item[2] for item in batch]  # 保持 list 形式

        return TupleData((images, slice_objs)), labels


class RandomWindowImage3DWithGTSliceDataset(RandomWindowImage3DDataset):
    ''''''

    def __getitem__(self, idx):
        image_voi, gt_voi = super().__getitem__(idx)

        if not hasattr(self, '_max_slice_size'):
            self._max_slice_size = get_max_slice_size(image_voi)

        image_idx = idx // self.num_samples_per_image

        slice_objs = []

        while len(slice_objs) == 0:
            slice_objs = self._get_slice(self._random_window, image_idx)

        gt_voi_slices = []

        for slice_obj in slice_objs:
            slice_ = slice_obj.get_data(gt_voi)

            slice_ = pad_image_2d(slice_, self._max_slice_size)

            gt_voi_slices.append(slice_)

        gt_voi_slices = torch.stack(gt_voi_slices) # 在 depth 维度上堆叠

        return image_voi, gt_voi_slices, slice_objs

    def _get_slice(self, voi, image_idx):
        slices = []

        slice_objs = self.config.slices[image_idx]

        for slice_obj in slice_objs:
            slice_obj_copy = None

            if slice_obj.plane == 'xy' and voi[0][0] <= slice_obj.index <= voi[1][0]:
                slice_obj_copy = copy.deepcopy(slice_obj)

                slice_obj_copy.index -= voi[0][0]

            elif slice_obj.plane == 'xz' and voi[0][1] <= slice_obj.index <= voi[1][1]:
                slice_obj_copy = copy.deepcopy(slice_obj)

                slice_obj_copy.index -= voi[0][1]

            elif slice_obj.plane == 'yz' and voi[0][2] <= slice_obj.index <= voi[1][2]:
                slice_obj_copy = copy.deepcopy(slice_obj)

                slice_obj_copy.index -= voi[0][2]

            if slice_obj_copy is not None:
                slices.append(slice_obj_copy)

        return slices


REGISTER('dataset', 'slice_dataset', SliceDataset)

REGISTER('dataset', 'image3d_with_gt_slice_dataset', Image3DWithGTSliceDataset)

REGISTER('dataset', 'slide_window_image3d_with_gt_slice_dataset', SlideWindowImage3DWithGTSliceDataset)

REGISTER('dataset', 'random_window_image3d_with_gt_slice_dataset', RandomWindowImage3DWithGTSliceDataset)