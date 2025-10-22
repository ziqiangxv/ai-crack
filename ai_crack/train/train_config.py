# -*- coding:utf-8 -*-

''''''

from __future__ import annotations
import typing
from dataclasses import dataclass, asdict, field
import os
import torch
import torch.nn as nn
import torch.utils.data
import toml
from dotmap import DotMap

from ..net import (
    VNet256,
    VNet128,
    VNet32,
    VNet64,
    DiceLoss,
    UNet2D_128,
    UNet2D_256,
    MultiPlaneUNet2D,
    MultiPlaneVnet,
)

from .dataset import (
    Image3DDataset,
    SlideWindowImage3DDataset,
    RandomWindowImage3DDataset,
    SliceDataset,
    Image3DWithGTSliceDataset,
    SlideWindowImage3DWithGTSliceDataset,
    RandomWindowImage3DWithGTSliceDataset,
    CachedImage3DDataset,
    CachedSlideWindowImage3DDataset,
    CachedRandomWindowImage3DDataset,
    CachedSliceDataset,
    CachedImage3DWithGTSliceDataset,
    CachedSlideWindowImage3DWithGTSliceDataset,
    CachedRandomWindowImage3DWithGTSliceDataset
)

from ..data_reader import mhd_reader
from .data_augmentation import get_augmentation


class TrainConfig:
    ''''''

    def __init__(self, config_path: str) -> None:
        with open(config_path) as f:
            config = DotMap(toml.load(f))

        self.config: DotMap = config

        self.config_path = config_path

        for k, v in self.config.env.items():
            if not hasattr(self, k):
                setattr(self, k, v)

        for k, v in self.config.training.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def dump(self, save_toml_path: str) -> None:
        ''''''

        if not save_toml_path.endswith('.toml'):
            save_toml_path += '.toml'

        import shutil

        shutil.copy(self.config_path, save_toml_path)

    @property
    def obj_net(self) -> nn.Module:
        if hasattr(self, '_obj_net'):
            return self._obj_net

        net_name = self.config.training.net
        out_channels = self.config.net.out_channels
        in_channels = self.config.net.in_channels

        if net_name == 'vnet256':
            net = VNet256(out_channel = out_channels)

        elif net_name == 'vnet128':
            net = VNet128(out_channel = out_channels)

        elif net_name == 'unet2d_256':
            net = UNet2D_256(in_channel = in_channels, out_channel = out_channels)

        elif net_name == 'unet2d_128':
            net = UNet2D_128(in_channel = in_channels, out_channel = out_channels)

        elif net_name == 'multi_plane_unet2d_256':
            net_ = UNet2D_256(in_channel = in_channels, out_channel = out_channels)
            net = MultiPlaneUNet2D(unet2d = net_)

        elif net_name == 'multi_plane_unet2d_128':
            net_ = UNet2D_128(in_channel = in_channels, out_channel = out_channels)
            net = MultiPlaneUNet2D(unet2d = net_)

        elif net_name == 'multi_plane_vnet_256':
            net = MultiPlaneVnet(vnet = VNet256(out_channel = out_channels))

        elif net_name == 'multi_plane_vnet_128':
            net = MultiPlaneVnet(vnet = VNet128(out_channel = out_channels))

        elif net_name == 'multi_plane_vnet_32':
            net = MultiPlaneVnet(vnet = VNet32(out_channel = out_channels))

        elif net_name == 'multi_plane_vnet_64':
            net = MultiPlaneVnet(vnet = VNet64(out_channel = out_channels))

        else:
            raise NotImplementedError('')

        if not self.config.training.use_amp:
            if self.config.training.dtype == 'float16':
                net = net.half()

            elif self.config.training.dtype == 'float32':
                net = net.float()

            elif self.config.training.dtype == 'bfloat16':
                net = net.bfloat16()

            elif self.config.training.dtype == 'float64':
                net = net.double()

            else:
                raise NotImplementedError('')

        init_method_map = {
            'kaiming': torch.nn.init.kaiming_normal_,
            'xavier': torch.nn.init.xavier_normal_,
        }

        net_init_method = self.config.net.init_method

        assert net_init_method in init_method_map.keys()
        init_method = init_method_map[net_init_method]

        def init(m):
            if isinstance(m, nn.Linear):
                init_method(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        net.apply(init)

        self._obj_net = net.to(self.config.env.device)

        return self._obj_net

    @property
    def load_checkpoint(self) -> str:
        checkpoint_path = ''

        workspace = self.config.env.workspace

        save_name = self.config.env.save_name

        load_checkpoint = self.config.training.load_checkpoint

        if isinstance(load_checkpoint, str) and load_checkpoint != '':
            file_dir, file_name = os.path.split(load_checkpoint)

            if file_dir == '':
                checkpoint_path = os.path.join(workspace, 'checkpoints', save_name, file_name)

            else:
                checkpoint_path = load_checkpoint

        elif isinstance(load_checkpoint, int):
            file_dir = os.path.join(workspace, 'checkpoints', save_name)

            checkpoint_path = os.path.join(file_dir, f'epoch_{load_checkpoint}.pth')

        if checkpoint_path != '':
            assert os.path.exists(checkpoint_path)

        return checkpoint_path

    @property
    def obj_loss(self) -> nn.Module:
        if hasattr(self, '_obj_loss'):
            return self._obj_loss

        loss_name = self.config.net.loss

        if loss_name == 'dice_loss':
            loss = DiceLoss(mode = 'multiclass', channel_weights = self.config.net.dice_loss.channel_weights)
        else:
            raise NotImplementedError('')

        self._obj_loss = loss.to(self.config.env.device)
        return self._obj_loss

    @property
    def obj_optimizer(self) -> torch.optim.Optimizer:
        if hasattr(self, '_obj_optimizer'):
            return self._obj_optimizer

        optimizer_name = self.config.training.optimizer

        if optimizer_name == 'adam':
            lr = self.config.optimizer.adam.lr

            self._obj_optimizer = torch.optim.Adam(self.obj_net.parameters(), lr = lr)

        elif optimizer_name == 'sgd':
            lr = self.config.optimizer.sgd.lr

            momentum = self.config.optimizer.sgd.momentum

            self._obj_optimizer = torch.optim.SGD(self.obj_net.parameters(), lr = lr, momentum = momentum)

        else:
            raise NotImplementedError('')

        return self._obj_optimizer

    @property
    def obj_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        if hasattr(self, '_obj_lr_scheduler'):
            return self._obj_lr_scheduler

        lr_scheduler_name = self.config.training.lr_scheduler.lower()

        if lr_scheduler_name == '':
            self._obj_lr_scheduler = None

        elif lr_scheduler_name == 'exponential_lr':
            gamma = self.config.optimizer.lr_scheduler.exponential_lr.gamma

            self._obj_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.obj_optimizer, gamma = gamma)

        elif lr_scheduler_name == 'step_lr':
            step_size = self.config.optimizer.lr_scheduler.step_lr.step_size

            gamma = self.config.optimizer.lr_scheduler.step_lr.gamma

            self._obj_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.obj_optimizer,
                    step_size = step_size,
                    gamma = gamma
                )

        elif lr_scheduler_name == 'multi_step_lr':
            milestones = self.config.optimizer.lr_scheduler.multi_step_lr.milestones

            gamma = self.config.optimizer.lr_scheduler.multi_step_lr.gamma

            self._obj_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.obj_optimizer,
                    milestones = milestones,
                    gamma = gamma
                )

        elif lr_scheduler_name == 'cosine_annealing_lr':
            T_max = self.config.optimizer.lr_scheduler.cosine_annealing_lr.T_max

            eta_min = self.config.optimizer.lr_scheduler.cosine_annealing_lr.eta_min

            self._obj_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.obj_optimizer,
                    T_max = T_max,
                    eta_min = eta_min
                )

        elif lr_scheduler_name == 'cosine_annealing_warm_restarts':
            T_0 = self.config.optimizer.lr_scheduler.cosine_annealing_warm_restarts.T_0

            T_mult = self.config.optimizer.lr_scheduler.cosine_annealing_warm_restarts.T_mult

            eta_min = self.config.optimizer.lr_scheduler.cosine_annealing_warm_restarts.eta_min

            self._obj_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.obj_optimizer,
                    T_0 = T_0,
                    T_mult = T_mult,
                    eta_min = eta_min
                )

        elif lr_scheduler_name == 'linear_lr':
            start_factor = self.config.optimizer.lr_scheduler.linear_lr.start_factor

            end_factor = self.config.optimizer.lr_scheduler.linear_lr.end_factor

            total_iters = self.config.optimizer.lr_scheduler.linear_lr.total_iters

            self._obj_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.obj_optimizer,
                    start_factor = start_factor,
                    end_factor = end_factor,
                    total_iters = total_iters
                )

        else:
            raise NotImplementedError('')

        return self._obj_lr_scheduler

    @property
    def obj_data_reader(self) -> typing.Callable[[str], torch.Tensor]:
        if hasattr(self, '_obj_data_reader'):
            return self._obj_data_reader

        data_reader_name = self.config.training.data_reader

        if data_reader_name == 'mhd_reader':
            self._obj_data_reader = mhd_reader
        else:
            raise NotImplementedError('')

        return self._obj_data_reader

    @property
    def obj_train_dataset(self) -> torch.utils.data.Dataset:
        if hasattr(self, '_obj_train_dataset'):
            return self._obj_train_dataset

        train_dataset_path = self.config.training.train_dataset

        in_channels = self.config.net.in_channel

        assert os.path.exists(train_dataset_path)

        dataset_tactic = self.config.training.dataset_tactic

        use_cache = self.config.training.use_cache

        cache_size = self.config.training.cache_size

        device = self.config.env.device

        if dataset_tactic == 'full':
            params = dict(
                config_path = train_dataset_path,
                reader = self.obj_data_reader,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedImage3DDataset(cache_size = cache_size, **params)

            else:
                dataset = Image3DDataset(**params)

        elif dataset_tactic == 'slide_window':
            window_size = self.config.datatset.slide_window.window_size

            window_overlap = self.config.datatset.slide_window.window_overlap

            params = dict(
                config_path = train_dataset_path,
                reader = self.obj_data_reader,
                window_size = window_size,
                window_overlap = window_overlap,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedSlideWindowImage3DDataset(cache_size = cache_size, **params)

            else:
                dataset = SlideWindowImage3DDataset(**params)

        elif dataset_tactic == 'random_window':
            window_size = self.config.datatset.random_window.window_size

            random_window_num = self.config.datatset.random_window.random_window_num

            params = dict(
                config_path = train_dataset_path,
                reader = self.obj_data_reader,
                window_size = window_size,
                num_samples_per_image = random_window_num,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedRandomWindowImage3DDataset(cache_size = cache_size, **params)

            else:
                dataset = RandomWindowImage3DDataset(**params)

        elif dataset_tactic == 'slice':
            params = dict(
                config_path = train_dataset_path,
                net_inchannel = in_channels,
                reader = self.obj_data_reader,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedSliceDataset(cache_size = cache_size, **params)

            else:
                dataset = SliceDataset(**params)

        elif dataset_tactic == 'slice_gt':
            params = dict(
                config_path = train_dataset_path,
                reader = self.obj_data_reader,
                net_inchannel = in_channels,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedImage3DWithGTSliceDataset(cache_size = cache_size, **params)

            else:
                dataset = Image3DWithGTSliceDataset(**params)

        elif dataset_tactic == 'slide_window_slice_gt':
            window_size = self.config.dataset.slide_window_slice_gt.window_size

            window_overlap = self.config.dataset.slide_window_slice_gt.window_overlap

            params = dict(
                config_path = train_dataset_path,
                reader = self.obj_data_reader,
                window_size = window_size,
                window_overlap = window_overlap,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedSlideWindowImage3DWithGTSliceDataset(cache_size = cache_size, **params)

            else:
                dataset = SlideWindowImage3DWithGTSliceDataset(**params)

        elif dataset_tactic == 'random_window_slice_gt':
            window_size = self.config.dataset.random_window_slice_gt.window_size

            num = self.config.dataset.random_window_slice_gt.random_window_num

            params = dict(
                config_path = train_dataset_path,
                window_size = window_size,
                num_samples_per_image = num,
                reader = self.obj_data_reader,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedRandomWindowImage3DWithGTSliceDataset(cache_size = cache_size, **params)

            else:
                dataset = RandomWindowImage3DWithGTSliceDataset(**params)

        else:
            raise NotImplementedError('')

        self._obj_train_dataset = dataset

        return dataset

    @property
    def obj_valid_dataset(self) -> torch.utils.data.Dataset:
        if hasattr(self, '_obj_valid_dataset'):
            return self._obj_valid_dataset

        valid_dataset_path = self.config.training.valid_dataset

        if valid_dataset_path == '':
            return None

        in_channels = self.config.net.in_channel

        assert os.path.exists(valid_dataset_path)

        dataset_tactic = self.config.training.dataset_tactic

        use_cache = self.config.training.use_cache

        cache_size = self.config.training.cache_size

        device = self.config.env.device

        if dataset_tactic == 'full':
            params = dict(
                config_path = valid_dataset_path,
                reader = self.obj_data_reader,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedImage3DDataset(cache_size = cache_size, **params)

            else:
                dataset = Image3DDataset(**params)

        elif dataset_tactic in ('slide_window', 'random_window'):
            window_size = self.config.datatset.slide_window.window_size

            window_overlap = self.config.datatset.slide_window.window_overlap

            params = dict(
                config_path = valid_dataset_path,
                reader = self.obj_data_reader,
                window_size = window_size,
                window_overlap = window_overlap,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedSlideWindowImage3DDataset(cache_size = cache_size, **params)

            else:
                dataset = SlideWindowImage3DDataset(**params)

        elif dataset_tactic == 'slice':
            params = dict(
                config_path = valid_dataset_path,
                net_inchannel = in_channels,
                reader = self.obj_data_reader,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedSliceDataset(cache_size = cache_size, **params)

            else:
                dataset = SliceDataset(**params)

        elif dataset_tactic == 'image3d_with_slice_gt':
            params = dict(
                config_path = valid_dataset_path,
                reader = self.obj_data_reader,
                net_inchannel = in_channels,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedImage3DWithGTSliceDataset(cache_size = cache_size, **params)

            else:
                dataset = Image3DWithGTSliceDataset(**params)

        elif dataset_tactic in ('slide_window_slice_gt', 'random_window_slice_gt'):
            window_size = self.config.datatset.slide_window_slice_gt.window_size

            window_overlap = self.config.datatset.slide_window_slice_gt.window_overlap

            params = dict(
                config_path = valid_dataset_path,
                reader = self.obj_data_reader,
                window_size = window_size,
                window_overlap = window_overlap,
                transform = self.obj_augmentation,
                normalize = True,
                device = device,
            )

            if use_cache:
                dataset = CachedSlideWindowImage3DWithGTSliceDataset(cache_size = cache_size, **params)

            else:
                dataset = SlideWindowImage3DWithGTSliceDataset(**params)

        else:
            raise NotImplementedError('')

        self._obj_valid_dataset = dataset

        return dataset

    @property
    def obj_augmentation(self) -> typing.Optional[typing.Sequence[typing.Callable[[torch.Tensor], torch.Tensor]]]:
        if hasattr(self, '_obj_augmentations'):
            return self._obj_augmentations

        augmentation_names = self.config.training.dataset_augmentation

        if len(augmentation_names) == 0:
            return None

        augmentations = []

        for aug_name in augmentation_names:
            if aug_name.lower() == 'rotate':
                rotate_config = self.config.dataset.augmentation.rotate

                augmentations.append(
                    dict(
                        name = 'rotate',
                        prob = rotate_config.prob,
                        range_x = rotate_config.range_x,
                        range_y = rotate_config.range_y,
                        range_z = rotate_config.range_z,
                    )
                )
            elif aug_name.lower() == 'gaussian_noise':
                gaussian_noise_config = self.config.dataset.augmentation.gaussian_noise

                augmentations.append(
                    dict(
                        name = 'gaussian_noise',
                        prob = gaussian_noise_config.prob,
                        mean = gaussian_noise_config.mean,
                        std = gaussian_noise_config.std,
                    )
                )
            elif aug_name.lower() == 'gaussian_smooth':
                gaussian_smooth_config = self.config.dataset.augmentation.gaussian_smooth

                augmentations.append(
                    dict(
                        name = 'gaussian_smooth',
                        prob = gaussian_smooth_config.prob,
                        sigma_x = gaussian_smooth_config.sigma_x,
                        sigma_y = gaussian_smooth_config.sigma_y,
                        sigma_z = gaussian_smooth_config.sigma_z,
                    )
                )

        self._obj_augmentations = get_augmentation(augmentations)
        return self._obj_augmentations
