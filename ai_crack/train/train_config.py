# -*- coding:utf-8 -*-

''''''

from __future__ import annotations
import typing
from dataclasses import dataclass, asdict
import os
import torch
import torch.nn as nn
import torch.utils.data
import toml
from ..net import VNet256, VNet128, DiceLoss
from .dataset import Image3DDataset, SlideWindowImage3DDataset, RandomWindowImage3DDataset
from ..data_reader import mhd_reader

@dataclass
class TrainConfig:
    ''''''

    workspace: str = '/home/gzz/dev/train_workspace'

    device: str = 'cuda:0'

    save_name: str = None

    net: str = 'vnet128'

    net_init_method: str = 'kaiming' # xavier

    train_tactic: str = 'slide_window' # full, random_window

    window_size: typing.Tuple[int, int, int] = (128, 196, 196) # slide_window, random_window 使用

    slide_window_overlap: typing.Tuple[int, int, int] = (16, 32, 32) # slide_window 使用

    random_window_num: int = 10 # random_window 使用

    epochs: int = 200

    batch_size: int = 2

    load_checkpoint: int = None

    save_checkpoint: int = 1

    train_dataset: str = '/home/gzz/dev/data/train_dataset_config.txt'

    valid_dataset: str = '/home/gzz/dev/data/validate_dataset_config.txt'

    data_reader: str = 'mhd_reader'

    lr: float = 1e-3

    optimizer: str = 'adam'

    lr_scheduler: str = 'ExponentialLR' # StepLR

    loss: str = 'dice_loss' # TverskyLoss

    use_focal_loss: bool = False


    def dump(self, save_toml: str) -> None:
        ''''''

        data_dict = asdict(self)
        toml_str = toml.dumps(data_dict)
        toml_str = "\n\n".join([line for line in toml_str.splitlines()])

        if not save_toml.endswith('.toml'):
            save_toml += '.toml'

        with open(save_toml, 'w', encoding='utf-8') as file:
            file.write(toml_str)

    @staticmethod
    def load(toml_path: str) -> TrainConfig:
        ''''''

        with open(toml_path, 'r', encoding='utf-8') as file:
            toml_str = file.read()

        data = toml.loads(toml_str)
        return TrainConfig(**data)

    @property
    def obj_net(self) -> nn.Module:
        if hasattr(self, '_obj_net'):
            return self._obj_net

        if self.net == 'vnet256':
            net = VNet256()
        elif self.net == 'vnet128':
            net = VNet128()
        else:
            raise NotImplementedError('')

        init_method_map = {
            'kaiming': torch.nn.init.kaiming_normal_,
            'xavier': torch.nn.init.xavier_normal_,
        }

        assert self.net_init_method in init_method_map.keys()
        init_method = init_method_map[self.net_init_method]

        def init(m):
            if isinstance(m, nn.Linear):
                init_method(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        net.apply(init)
        self._obj_net = net.to(self.device)
        return self._obj_net

    @property
    def obj_loss(self) -> nn.Module:
        if hasattr(self, '_obj_loss'):
            return self._obj_loss

        if self.loss == 'dice_loss':
            loss = DiceLoss(mode = 'multiclass')
        else:
            raise NotImplementedError('')

        self._obj_loss = loss.to(self.device)
        return self._obj_loss

    @property
    def obj_optimizer(self) -> torch.optim.Optimizer:
        if hasattr(self, '_obj_optimizer'):
            return self._obj_optimizer

        if self.optimizer == 'adam':
            self._obj_optimizer = torch.optim.Adam(self.obj_net.parameters(), lr = self.lr)
        elif self.optimizer == 'sgd':
            self._obj_optimizer = torch.optim.SGD(self.obj_net.parameters(), lr = self.lr)
        else:
            raise NotImplementedError('')

        return self._obj_optimizer

    @property
    def obj_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        if hasattr(self, '_obj_lr_scheduler'):
            return self._obj_lr_scheduler

        if self.lr_scheduler == 'ExponentialLR':
            self._obj_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.obj_optimizer, gamma = 0.99)
        else:
            raise NotImplementedError('')

        return self._obj_lr_scheduler

    @property
    def obj_data_reader(self) -> typing.Callable[[str], torch.Tensor]:
        if hasattr(self, '_obj_data_reader'):
            return self._obj_data_reader

        if self.data_reader == 'mhd_reader':
            self._obj_data_reader = mhd_reader
        else:
            raise NotImplementedError('')

        return self._obj_data_reader

    @property
    def obj_train_dataset(self) -> torch.utils.data.Dataset:
        if hasattr(self, '_obj_train_dataset'):
            return self._obj_train_dataset

        assert os.path.exists(self.train_dataset)

        if self.train_tactic == 'full':
            self._obj_train_dataset = Image3DDataset(self.train_dataset, self.obj_data_reader)

        elif self.train_tactic == 'slide_window':
            self._obj_train_dataset = SlideWindowImage3DDataset(
                config_path = self.train_dataset,
                reader = self.obj_data_reader,
                window_size = self.window_size,
                overlap = self.slide_window_overlap,
            )

        elif self.train_tactic == 'random_window':
            self._obj_train_dataset = RandomWindowImage3DDataset(
                config_path = self.train_dataset,
                reader = self.obj_data_reader,
                window_size = self.window_size,
                num_samples_per_image = self.random_window_num,
            )

        else:
            raise NotImplementedError('')

        return self._obj_train_dataset

    @property
    def obj_valid_dataset(self) -> torch.utils.data.Dataset:
        if hasattr(self, '_obj_valid_dataset'):
            return self._obj_valid_dataset

        assert os.path.exists(self.valid_dataset)

        if self.train_tactic == 'full':
            self._obj_valid_dataset = Image3DDataset(self.valid_dataset, self.obj_data_reader)

        elif self.train_tactic in ('slide_window', 'random_window'):
            self._obj_valid_dataset = SlideWindowImage3DDataset(
                config_path = self.valid_dataset,
                reader = self.obj_data_reader,
                window_size = self.window_size,
                overlap = self.slide_window_overlap,
            )

        else:
            raise NotImplementedError('')

        return self._obj_valid_dataset

