# -*- coding:utf-8 -*-

''''''

import torch
import torch.nn as nn
import os.path as osp
import os
import numpy as np

from typing import Optional, Literal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..utils import color
from ..logger import logger
from .train_config import TrainConfig


class Trainer:
    """"""

    def __init__(self, config: TrainConfig) -> None:
        ''''''

        self._init(
            model = config.obj_net,
            optimizer = config.obj_optimizer,
            criterion = config.obj_loss,
            train_dataset = config.obj_train_dataset,
            valid_dataset = config.obj_valid_dataset,
            lr_scheduler = config.obj_lr_scheduler,
            device = config.device,
            epochs = config.epochs,
            batch_size = config.batch_size,
            load_checkpoint = config.load_checkpoint,
            save_checkpoint = config.save_checkpoint,
            save_name = config.save_name,
            workspace = config.workspace,
        )

    def _init(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        save_name: str,
        workspace: str,
        *,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
        lr_scheduler = None,
        device: str = 'cuda:0',
        epochs: int = 200,
        batch_size: int = 128,
        load_checkpoint :Optional[int] = None,
        save_checkpoint :Optional[int] = None,
    ) -> None:
        """"""

        self.workspace = workspace or osp.dirname(osp.dirname(osp.abspath(__file__)))

        self.epochs = epochs

        self.batch_size = batch_size

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        self.device = torch.device(device)

        self.model = model

        if load_checkpoint is not None:
            self._load_checkpoint(load_checkpoint)

        self.model = self.model.to(self.device)

        self.optimizer = optimizer

        self.criterion = criterion

        self.lr_scheduler = lr_scheduler

        self.save_checkpoint = save_checkpoint if save_checkpoint is not None else 10

        os.makedirs(self.workspace, exist_ok = True)

        curves_dir = osp.join(self.workspace, 'curves')
        os.makedirs(curves_dir, exist_ok = True)
        self.curve_writer = SummaryWriter(log_dir = osp.join(curves_dir, save_name))

        log_dir = osp.join(self.workspace, 'logs')
        os.makedirs(log_dir, exist_ok = True)
        self.logger_path = osp.join(log_dir, f"{save_name}.log")

        self.checkpoint_dir = osp.join(self.workspace, 'checkpoints', save_name)
        os.makedirs(self.checkpoint_dir, exist_ok = True)

        self.logger = logger(self.logger_path)


    def train(self, epochs: int = None, epoch_prints: int = None) -> None:
        """"""

        if epochs is not None:
            self.epochs = epochs

        if epoch_prints is None:
            epoch_prints = 4

        print_stride = len(self.train_dataloader) // epoch_prints

        for epoch in range(self.epochs):
            self.model.train()

            loss_sum = 0

            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.device), target.to(self.device)

                loss = self._train(data, target)

                loss_sum += loss.item()

                if (batch_idx + 1) % print_stride == 0:
                    progress = (batch_idx + 1) / len(self.train_dataloader)

                    infos = (
                        f"Train | epoch {(epoch + 1):>03} | "
                        f"[{(batch_idx + 1):>04}/{len(self.train_dataloader):>04}"
                        f"({progress:.2%})] | loss: {loss.item():0.6f}"
                    )

                    print(infos)

            if (epoch + 1) % self.save_checkpoint == 0:
                self._save_checkpoint(epoch, loss_sum)

            self.valid(epoch, "TRAIN")

            self.valid(epoch, "VALID")


    def valid(self, epoch: int, dataset: Literal["TRAIN", "VALID"]) -> None:
        """"""

        self.model.eval()

        test_loss = 0.0

        dataloader = self.train_dataloader if dataset == "TRAIN" else self.valid_dataloader

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)

                test_loss += self.criterion(output, target).item()

        test_loss /= len(dataloader)

        self.curve_writer.add_scalar(f"Loss/{dataset}", test_loss, epoch)

        infos = f"VALID | EPOCH {epoch}/{self.epochs} | {dataset} DATASET | AVG_LOSS: {test_loss:0.6f}"

        self.logger.info(color(infos))


    def apply_best_epoch(self, mode: str = 'TRAIN') -> nn.Module:
        ''''''

        epochs, losses = self._read_log(self.train_logger)

        min_loss_index = np.argmin(losses)

        self._load_checkpoint(epochs[min_loss_index])

        return self.model


    def apply_last_epoch(self) -> nn.Module:
        ''''''

        epochs, losses = self._read_log(self.train_logger)

        last_epoch = epochs[-1]

        self._load_checkpoint(last_epoch)

        return self.model


    def apply_epoch(self, epoch) -> nn.Module:
        ''''''

        self._load_checkpoint(epoch)

        return self.model


    def _train(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """"""

        self.optimizer.zero_grad()

        output = self.model(data)

        # import SimpleITK as sitk
        # sitk.WriteImage(sitk.GetImageFromArray(output.detach().cpu().numpy()[0, 0, :, :, :]), '/home/xzq/dev/zuo/tmp/mask_channel_0.mhd')
        # sitk.WriteImage(sitk.GetImageFromArray(output.detach().cpu().numpy()[0, 1, :, :, :]),'/home/xzq/dev/zuo/tmp/mask_channel_1.mhd')

        loss = self.criterion(output, target)

        loss.backward()

        self.optimizer.step()

        return loss


    def _read_log(self, mode: str = 'TRAIN'):
        ''''''

        epochs = []

        losses = []

        with open(self.logger_path, 'r') as f:
            for line in f.readlines():
                if f'{mode} DATASET' not in line:
                    continue

                parts = line.split('|')

                epoch_info = parts[1][len('Epoch'):].strip()

                epoch_info = epoch_info.split('/')

                epoch = int(epoch_info[0])

                epochs.append(epoch)

                loss = float(parts[-1][len('AVG_LOSS: '):].strip())

                losses.append(loss)

        return epochs, losses


    def _load_checkpoint(self, epoch):
        ''''''

        checkpoint_path = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')

        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['lr_scheduler_state_dict'] is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        start_epoch = checkpoint['epoch']

        return start_epoch


    def _save_checkpoint(self, epoch, loss):
        ''''''

        checkpoint_file = os.path.join(self.checkpoint_dir, f'epoch_{epoch}.pth')

        torch.save(
            {
                'epoch': epoch,
                'loss': loss,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            },
            checkpoint_file
        )
