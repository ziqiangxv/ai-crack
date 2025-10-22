# -*- coding:utf-8 -*-

''''''

import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..factory import REGISTER

class DiceLoss(nn.Module):
    def __init__(
        self,
        mode = 'multiclass',
        channel_weights = [0.5, 0.5],
        smooth = 1e-5,
        from_logits = False,
        ignore_index = None,
        eps = 1e-7,
        list_prediction = True,
        list_prediction_weights = [0.5, 0.5]):
        """
        Dice loss for 3D/2D image segmentation task.
        It supports binary and multiclass cases.

        Args:
            mode: Loss mode 'binary' or 'multiclass'
            smooth: Smoothness constant for dice coefficient
            from_logits: If True, assumes input is raw logits
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
        """

        super(DiceLoss, self).__init__()

        self.mode = mode
        self.channel_weights = channel_weights
        self.smooth = smooth
        self.from_logits = from_logits
        self.ignore_index = ignore_index
        self.eps = eps
        self.list_prediction = list_prediction
        self.list_prediction_weights = list_prediction_weights

    def forward(self, y_preds, y_true):
        if self.list_prediction and isinstance(y_preds, typing.Sequence):
            assert len(y_preds) == len(self.list_prediction_weights)

            loss = 0

            for i, y_pred in enumerate(y_preds):
                loss += self._forward(y_pred, y_true) * self.list_prediction_weights[i]

            return loss

        else:
            return self._forward(y_preds, y_true)


    def _forward(self, y_pred, y_true):
        """
        Computes the Dice loss for 3D/2D images.

        Args:
            y_pred: Tensor of predictions (batch_size, C, D, H, W).
            y_true: Tensor of ground truth (batch_size, D, H, W) or (batch_size, C, D, H, W).

        Returns:
            Scalar Dice Loss.
        """

        if self.from_logits:
            if self.mode == 'binary':
                y_pred = torch.sigmoid(y_pred)

            elif self.mode == 'multiclass':
                y_pred = F.softmax(y_pred, dim=1)

            else:
                raise ValueError("Unsupported mode. Choose 'binary' or 'multiclass'.")

        if len(y_pred.shape) == 5:
            return 1 - self._compute_dice_3d(y_pred, y_true)

        elif len(y_pred.shape) == 4:
            return 1 - self._compute_dice_2d(y_pred, y_true)

        else:
            raise ValueError("Unsupported input shape. Expected 4D (2D) or 5D (3D).")


    def _compute_dice_3d(self, y_pred, y_true):
        """
        Computes the Dice coefficient for 3D images.
        Args:
            y_pred: Tensor of predictions (batch_size, C, D, H, W).
            y_true: Tensor of ground truth (batch_size, C, D, H, W).
        Returns:
            Dice coefficient.
        """

        if self.mode == 'binary':
            intersection = (y_pred * y_true).sum(dim=(1, 2, 3, 4))

            union = y_pred.sum(dim=(1, 2, 3, 4)) + y_true.sum(dim=(1, 2, 3, 4))

            dice = (2. * intersection + self.smooth) / (union + self.smooth)

            return dice.mean()

        elif self.mode == 'multiclass':
            y_true = F.one_hot(y_true.to(torch.long), num_classes=y_pred.shape[1]).squeeze(1).permute(0, 4, 1, 2, 3)

            intersection = (y_pred * y_true).sum(dim=(2, 3, 4))

            union = y_pred.sum(dim=(2, 3, 4)) + y_true.sum(dim=(2, 3, 4))

            dice = (2. * intersection + self.smooth) / (union + self.smooth)

            weights = torch.tensor([self.channel_weights] * dice.shape[0], device=y_pred.device, requires_grad=False)

            return (weights * dice).sum(axis=1).mean()

        else:
            raise ValueError("Unsupported mode. Choose 'binary' or 'multiclass'.")

    def _compute_dice_2d(self, y_pred, y_true):
        """
        Computes the Dice coefficient for 2D images.
        Args:
            y_pred: Tensor of predictions (batch_size, C, H, W).
            y_true: Tensor of ground truth (batch_size, C, H, W).
        Returns:
            Dice coefficient.
        """

        if self.mode == 'binary':
            intersection = (y_pred * y_true).sum(dim=(1, 2, 3))

            union = y_pred.sum(dim=(1, 2, 3)) + y_true.sum(dim=(1, 2, 3))

            dice = (2. * intersection + self.smooth) / (union + self.smooth)

            return dice.mean()

        elif self.mode =='multiclass':
            # y_true[y_true == 3] = 2 # label必要要连续

            y_true = F.one_hot(y_true.to(torch.long), num_classes=y_pred.shape[1]).squeeze(1).permute(0, 3, 1, 2)

            intersection = (y_pred * y_true).sum(dim=(2, 3))

            union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))

            dice = (2. * intersection + self.smooth) / (union + self.smooth)

            weights = torch.tensor([self.channel_weights] * dice.shape[0], device=y_pred.device, requires_grad=False)

            return (weights * dice).sum(axis=1).mean()


REGISTER('loss', 'dice_loss', DiceLoss)
