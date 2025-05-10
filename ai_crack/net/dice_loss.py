# -*- coding:utf-8 -*-

''''''

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, mode, smooth=1e-5, from_logits=False, ignore_index=None, eps=1e-7):
        """
        Dice loss for 3D image segmentation task.
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
        self.smooth = smooth
        self.from_logits = from_logits
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Computes the Dice loss for 3D images.

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

        if self.mode == 'binary':
            intersection = (y_pred * y_true).sum(dim=(1, 2, 3, 4))
            union = y_pred.sum(dim=(1, 2, 3, 4)) + y_true.sum(dim=(1, 2, 3, 4))
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()

        elif self.mode == 'multiclass':
            y_true = F.one_hot(y_true.to(torch.long), num_classes=y_pred.shape[1]).squeeze(1).permute(0, 4, 1, 2, 3)
            intersection = (y_pred * y_true).sum(dim=(2, 3, 4))
            union = y_pred.sum(dim=(2, 3, 4)) + y_true.sum(dim=(2, 3, 4))
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()

            # y_pred = y_pred.max(1)[0]
            # # y_true.squeeze_(1)
            # y_pred = torch.unsqueeze(y_pred, dim=1)
            # intersection = (y_pred * y_true).sum(dim=(2, 3, 4))
            # union = y_pred.sum(dim=(2, 3, 4)) + y_true.sum(dim=(2, 3, 4))
            # dice = (2. * intersection + self.smooth) / (union + self.smooth)
            # return 1 - dice.mean()

        else:
            raise ValueError("Unsupported mode. Choose 'binary' or 'multiclass'.")
