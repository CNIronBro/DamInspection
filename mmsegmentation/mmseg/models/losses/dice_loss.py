# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import weight_reduce_loss


def _expand_onehot_labels_dice(pred: torch.Tensor,
                               target: torch.Tensor) -> torch.Tensor:
    """Expand onehot labels to match the size of prediction.

    Args:
        pred (torch.Tensor): The prediction, has a shape (N, num_class, H, W).
        target (torch.Tensor): The learning label of the prediction,
            has a shape (N, H, W).

    Returns:
        torch.Tensor: The target after one-hot encoding,
            has a shape (N, num_class, H, W).
    """
    num_classes = pred.shape[1]
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target,
                                                 num_classes + 1)
    one_hot_target = one_hot_target[..., :num_classes].permute(0, 3, 1, 2)
    return one_hot_target

import torch
import torch.nn.functional as F

def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              avg_factor=None,
              naive_dice=False,
              **kwargs):
    """Dice loss with pixel-wise ignore (label==255), compatible with mmseg target shapes.

    pred:   (N, C, H, W) logits
    target: (N, H, W) or (N, 1, H, W) or (H, W)
    """
    # ---- normalize target shape to (N, H, W) ----
    if target.dim() == 4:
        # (N, 1, H, W) -> (N, H, W)
        if target.size(1) == 1:
            target = target.squeeze(1)
        else:
            # unexpected multi-channel target
            target = target[:, 0, ...]
    elif target.dim() == 2:
        # (H, W) -> (1, H, W)
        target = target.unsqueeze(0)

    # pred should be (N, C, H, W)
    num_classes = pred.shape[1]

    # logits -> probs
    pred = F.softmax(pred, dim=1)

    # ---- pixel-wise ignore ----
    pixel_ignore_index = 255
    valid_mask = (target != pixel_ignore_index)  # (N,H,W) bool

    target_safe = target.clone()
    target_safe[~valid_mask] = 0  # make ignore pixels safe for one_hot

    # one-hot: (N,H,W) -> (N,H,W,C) -> (N,C,H,W)
    target_onehot = F.one_hot(
        target_safe.long(), num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

    # apply valid mask to both pred and target_onehot
    valid_mask_f = valid_mask.unsqueeze(1).float()  # (N,1,H,W)
    pred = pred * valid_mask_f
    target_onehot = target_onehot * valid_mask_f

    # dice over batch+spatial (per class)
    intersection = torch.sum(pred * target_onehot, dim=(0, 2, 3))
    union = torch.sum(pred + target_onehot, dim=(0, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)

    loss = 1 - dice  # (C,)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


@MODELS.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 loss_name='loss_dice'):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            ignore_index (int, optional): The label index to be ignored.
                Default: 255.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_dice'.
        """

        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        one_hot_target = target
        if (pred.shape != target.shape):
            one_hot_target = _expand_onehot_labels_dice(pred, target)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            elif pred.shape[1] != 1:
                # softmax does not work when there is only 1 class
                pred = pred.softmax(dim=1)
        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index)

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
