# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/Kevinz-code/CSRA
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList

from mmpretrain.registry import MODELS
from mmpretrain.models.heads.multi_label_csra_head import CSRAClsHead
from mmpretrain.structures import DataSample


@MODELS.register_module()
class CSRAMultiTaskHead(CSRAClsHead):
    """Class-specific residual attention classifier head.

    Please refer to the `Residual Attention: A Simple but Effective Method for
    Multi-Label Recognition (ICCV 2021) <https://arxiv.org/abs/2108.02456>`_
    for details.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        num_heads (int): Number of residual at tensor heads.
        loss (dict): Config of classification loss.
        lam (float): Lambda that combines global average and max pooling
            scores.
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to use ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(
        self,
        tasks: List[str],
        loss_weights: List[float],
        *args,
        **kwargs,
    ):
        super().__init__(num_classes=len(tasks), *args, **kwargs)

        self.loss_weights = loss_weights
        self.tasks = tasks

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``CSRAClsHead``, we just obtain the
        feature of the last stage.
        """
        # The CSRAClsHead doesn't have other module, just return after
        # unpacking.
        return feats[-1]

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        logit = sum([head(pre_logits) for head in self.csra_heads])

        task_logits = {task: logit for task, logit in zip(self.tasks, logit.T)}

        return task_logits

    def _get_loss(
        self,
        cls_score: Dict[str, torch.Tensor],
        data_samples,
        **kwargs,
    ):
        """Unpack data samples and compute loss."""
        losses = {}
        for task, loss_weight in zip(self.tasks, self.loss_weights):
            task_score = cls_score[task]            
            task_score = torch.stack((-task_score, task_score), dim=1)

            for sample in data_samples:
                sample.gt_label = getattr(sample, task).gt_label

            task_losses = super()._get_loss(task_score, data_samples, **kwargs)
            for loss, value in task_losses.items():
                losses[f'{task}_{loss}'] = loss_weight * value
                
        return losses
    
    def _get_predictions(self, cls_score: Dict[str, torch.Tensor],
                         data_samples: List[DataSample]):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = {
            task: torch.sigmoid(score)
            for task, score in cls_score.items()
        }
        pred_scores = {
            task: torch.stack((1 - score, score), dim=1)
            for task, score in pred_scores.items()
        }

        for task, scores in pred_scores.items():
            for data_sample, score in zip(data_samples, scores):
                if self.thr is not None:
                    # a label is predicted positive if larger than thr
                    label = torch.where(score > self.thr)[0]
                else:
                    # top-k labels will be predicted positive for any example
                    _, label = score.topk(self.topk)

                getattr(data_sample, task)\
                    .set_pred_score(score)\
                    .set_pred_label(label)\
                    .set_field(
                        task in data_sample.tasks,
                        'eval_mask',
                        field_type='metainfo',
                    )

        return data_samples
