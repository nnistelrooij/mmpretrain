# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/Kevinz-code/CSRA
from typing import Dict, List, Tuple

import torch

from mmpretrain.registry import MODELS
from mmpretrain.models.heads import MultiLabelClsHead
from mmpretrain.structures import DataSample


@MODELS.register_module()
class ClsMultiTaskHead(MultiLabelClsHead):
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
        task_classes: Dict[str, int],
        loss_weights: List[float],
        use_multilabel: bool,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.task_classes = task_classes
        self.loss_weights = loss_weights
        self._get_loss = self._get_multilabel_losses if use_multilabel else self._get_binary_losses

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
        logits = self.pre_logits(feats)
        logits = logits.split(tuple(self.task_classes.values()), dim=1)

        task_logits = {task: logit for task, logit in zip(self.task_classes, logits)}

        return task_logits
    
    def _get_multilabel_losses(
        self,
        cls_score: Dict[str, torch.Tensor],
        data_samples,
        **kwargs,
    ):
        """Unpack data samples and compute loss."""
        cls_score = torch.column_stack([cls_score[k] for k in self.task_classes])

        for sample in data_samples:
            binary_labels = [getattr(sample, k).gt_label for k in self.task_classes]            
            sample.gt_label = torch.cat(binary_labels).nonzero()[:, 0]

        losses = super()._get_loss(cls_score, data_samples, **kwargs)
                
        return losses
        
    def _get_binary_losses(
        self,
        cls_score: Dict[str, torch.Tensor],
        data_samples,
        **kwargs,
    ):
        """Unpack data samples and compute loss."""
        losses = {}
        for task, loss_weight in zip(self.task_classes, self.loss_weights):
            task_score = cls_score[task]            
            if task_score.shape[1] == 1:
                task_score = torch.cat((-task_score, task_score), dim=1)

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
        for task, scores in cls_score.items():
            if scores.shape[1] == 1:
                scores = torch.sigmoid(scores)
                scores = torch.column_stack((1 - scores, scores))
            else:
                scores = torch.softmax(scores, dim=1)

            
            for data_sample, scores in zip(data_samples, scores):
                label = (scores > self.thr).to(int)

                getattr(data_sample, task)\
                    .set_pred_score(scores)\
                    .set_pred_label(label)\
                    .set_field(
                        task in data_sample.tasks,
                        'eval_mask',
                        field_type='metainfo',
                    )

        return data_samples
