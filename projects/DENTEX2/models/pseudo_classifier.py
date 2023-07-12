# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from mmengine.logging import MMLogger

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from mmpretrain.models import ImageClassifier


@MODELS.register_module()
class PseudoClassifier(ImageClassifier):
    """Image classifiers for supervised classification task.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmpretrain.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmpretrain.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmpretrain.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        train_cfg (dict, optional): The training setting. The acceptable
            fields are:

            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in
              :mod:`mmpretrain.model.utils.augment`.
            - probs (List[float], optional): The probability of every batch
              augmentation methods. If None, choose evenly. Defaults to None.

            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "ClsDataPreprocessor" as type. See :class:`ClsDataPreprocessor` for
            more details. Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.step = torch.tensor(100)
        self.prev_labeled = True

    @staticmethod
    def _alpha_weight(
        step: torch.Tensor,
        T1: int=100,
        T2: int=200,
        af: float=3.0,
    ):
        alpha = (step - T1) / (T2 - T1)

        return af * alpha.clip(0, 1)

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        alpha = 1.0
        if not data_samples[0].labeled:
            self.eval()
            data_samples = self.predict(inputs, data_samples)
            for data_sample in data_samples:
                if hasattr(data_sample, 'tasks'):
                    for task in data_sample.tasks:
                        getattr(data_sample, task).set_gt_label(
                            getattr(data_sample, task).pred_label,
                        )
                        delattr(getattr(data_sample, task), 'pred_label')
                        delattr(getattr(data_sample, task), 'pred_score')
                else:
                    data_sample.set_gt_label(data_sample.pred_label)
                    delattr(data_sample, 'pred_label')
                    delattr(data_sample, 'pred_score')

            if self.prev_labeled:
                self.step += 1

            alpha = self._alpha_weight(self.step)
            MMLogger.get_current_instance().info(f'Alpha: {alpha.item()}')

        self.train()
        feats = self.extract_feat(inputs)
        loss = self.head.loss(feats, data_samples)

        loss = {k: alpha * v for k, v in loss.items()}

        self.prev_labeled = data_samples[0].labeled

        return loss
