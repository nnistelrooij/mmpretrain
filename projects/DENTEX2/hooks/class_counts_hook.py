from mmengine.hooks import Hook
from mmengine.runner import Runner
import torch

from mmpretrain.registry import HOOKS
from mmengine.logging import MMLogger


@HOOKS.register_module()
class ClassCountsHook(Hook):

    def __init__(
        self,
        num_classes: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.counts = torch.zeros(num_classes, dtype=int)

    def before_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch,
    ) -> None:
        for data_sample in data_batch['data_samples']:
            self.counts[data_sample.gt_label] += 1

    def after_train_epoch(
        self,
        runner: Runner,
    ):
        logger = MMLogger.get_current_instance()
        logger.info(f'Class counts: {self.counts}')

        self.counts *= 0
