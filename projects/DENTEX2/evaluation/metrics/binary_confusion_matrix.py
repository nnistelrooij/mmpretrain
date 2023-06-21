from typing import Sequence

import torch

from mmpretrain.registry import METRICS
from mmpretrain.evaluation import ConfusionMatrix


@METRICS.register_module()
class BinaryConfusionMatrix(ConfusionMatrix):
    
    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        pred_labels, gt_labels = [], []
        for data_sample in data_samples:
            pred_label, gt_label = 0, 0
            for task in data_sample:
                pred_label |= data_sample[task]['pred_label'][0]
                gt_label |= data_sample[task]['gt_label'][0]

            pred_labels.append(pred_label)
            gt_labels.append(gt_label)

        self.results.append({
            'pred_label': torch.stack(pred_labels),
            'gt_label': torch.stack(gt_labels),
            'num_classes': 2,
        })