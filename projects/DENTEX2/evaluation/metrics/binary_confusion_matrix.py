from typing import Sequence

from mmpretrain.registry import METRICS
from mmpretrain.evaluation import ConfusionMatrix


@METRICS.register_module()
class BinaryConfusionMatrix(ConfusionMatrix):
    
    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_label, gt_label = 0, 0
            for task in data_sample.tasks:
                pred_label |= getattr(data_sample, task).pred_label[1]
                gt_label |= getattr(data_sample, task).gt_label[0]

            self.results.append({
                'pred_label': [pred_label],
                'gt_label': [gt_label],
            })
