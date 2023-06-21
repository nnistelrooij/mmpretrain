from typing import Sequence

import matplotlib.pyplot as plt
from mmengine.visualization import Visualizer
import numpy as np
from sklearn.metrics import RocCurveDisplay
import torch

from mmpretrain.evaluation import SingleLabelMetric
from mmpretrain.registry import METRICS


def draw_roc_curve(results, prefix):
    pred_scores = torch.stack([r['pred_score'][1] for r in results]).cpu().numpy()
    gt_labels = torch.cat([r['gt_label'] for r in results]).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax = RocCurveDisplay.from_predictions(gt_labels, pred_scores, ax=ax).ax_
    ax.grid()

    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)  # (H, W, 3)

    vis = Visualizer.get_current_instance()
    vis.add_image(f'{prefix}/roc_curve', image, step=0)


@METRICS.register_module()
class BinaryLabelMetric(SingleLabelMetric):

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_score, pred_label, gt_label = torch.tensor(0.0), 0, 0
            for task in data_sample:
                pred_score = torch.maximum(pred_score, data_sample[task]['pred_score'])
                pred_label |= data_sample[task]['pred_label'][0]
                gt_label |= data_sample[task]['gt_label'][0]

            self.results.append({
                'pred_score': pred_score,
                'pred_label': pred_label.reshape(-1),
                'gt_label': gt_label.reshape(-1),
                'num_classes': 2,
            })

    def evaluate(self, size):
        # determine and draw ROC curve of results
        draw_roc_curve(self.results, self.prefix)


        # determine and log classification metrics of results
        for result in self.results:
            result.pop('pred_score')

        metrics = super().evaluate(size)


        # additionally save classification metrics of only positive clas
        if self.average is not None:
            return metrics
        
        metrics['binary-label/recall_positive'] = metrics['binary-label/recall_classwise'][1]
        metrics['binary-label/precision_positive'] = metrics['binary-label/precision_classwise'][1]
        metrics['binary-label/f1-score_positive'] = metrics['binary-label/f1-score_classwise'][1]

        return metrics
