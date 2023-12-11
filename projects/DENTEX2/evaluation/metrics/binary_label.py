from typing import Dict, Sequence

import cv2
import matplotlib.pyplot as plt
from mmengine.visualization import Visualizer
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)
import torch
from torchvision.utils import make_grid

from mmpretrain.evaluation import SingleLabelMetric
from mmpretrain.registry import METRICS


def draw_roc_curve(
    results,
    prefix,
    steps=0,
    specificity_range=[0.9, 1.0],
    thr_range=[0.1, 0.9],
    thr_method: str='er',
) -> Dict[str, float]:
    pred_scores = torch.stack([r['pred_score'][1] for r in results]).cpu().numpy()
    gt_labels = torch.cat([r['gt_label'] for r in results]).cpu().numpy()

    # determine statistics at thresholds
    fpr, tpr, thrs = roc_curve(gt_labels, pred_scores)
    sensitivity, specificity = np.stack((tpr, 1 - fpr))

    # at least specificity of 90% and reasonable thresholds
    mask = (
        (specificity >= specificity_range[0]) &
        (specificity <= specificity_range[1]) &
        (thrs >= thr_range[0]) &
        (thrs <= thr_range[1])
    )
    if np.any(mask):
        fpr, tpr, thrs = fpr[mask], tpr[mask], thrs[mask]
        sensitivity, specificity = sensitivity[mask], specificity[mask]

    # compute criteria to determine optimal threshold
    if thr_method == 'iu':
        auc = roc_auc_score(gt_labels, pred_scores)
        criteria = np.abs(sensitivity - auc) + np.abs(specificity - auc)
    elif thr_method == 'er':
        criteria = np.sqrt((1 - sensitivity) ** 2 + (1 - specificity) ** 2)
    elif thr_method == 'f1':
        criteria = np.zeros_like(thrs)
        for i, thr in enumerate(thrs):
            criteria[i] = -f1_score(gt_labels, pred_scores >= thr)
    else:
        raise TypeError('thr_method must be one of ["iu", "er", "f1"].')

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    RocCurveDisplay.from_predictions(gt_labels, pred_scores, ax=ax)
    ax.scatter([fpr[criteria.argmin()]], [tpr[criteria.argmin()]], s=48, c='r')
    ax.grid()

    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)  # (H, W, 3)

    vis = Visualizer.get_current_instance()
    vis.add_image(f'{prefix}/roc_curve', image, step=steps)
    plt.close(fig)

    return {
        'auc': roc_auc_score(gt_labels, pred_scores),
        'optimal_thr': thrs[criteria.argmin()],
    }


def draw_confusion_matrix(
    results,
    thr,
    prefix,
    steps=0,
):
    pred_labels = torch.stack([r['pred_score'][1] >= thr for r in results]).cpu().numpy()
    gt_labels = torch.cat([r['gt_label'] for r in results]).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(gt_labels, pred_labels, ax=ax)

    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)  # (H, W, 3)

    vis = Visualizer.get_current_instance()
    vis.add_image(f'{prefix}/confusion_matrix', image, step=steps)
    plt.close(fig)

    return {
        'precision': precision_score(gt_labels, pred_labels),
        'recall': recall_score(gt_labels, pred_labels),
        'f1-score': f1_score(gt_labels, pred_labels),
    }


def draw_false_positive(
    results,
    prefix,
    steps,
):
    diffs = torch.cat([r['pred_score'][1] - r['gt_label'] for r in results])
    for i, idx in enumerate(torch.argsort(diffs, descending=True)[:5]):
        image = cv2.imread(results[idx]['img_path'])
        if (
            results[idx]['gt_label'][0] == 1
            or results[idx]['pred_label'][0] == 0
        ):
            image = np.zeros_like(image)
        vis = Visualizer.get_current_instance()
        vis.add_image(f'{prefix}/false_positive_{i}', image, step=steps)


def draw_false_negative(
    results,
    prefix,
    steps,
):
    diffs = torch.cat([r['gt_label'] - r['pred_score'][1] for r in results])
    for i, idx in enumerate(torch.argsort(diffs, descending=True)[:5]):
        image = cv2.imread(results[idx]['img_path'])
        if (
            results[idx]['gt_label'][0] == 0
            or results[idx]['pred_label'][0] == 1
        ):
            image = np.zeros_like(image)
        vis = Visualizer.get_current_instance()
        vis.add_image(f'{prefix}/false_negative_{i}', image, step=steps)


@METRICS.register_module()
class BinaryLabelMetric(SingleLabelMetric):

    def __init__(
        self,
        prefix: str='binary-label',
        gt_score_thr: float=0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, prefix='binary-label')

        self.prefix_ = prefix
        self.gt_score_thr = gt_score_thr
        self.steps = 0  

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_label' not in data_sample:
                pred_score, pred_label, gt_label = torch.tensor(0.0), 0, 0
                for task in data_sample:
                    pred_score = torch.maximum(pred_score, data_sample[task]['pred_score'])
                    pred_label |= data_sample[task]['pred_label'][0]
                    if 'gt_score' in data_sample:
                        gt_label |= data_sample[task]['gt_score'][1] >= self.gt_score_thr
                    else:
                        gt_label |= data_sample[task]['gt_label'][0]
            else:
                pred_score = data_sample['pred_score']
                pred_label = data_sample['pred_label'][0]
                if 'gt_score' in data_sample:
                    gt_label = data_sample['gt_score'][1] >= self.gt_score_thr
                else:
                    gt_label = data_sample['gt_label'][0]

            self.results.append({
                'img_path': data_sample['img_path'],
                'pred_score': pred_score,
                'pred_label': pred_label.reshape(-1).long(),
                'gt_label': gt_label.reshape(-1).long(),
                'num_classes': 2,
            })

    def evaluate(self, size):
        # determine and draw ROC curve of results
        self.steps += 1

        draw_false_positive(self.results, self.prefix_, self.steps)
        draw_false_negative(self.results, self.prefix_, self.steps)

        roc_metrics = draw_roc_curve(self.results, self.prefix_, self.steps, thr_method='f1')

        thr = roc_metrics['optimal_thr']
        cm_metrics = draw_confusion_matrix(self.results, thr, self.prefix_, self.steps)

        self.results.clear()


        if self.average is not None:
            metrics = {**roc_metrics, **cm_metrics}
            return {f'{self.prefix}/{k}': v for k, v in metrics.items()}

        return {}
