import copy

from mmpretrain.evaluation import SingleLabelMetric
from mmpretrain.registry import METRICS

from .binary_label import draw_confusion_matrix, draw_roc_curve


@METRICS.register_module()
class PositiveLabelMetric(SingleLabelMetric):

    def __init__(
        self,
        prefix: str='positive-label',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, prefix='positive-label')

        self.prefix_ = prefix  
        self.steps = 0      

    def evaluate(self, size):
        # determine and draw ROC curve of results
        self.steps += 1
        roc_metrics = draw_roc_curve(self.results, self.prefix_, self.steps)

        thr = roc_metrics['optimal_thr']
        cm_metrics = draw_confusion_matrix(self.results, thr, self.prefix_, self.steps)

        self.results.clear()


        if self.average is not None:
            metrics = {**roc_metrics, **cm_metrics}
            return {f'{self.prefix}/{k}': v for k, v in metrics.items()}

        return {}
        