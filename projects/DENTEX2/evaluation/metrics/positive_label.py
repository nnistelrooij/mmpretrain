from mmpretrain.evaluation import SingleLabelMetric
from mmpretrain.registry import METRICS

from .binary_label import draw_roc_curve


@METRICS.register_module()
class PositiveLabelMetric(SingleLabelMetric):

    def __init__(
        self,
        prefix: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, prefix='positive-label')

        self.prefix_ = prefix        

    def evaluate(self, size):
        # determine and draw ROC curve of results
        draw_roc_curve(self.results, self.prefix_)

        metrics = super().evaluate(size)

        if self.average is not None:
            return metrics
        
        metrics['positive-label/recall_positive'] = metrics['positive-label/recall_classwise'][1]
        metrics['positive-label/precision_positive'] = metrics['positive-label/precision_classwise'][1]
        metrics['positive-label/f1-score_positive'] = metrics['positive-label/f1-score_classwise'][1]

        return metrics
        