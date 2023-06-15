from mmpretrain.evaluation import SingleLabelMetric
from mmpretrain.registry import METRICS


@METRICS.register_module()
class PositiveLabelMetric(SingleLabelMetric):

    def evaluate(self, size):
        metrics = super().evaluate(size)

        if self.average is not None:
            return metrics
        
        metrics['single-label/recall_positive'] = metrics['single-label/recall_classwise'][1]
        metrics['single-label/precision_positive'] = metrics['single-label/precision_classwise'][1]
        metrics['single-label/f1-score_positive'] = metrics['single-label/f1-score_classwise'][1]

        return metrics
        