import numpy as np

from mmpretrain.evaluation import MultiTasksMetric
from mmpretrain.registry import METRICS


@METRICS.register_module()
class MultiTasksAggregateMetric(MultiTasksMetric):

    def evaluate(self, size):
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.
        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are
            "{task_name}_{metric_name}" , and the values
            are corresponding results.
        """
        metrics = super().evaluate(size)
        
        aggregate_metrics = {}
        for metric, value in metrics.items():
            _, metric = metric.split('_', 1)
            aggregate_metrics.setdefault(metric, []).append(value)
        
        for metric, values in aggregate_metrics.items():
            aggregate_metrics[metric] = np.mean(values).item()

        metrics.update(aggregate_metrics)

        return metrics
        