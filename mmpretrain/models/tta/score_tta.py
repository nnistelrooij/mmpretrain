# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.model import BaseTTAModel

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample, MultiTaskDataSample


@MODELS.register_module()
class AverageClsScoreTTA(BaseTTAModel):

    def merge_preds(
        self,
        data_samples_list: List[List[DataSample]],
    ) -> List[DataSample]:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[List[DataSample]]): List of predictions
                of all enhanced data.

        Returns:
            List[DataSample]: Merged prediction.
        """
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples

    def _merge_single_sample(self, data_samples):
        if hasattr(data_samples[0], 'tasks'):
            merged_data_sample: MultiTaskDataSample = data_samples[0].new()
            for task in merged_data_sample.tasks:
                merged_score = sum(
                    getattr(data_sample, task).pred_score
                    for data_sample in data_samples
                ) / len(data_samples)
                getattr(merged_data_sample, task).set_pred_score(merged_score)
        else:
            merged_data_sample: DataSample = data_samples[0].new()
            merged_score = sum(data_sample.pred_score
                            for data_sample in data_samples) / len(data_samples)
            merged_data_sample.set_pred_score(merged_score)
        
        return merged_data_sample
