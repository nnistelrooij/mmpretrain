from mmpretrain.registry import MODELS
from mmpretrain.models.utils import ClsDataPreprocessor

import torch


@MODELS.register_module()
class LatentDataPreprocessor(ClsDataPreprocessor):

    def forward(self, data: dict, training: bool = False) -> dict:
        results = super().forward(data, training)

        if isinstance(getattr(results['data_samples'][0], 'logits'), list):
            logits = []
            for data_sample in results['data_samples']:
                logits.append(torch.tensor(data_sample.logits))

            logits = torch.stack(logits).to(results['inputs'])
            logits = torch.cat((
                torch.maximum(logits[:, :8], logits[:, 8:16]),
                torch.maximum(logits[:, 16:24], logits[:, 24:]),
            ), dim=-1)
        else:
            logits = torch.tensor(0)

        results['inputs'] = results['inputs'], logits

        return results
        