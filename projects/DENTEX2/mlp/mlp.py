from typing import Any, Dict, List

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MultilabelMLP(nn.Module):

    def __init__(
        self,
        checkpoints: List[str],
        model_cfg: Dict[str, Any],
    ):
        super().__init__()

        self.binary_classifiers = nn.ModuleList()
        for checkpoint in checkpoints:
            checkpoint = torch.load(checkpoint)

            model_cfg['head']['num_classes'] = 2
            model_cfg['train_cfg'] = None

            classifier = MODELS.build(model_cfg)
            classifier.load_state_dict(checkpoint['state_dict'])
            classifier.requires_grad_(False)

            self.binary_classifiers.append(classifier)

        in_channels = 2 * len(checkpoints)
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        logits = []
        for classifier in self.binary_classifiers:
            pred = classifier(x)
            logits.append(pred)

        logits = torch.column_stack(logits)
        
        pred = self.MLP(logits)
        # pred = logits[:, 1::2]
        # pred.requires_grad_(True)

        return (pred,)
