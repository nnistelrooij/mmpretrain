from typing import Any, Dict, List

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MultilabelMLP(nn.Module):

    def __init__(
        self,
        classifiers: Dict[str, Any],
    ):
        super().__init__()

        self.binary_classifiers = nn.ModuleDict()
        for task, classifier in classifiers.items():
            classifier = MODELS.build(classifier)
            self.binary_classifiers[task] = classifier

        self.MLP = nn.Sequential(
            nn.Linear(2 * len(classifiers), 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, len(classifiers)),
        )

        self.full_init = False

    def forward(self, x):
        if not self.full_init:
            for _, classifier in self.binary_classifiers.items():
                ckpt = torch.load(classifier.init_cfg['checkpoint'])
                classifier.load_state_dict(ckpt['state_dict'], strict=False)

            self.full_init = True

        with torch.no_grad():
            logits = []
            for task, classifier in self.binary_classifiers.items():
                pred = classifier(x)
                logits.append(pred)

        logits = torch.column_stack(logits)
        
        pred = self.MLP(logits)

        # pred = logits[:, 1::2] - logits[:, ::2]
        # pred.requires_grad_(True)

        return (pred,)
