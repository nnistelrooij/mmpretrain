from typing import Any, Dict, List

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MultilabelConvNeXts(nn.Module):

    def __init__(
        self,
        classifiers: Dict[str, Any],
        pretrained: str,

    ):
        super().__init__()

        self.binary_classifiers = nn.ModuleDict()
        for task, classifier in classifiers.items():
            classifier = MODELS.build(classifier)
            self.binary_classifiers[task] = classifier

        self.MLP = nn.Sequential(
            nn.Linear(2 * len(classifiers), 256, bias=False),
            nn.LeakyReLU(),
            nn.Linear(256, 256, bias=False),
            nn.LeakyReLU(),
            nn.Linear(256, len(classifiers)),
        )
        # self.MLP = nn.Sequential(
        #     nn.Linear(2 * len(classifiers), len(classifiers)),
        # )

        self.pretrained = pretrained
        multilabel_backbone = list(classifiers.values())[0].backbone
        self.multilabel_backbone = MODELS.build(multilabel_backbone)

        self.full_init = False

    def forward(self, x):
        if not self.full_init:
            for _, classifier in self.binary_classifiers.items():
                if classifier.init_cfg is None:
                    continue

                ckpt = torch.load(classifier.init_cfg['checkpoint'])
                classifier.load_state_dict(ckpt['state_dict'], strict=False)

            state_dict = torch.load(self.pretrained)['state_dict']
            backbone_dict = {}
            for key,value in state_dict.items():
                if 'backbone' in key:
                    backbone_dict[key.replace('backbone.', '')] = value

            self.multilabel_backbone.load_state_dict(backbone_dict, strict=False)

            self.full_init = True

        with torch.no_grad():
            logits = []
            for task, classifier in self.binary_classifiers.items():
                pred = classifier(x)
                logits.append(pred)

        logits = torch.column_stack(logits)
        
        binary_logits = self.MLP(logits)

        multilabel_feats = self.multilabel_backbone(x)

        # pred = logits[:, 1::2] - logits[:, ::2]
        # pred.requires_grad_(True)

        return binary_logits, multilabel_feats[0]
