from typing import Any, Dict

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS


@MODELS.register_module()
class MultilabelConvNeXts(nn.Module):

    def __init__(
        self,
        classifiers: Dict[str, Any],
        pretrained: str,
        mix_features: bool,
        mix_outputs: bool,
    ):
        super().__init__()

        if mix_features or mix_outputs:
            self.binary_classifiers = nn.ModuleDict()
            for task, classifier in classifiers.items():
                classifier = MODELS.build(classifier)
                self.binary_classifiers[task] = classifier

        if mix_outputs:
            self.MLP = nn.Sequential(
                nn.Linear(2 * len(classifiers), 256, bias=False),
                nn.LeakyReLU(),
                nn.Linear(256, 256, bias=False),
                nn.LeakyReLU(),
                nn.Linear(256, len(classifiers)),
            )

        multilabel_backbone = list(classifiers.values())[0].backbone
        self.multilabel_backbone = MODELS.build(multilabel_backbone)

        self.full_init = False
        self.pretrained = pretrained
        self.mix_features = mix_features
        self.mix_outputs = mix_outputs

    def _init_pretrained(self):
        # load pretrained ImageNet-1k weights
        state_dict = torch.load(self.pretrained)['state_dict']
        backbone_dict = {}
        for key,value in state_dict.items():
            if 'backbone' in key:
                backbone_dict[key.replace('backbone.', '')] = value

        self.multilabel_backbone.load_state_dict(backbone_dict, strict=False)

        # load pretrained binary classifier weights
        if not self.mix_features and not self.mix_outputs:
            return

        for _, classifier in self.binary_classifiers.items():
            if classifier.init_cfg is None:
                continue

            ckpt = torch.load(classifier.init_cfg['checkpoint'])
            classifier.load_state_dict(ckpt['state_dict'], strict=False)

    def _binary_forward(self, x):
        binary_feats = torch.zeros((x.shape[0], 0, x.shape[2] // 32, x.shape[3] // 32))
        binary_logits = torch.zeros((x.shape[0], len(self.binary_classifiers)))

        if self.mix_features:
            for classifier in self.binary_classifiers.values():
                feats = classifier.backbone(x)
                binary_feats = torch.cat((binary_feats, feats[-1]), dim=1)

        if self.mix_outputs:
            logits = torch.zeros((x.shape[0], 0))
            for classifier in self.binary_classifiers.values():
                logits = torch.cat((logits, classifier(x)), dim=1)

            binary_logits += self.MLP(logits)

        return binary_logits, binary_feats

    def forward(self, x):
        if not self.full_init:
            self._init_pretrained()
            self.full_init = True

        with torch.no_grad():
            binary_logits, binary_feats = self._binary_forward(x)

        multilabel_feats = self.multilabel_backbone(x)[-1]

        return binary_logits, torch.cat((binary_feats, multilabel_feats), dim=1)
