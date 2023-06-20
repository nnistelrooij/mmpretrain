# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import PatchEmbed

from mmpretrain.registry import MODELS

from mmpretrain.models.utils import (
    build_norm_layer,
    resize_pos_embed,
    to_2tuple,
)
from mmpretrain.models import BEiTPretrainViT, SwinTransformerV2
from mmpretrain.models.utils import NormEMAVectorQuantizer
from mmpretrain.models.selfsup import VQKD


MODELS.register_module()
class BEiTSwin(SwinTransformerV2):
    
    def __init__(
        self,
        arch: str,
        img_size: int,
        in_channels: int,
        patch_size: int,
        with_cls_token: bool,
        out_type: str,
        final_norm,
        norm_cfg,
        frozen_stages,
        patch_cfg,
        drop_path_rate: float,
        *args,
        **kwargs,
    ):
        super().__init__(
            arch=arch,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            *args,
            **kwargs,
        )

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # Set cls token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
            self.num_extra_tokens = 1
        elif out_type != 'cls_token':
            self.cls_token = None
            self.num_extra_tokens = 0
        else:
            raise ValueError(
                'with_cls_token must be True when `out_type="cls_token"`.')

        self.frozen_stages = frozen_stages
        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        if out_type == 'avg_featmap':
            self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    def forward(self, x):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.use_abs_pos_embed:
            x = x + resize_pos_embed(
                self.absolute_pos_embed, self.patch_resolution, hw_shape,
                self.interpolate_mode, self.num_extra_tokens)
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out.view(-1, *hw_shape,
                               stage.out_channels).permute(0, 3, 1,
                                                           2).contiguous()
                outs.append(out)

        return tuple(outs)

    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return self.ln2(patch_token.mean(dim=1))
        
@MODELS.register_module()
class VQKDSwin(VQKD):
    """Vector-Quantized Knowledge Distillation.

    The module only contains encoder and VectorQuantizer part
    Modified from https://github.com/microsoft/unilm/blob/master/beit2/modeling_vqkd.py

    Args:
        encoder_config (dict): The config of encoder.
        decoder_config (dict, optional): The config of decoder. Currently,
            VQKD only support to build encoder. Defaults to None.
        num_embed (int): Number of embedding vectors in the codebook. Defaults
            to 8192.
        embed_dims (int) : The dimension of embedding vectors in the codebook.
            Defaults to 32.
        decay (float): The decay parameter of EMA. Defaults to 0.99.
        beta (float): The mutiplier for VectorQuantizer loss. Defaults to 1.
        quantize_kmeans_init (bool): Whether to use k-means to initialize the
            VectorQuantizer. Defaults to True.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """  # noqa: E501

    def __init__(self,
                 encoder_config: dict,
                 decoder_config: Optional[dict] = None,
                 num_embed: int = 8192,
                 embed_dims: int = 32,
                 decay: float = 0.99,
                 beta: float = 1.0,
                 quantize_kmeans_init: bool = True,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.encoder = BEiTSwin(**encoder_config)
        if decoder_config is not None:
            self.decoder = BEiTSwin(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            num_embed=num_embed,
            embed_dims=embed_dims,
            beta=beta,
            decay=decay,
            kmeans_init=quantize_kmeans_init,
        )

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.encoder.arch_settings['embed_dims'],
                      self.encoder.arch_settings['embed_dims']), nn.Tanh(),
            nn.Linear(self.encoder.arch_settings['embed_dims'], embed_dims))


class BEiTPretrainSwin(BEiTPretrainViT):

    def __init__(
        self,
        encoder_config: dict,
        decoder_config: Optional[dict]=None,
        num_embed: int=8192,
        embed_dims: int=32,
        decay: float=0.99,
        beta: float=1.0,
        quantize_kmeans_init: bool=True,
        init_cfg: Optional[dict]=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.encoder = BEiTSwin(**encoder_config)
        if decoder_config is not None:
            self.decoder = BEiTSwin(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            num_embed=num_embed,
            embed_dims=embed_dims,
            beta=beta,
            decay=decay,
            kmeans_init=quantize_kmeans_init,
        )

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(self.encoder.arch_settings['embed_dims'],
                      self.encoder.arch_settings['embed_dims']), nn.Tanh(),
            nn.Linear(self.encoder.arch_settings['embed_dims'], embed_dims))
