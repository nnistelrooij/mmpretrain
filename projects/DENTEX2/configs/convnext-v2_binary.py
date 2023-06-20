# _base_ = '../../../configs/convnext_v2/convnext-v2-tiny_32xb32_in1k.py'
# _base_ = '../../../configs/efficientnet_v2/efficientnetv2-s_8xb32_in21k.py'
# _base_ = '../../../configs/efficientnet/efficientnet-b7_8xb32_in1k.py'
# _base_ = '../../../configs/swin_transformer_v2/swinv2-tiny-w16_16xb64_in1k-256px.py'
_base_ = '../../../configs/beitv2/benchmarks/beit-base-p16_8xb128-coslr-100e_in1k.py'

custom_imports = dict(
    imports=[
        'projects.DENTEX2.datasets',
        'projects.DENTEX2.datasets.samplers.class_aware_sampler',
        'projects.DENTEX2.datasets.transforms',
        'projects.DENTEX2.hooks.class_counts_hook',
        'projects.DENTEX2.evaluation.metrics.positive_label',
    ],
    allow_failed_imports=False,
)

data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
export = 'fdi-checkedv2'
fold = '_diagnosis_0'
run = 1
multilabel = False
data_prefix = data_root + 'images'
ann_prefix = data_root + f'releases/{export}/other_formats/coco/'

pretrain_checkpoint = 'work_dirs/beit-v2/pretrained.pth'

classes = [
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
]
attributes = [
    'Control',
    # 'Caries',
    # 'Deep Caries',
    'Impacted',
    # 'Periapical Lesion',
]
# classes = attributes

train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.1,
        dataset=dict(
            type='ToothCropDataset',
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_prefix + f'train{fold}.json',
            pred_file='full_pred.json',
            metainfo=dict(classes=classes, attributes=attributes),
            extend=0.1,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='ResizeEdge',
                    scale=256,
                    edge='short',
                    backend='pillow',
                    interpolation='bicubic'),
                dict(type='CenterCrop', crop_size=256),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='PackInputs')
            ],
        ),
    ),
)

batch_size = 128
train_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(_delete_=True, type='ClassAwareSampler'),
    dataset=dict(
        _delete_=True,
        type='ToothCropDataset',
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file=ann_prefix + f'train{fold}.json',
        pred_file='full_pred.json',
        metainfo=dict(classes=classes, attributes=attributes),
        extend=0.1,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedClassPreservingCrop', scale=256),
            # dict(type='SeparateOPGMask'),
            dict(type='NNUNetSpatialIntensityAugmentations'),
            # *([dict(type='MaskTooth')] if 'Caries' in attributes[-1] else []),
            # dict(
            #     type='ResizeEdge',
            #     scale=256,
            #     edge='short',
            #     backend='pillow',
            #     interpolation='bicubic'),
            # dict(type='CenterCrop', crop_size=224),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs')
        ],
    ),
)

val_dataloader = dict(dataset=dict(
    type='ToothCropDataset',
    data_root=data_root,
    data_prefix=data_prefix,
    ann_file=ann_prefix + f'val{fold}.json',
    pred_file='full_pred.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
))

test_dataloader = dict(dataset=dict(
    type='ToothCropDataset',
    data_root=data_root,
    data_prefix=data_prefix,
    ann_file=ann_prefix + f'val{fold}.json',
    pred_file='full_pred.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
))

data_preprocessor = dict(num_classes=2)
model = dict(
    data_preprocessor=(
        dict(
            mean=[118.4439122, 118.4439122, 20.45650507],
            std=[45.39504314, 45.39504314, 69.26716533],
        ) if pretrain_checkpoint else dict()
    ),
    backbone=dict(
        init_cfg=(
            dict(type='Pretrained', checkpoint='work_dirs/beit-v2/pretrained.pth', prefix='backbone')
            if pretrain_checkpoint else
            None
        ),
    ),
    head=dict(
        num_classes=2,
        loss=dict(_delete_=True, type='LabelSmoothLoss', label_smooth_val=0.1),
        # loss=dict(_delete_=True, type='FocalLoss'),
    ),
    train_cfg=None,
)
# auto_scale_lr = dict(enable=True)

if not pretrain_checkpoint:
    if _base_.model.backbone.type == 'ConvNeXt' and _base_.model.backbone.arch == 'base':
        load_from = 'checkpoints/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k_20230104-c48d16a5.pth'
        # load_from = 'checkpoints/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth'
    elif _base_.model.backbone.type == 'ConvNeXt' and _base_.model.backbone.arch == 'tiny':
        load_from = 'checkpoints/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
    elif _base_.model.backbone.type == 'EfficientNetV2':
        load_from = 'checkpoints/efficientnetv2-s_in21k-pre-3rdparty_in1k_20221220-7a7c8475.pth'
    else:
        load_from = 'checkpoints/swinv2-tiny-w16_3rdparty_in1k-256px_20220803-9651cdd7.pth'

optim_wrapper = dict(
    optimizer=dict(type='Lion', lr=0.001, weight_decay=0.001),
    clip_grad=(
        dict(_delete_=True, max_norm=5.0)
        if _base_.model.backbone.type == 'ConvNeXt' else
        dict(max_norm=5.0)
    ),
    accumulative_counts=128 // batch_size,
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=5 * (batch_size // 32),
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=5 * (batch_size // 32))
]

val_evaluator = [
    dict(type='PositiveLabelMetric', average=None),
    dict(type='PositiveLabelMetric'),
    dict(type='ConfusionMatrix', num_classes=2),
]
test_evaluator = [
    dict(type='PositiveLabelMetric', average=None),
    dict(type='PositiveLabelMetric'),
    dict(type='ConfusionMatrix', num_classes=2),
]

train_cfg = dict(max_epochs=100)
custom_hooks = [
    dict(type='ClassCountsHook', num_classes=2),
    # dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL'),
]

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=1,        
        save_best='single-label/f1-score_positive',
        rule='greater',
    ),
)
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])

work_dir = f'work_dirs/opg_crops_fold{fold}_{attributes[-1]}_{run}_swin'
