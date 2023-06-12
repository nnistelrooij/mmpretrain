_base_ = '../../../configs/convnext_v2/convnext-v2-base_32xb32_in1k.py'
# _base_ = '../../../configs/efficientnet_v2/efficientnetv2-s_8xb32_in1k-384px.py'

custom_imports = dict(
    imports=[
        'projects.DENTEX2.datasets',
        'projects.DENTEX2.datasets.samplers.class_aware_sampler',
        'projects.DENTEX2.datasets.transforms.processing',
        'projects.DENTEX2.hooks.class_counts_hook',
    ],
    allow_failed_imports=False,
)

data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
export = 'fdi-checkedv2'
fold = '_diagnosis_4'
multilabel = False
data_prefix = data_root + 'images'
ann_prefix = data_root + f'releases/{export}/other_formats/coco/'

classes = [
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
]
attributes = [
    'Control',
    'Caries',
    # 'Deep Caries',
    # 'Impacted',
    # 'Periapical Lesion',
]
# classes = attributes

train_dataloader = dict(dataset=dict(
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
            dict(type='CenterCrop', crop_size=224),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs')
        ],
    )
))

train_dataloader = dict(
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
            dict(
                type='RandomResizedClassPreservingCrop',
                scale=224,
            ),
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
    head=dict(
        num_classes=2,
        loss=dict(_delete_=True, type='FocalLoss'),
    ),
    train_cfg=None,
)
# auto_scale_lr = dict(enable=True)
if _base_.model.backbone.arch == 'base':
    load_from = 'checkpoints/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k_20230104-c48d16a5.pth'
    # load_from = 'checkpoints/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth'
elif _base_.model.backbone.arch == 'tiny':
    load_from = 'checkpoints/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
else:
    load_from = 'checkpoints/efficientnetv2-s_in21k-pre-3rdparty_in1k_20221220-7a7c8475.pth'

optim_wrapper = dict(optimizer=dict(lr=0.001))

val_evaluator = [
    dict(type='SingleLabelMetric', average=None),
    dict(type='SingleLabelMetric'),
    dict(type='ConfusionMatrix', num_classes=2),
]
test_evaluator = [
    dict(type='SingleLabelMetric', average=None),
    dict(type='SingleLabelMetric'),
    dict(type='ConfusionMatrix', num_classes=2),
]

train_cfg = dict(max_epochs=100)
custom_hooks = [dict(type='ClassCountsHook', num_classes=2)]

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=1,        
        save_best='single-label/f1-score',
        rule='greater',
    ),
)
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])

work_dir = f'work_dirs/opg_crops_fold{fold}_{attributes[-1]}'
