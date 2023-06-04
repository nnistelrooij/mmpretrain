_base_ = '../../../configs/convnext_v2/convnext-v2-base_32xb32_in1k.py'

custom_imports = dict(
    imports=[
        'projects.DENTEX.datasets',
    ],
    allow_failed_imports=False,
)

data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
export = 'fdi-checked'
fold = '_diagnosis_0'
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
    'Deep Caries',
    'Impacted',
    'Periapical Lesion',
]
# classes = attributes

train_dataloader = dict(dataset=dict(
    _delete_=True,
    type='ClassBalancedDataset',
    oversample_thr=0.1,
    dataset=dict(
        type='ToothCropMultilabelDataset',
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file=ann_prefix + f'train{fold}.json',
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

val_dataloader = dict(dataset=dict(
    type='ToothCropMultilabelDataset',
    data_root=data_root,
    data_prefix=data_prefix,
    ann_file=ann_prefix + f'val{fold}.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
))

test_dataloader = dict(dataset=dict(
    type='ToothCropMultilabelDataset',
    data_root=data_root,
    data_prefix=data_prefix,
    ann_file=ann_prefix + f'val{fold}.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
))

data_preprocessor = dict(num_classes=len(attributes) - 1)
model = dict(
    backbone=dict(
        gap_before_final_norm=False,
    ),
    head=dict(
        type='CSRAClsHead',
        num_classes=len(attributes) - 1,
        loss=dict(_delete_=True, type='FocalLoss'),
        num_heads=6,
        lam=0.1,
    ),
    train_cfg=None,
)
# auto_scale_lr = dict(enable=True)
load_from = 'checkpoints/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k_20230104-c48d16a5.pth'
# load_from = 'work_dirs/opg_crops_fold_diagnosis_0_multilabel/epoch_13.pth'

val_evaluator = [
    dict(type='MultiLabelMetric', average=None),
    dict(type='MultiLabelMetric'),
]
test_evaluator = [
    dict(type='MultiLabelMetric', average=None),
    dict(type='MultiLabelMetric'),
]

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=3,        
        save_best='multi-label/f1-score',
        rule='greater',
    ),
)
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])

work_dir = f'work_dirs/opg_crops_fold{fold}_multilabel'
