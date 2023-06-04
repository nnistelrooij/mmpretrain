_base_ = '../../../configs/convnext_v2/convnext-v2-base_32xb32_in1k.py'

custom_imports = dict(
    imports=[
        'projects.DENTEX2.datasets',
        'projects.DENTEX2.datasets.samplers.class_aware_sampler',
        'projects.DENTEX2.convnext.csra_head',
        'projects.DENTEX2.evaluation.metrics.multi_tasks_aggregate',
        'projects.DENTEX2.visualization.visualizer',
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

woll = dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label',))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    woll,
]
train_dataloader = dict(    
    sampler=dict(_delete_=True, type='ClassAwareSampler'),
    dataset=dict(
        _delete_=True,
        type='ToothCropMultitaskDataset',
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file=ann_prefix + f'train{fold}.json',
        metainfo=dict(classes=classes, attributes=attributes),
        extend=0.1,
        pipeline=train_pipeline
    ),
)
train_dataloader = dict(dataset=dict(
    _delete_=True,
    type='ClassBalancedDataset',
    oversample_thr=0.1,
    dataset=dict(
        type='ToothCropMultitaskDataset',
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file=ann_prefix + f'train{fold}.json',
        metainfo=dict(classes=classes, attributes=attributes),
        extend=0.1,
        pipeline=train_pipeline
    ),
))

val_pipeline = _base_.val_dataloader.dataset.pipeline[:-1] + [woll]
val_dataloader = dict(dataset=dict(
    type='ToothCropMultitaskDataset',
    data_root=data_root,
    data_prefix=data_prefix,
    ann_file=ann_prefix + f'val{fold}.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
    pipeline=val_pipeline,
))

test_pipeline = _base_.test_dataloader.dataset.pipeline[:-1] + [woll]
test_dataloader = dict(
    sampler=dict(shuffle=True),
    dataset=dict(
        type='ToothCropMultitaskDataset',
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file=ann_prefix + f'val{fold}.json',
        metainfo=dict(classes=classes, attributes=attributes),
        extend=0.1,
        pipeline=test_pipeline,
    ),
)

data_preprocessor = dict(num_classes=len(attributes) - 1)
model = dict(
    backbone=dict(gap_before_final_norm=False),
    head=dict(
        _delete_=True,
        type='CSRAMultiTaskHead',
        tasks=attributes[1:],
        loss=dict(type='CrossEntropyLoss'),
        in_channels=1024,
        num_heads=6,
        lam=0.1,
    ),

    # head=dict(
    #     type='MultiTaskHead',
    #     task_heads={
    #         attr: dict(
    #             # type='LinearClsHead',
    #             type='CSRAClsHead',
    #             lam=0.1,
    #             num_heads=6,
    #             num_classes=2,
    #             loss=dict(type='FocalLoss', loss_weight=loss_weight),
    #         ) for attr, loss_weight in zip(
    #             attributes[1:], [1.0, 1.0, 0.2, 0.2],
    #         )
    #     },
    # ),
    train_cfg=None,
)
# auto_scale_lr = dict(enable=True)
load_from = 'checkpoints/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k_20230104-c48d16a5.pth'
# load_from = 'work_dirs/opg_crops_fold_diagnosis_0_multilabel/epoch_13.pth'

val_evaluator = dict(
    _delete_=True,
    type='MultiTasksAggregateMetric',
    task_metrics={
        attr: [
            dict(type='SingleLabelMetric', num_classes=2),
            dict(type='SingleLabelMetric', num_classes=2, average=None),
        ]
        for attr in attributes[1:]
    }
)
test_evaluator = dict(
    _delete_=True,
    type='MultiTasksAggregateMetric',
    task_metrics={
        attr: [
            dict(type='SingleLabelMetric'),
            dict(type='SingleLabelMetric', average=None),
        ]
        for attr in attributes[1:]
    }
)

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=3,        
        save_best='single-label/f1-score',
        rule='greater',
    ),
)
visualizer = dict(
    type='MultiTaskVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        end=10,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=1e-05, by_epoch=True, begin=10)
]
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)

work_dir = f'work_dirs/opg_crops_fold{fold}_multitask'
