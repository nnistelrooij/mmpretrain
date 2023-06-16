# _base_ = f'../../../configs/convnext_v2/convnext-v2-base_32xb32_in1k.py'
_base_ = '../../../configs/swin_transformer_v2/swinv2-tiny-w16_16xb64_in1k-256px.py'

custom_imports = dict(
    imports=[
        'projects.DENTEX2.datasets',
        'projects.DENTEX2.datasets.samplers.class_aware_sampler',
        'projects.DENTEX2.convnext.csra_head',
        'projects.DENTEX2.datasets.transforms.processing',
        'projects.DENTEX2.evaluation.metrics.multi_tasks_aggregate',
        'projects.DENTEX2.visualization.visualizer',
    ],
    allow_failed_imports=False,
)

data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
export = 'fdi-checkedv2'
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

supervise_number = False

woll = dict(type='PackMultiTaskInputs', multi_task_fields=('gt_label',))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedClassPreservingCrop', scale=256),
    dict(type='NNUNetSpatialIntensityAugmentations'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    woll,
]
# train_pipeline = _base_.train_dataloader.dataset.pipeline[:-1] + [woll]
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
        supervise_number=supervise_number,
        pipeline=train_pipeline
    ),
)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.1,
        dataset=dict(
            type='ToothCropMultitaskDataset',
            data_root=data_root,
            data_prefix=data_prefix,
            pred_file='full_pred.json',
            ann_file=ann_prefix + f'train{fold}.json',
            metainfo=dict(classes=classes, attributes=attributes),
            extend=0.1,
            supervise_number=supervise_number,
            pipeline=train_pipeline,
        ),
    ),
)

val_pipeline = _base_.val_dataloader.dataset.pipeline[:-1] + [woll]
val_dataloader = dict(dataset=dict(
    type='ToothCropMultitaskDataset',
    data_root=data_root,
    data_prefix=data_prefix,
    pred_file='full_pred.json',
    ann_file=ann_prefix + f'val{fold}.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
    supervise_number=supervise_number,
    pipeline=val_pipeline,
))

test_pipeline = _base_.test_dataloader.dataset.pipeline[:-1] + [woll]
test_dataloader = dict(
    sampler=dict(shuffle=True),
    dataset=dict(
        type='ToothCropMultitaskDataset',
        data_root=data_root,
        data_prefix=data_prefix,
        pred_file='full_pred.json',
        ann_file=ann_prefix + f'train{fold}.json',
        metainfo=dict(classes=classes, attributes=attributes),
        supervise_number=supervise_number,
        extend=0.1,
        pipeline=test_pipeline,
    ),
)

# auto_scale_lr = dict(enable=True)
if _base_.model.backbone.type == 'ConvNeXt' and _base_.model.backbone.arch == 'base':
    load_from = 'checkpoints/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k_20230104-c48d16a5.pth'
    # load_from = 'checkpoints/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth'
elif _base_.model.backbone.type == 'ConvNeXt' and _base_.model.backbone.arch == 'tiny':
    load_from = 'checkpoints/convnext-v2-tiny_fcmae-in21k-pre_3rdparty_in1k_20230104-8cc8b8f2.pth'
elif _base_.model.backbone.type == 'EfficientNetV2':
    load_from = 'checkpoints/efficientnetv2-s_in21k-pre-3rdparty_in1k_20221220-7a7c8475.pth'
else:
    load_from = 'checkpoints/swinv2-tiny-w16_3rdparty_in1k-256px_20220803-9651cdd7.pth'
    in_channels = 768


data_preprocessor = dict(num_classes=len(attributes) - 1)
model = dict(
    backbone=dict(gap_before_final_norm=False) if _base_.model.backbone.type == 'ConvNeXt' else dict(),
    neck=dict() if _base_.model.backbone.type == 'ConvNeXt' else None,
    head=dict(
        _delete_=True,
        type='CSRAMultiTaskHead',
        task_classes={
            **{a: 1 for a in attributes[1:]},
            **({'Number': 8} if supervise_number else {}),
        },
        loss=dict(
            type='LabelSmoothLoss',
            use_sigmoid=True,
            label_smooth_val=0.1,
            mode='multi_label',
        ),
        use_multilabel=True,
        loss_weights=[0.5, 1.0, 0.5, 1.0] + ([1.0] if supervise_number else []),
        in_channels=in_channels,
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


val_evaluator = dict(
    _delete_=True,
    type='MultiTasksAggregateMetric',
    task_metrics={
        **{attr: [
            dict(type='SingleLabelMetric', num_classes=2),
            dict(type='SingleLabelMetric', num_classes=2, average=None),
            dict(type='ConfusionMatrix', num_classes=2),
        ]
        for attr in attributes[1:]},
        **({'Number': [
            dict(type='SingleLabelMetric', num_classes=8),
            dict(type='SingleLabelMetric', num_classes=8, average=None),
            dict(type='ConfusionMatrix', num_classes=8),
        ]} if supervise_number else {}),
    }
)
test_evaluator = dict(
    _delete_=True,
    type='MultiTasksAggregateMetric',
    task_metrics={
        **{attr: [
            dict(type='SingleLabelMetric', num_classes=2),
            dict(type='SingleLabelMetric', num_classes=2, average=None),
            dict(type='ConfusionMatrix', num_classes=2),
        ]
        for attr in attributes[1:]},
        **({'Number': [
            dict(type='SingleLabelMetric', num_classes=8),
            dict(type='SingleLabelMetric', num_classes=8, average=None),
            dict(type='ConfusionMatrix', num_classes=8),
        ]} if supervise_number else {}),
    }
)

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=3,        
        save_best='single-label/f1-score_classwise',
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

optim_wrapper = dict(
    optimizer=dict(lr=0.001, weight_decay=0.001),
    clip_grad=(
        dict(_delete_=True, max_norm=5.0)
        if _base_.model.backbone.type == 'ConvNeXt' else
        dict(max_norm=5.0)
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        end=20,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=1e-05, by_epoch=True, begin=20)
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)

custom_hooks = [
    # dict(type='ClassCountsHook', num_classes=len(attributes) - 1),
    # dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL'),
]

work_dir = f'work_dirs/opg_crops_fold{fold}'
