_base_ = ['./swins.py']


custom_imports = dict(
    imports=[
        'projects.DENTEX2.datasets',
        'projects.DENTEX2.datasets.transforms.processing',
        'projects.DENTEX2.mlp',
        'projects.DENTEX2.convnext.csra_head',
        'projects.DENTEX2.visualization.visualizer',
        'projects.DENTEX2.evaluation.metrics',
        'projects.DENTEX2.convnext.multilabel',
    ],
    allow_failed_imports=False,
)


data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
export = 'fdi-checkedv2'
fold = '_diagnosis_0'
run = 0
multilabel = False
supervise_number = False
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
batch_size = 128
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.1,
        dataset=dict(
            type='ToothCropMultitaskDataset',
            supervise_number=supervise_number,
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
                woll,
            ],
        ),
    ),
)

val_dataloader = dict(dataset=dict(
    type='ToothCropMultitaskDataset',
    supervise_number=supervise_number,
    data_root=data_root,
    data_prefix=data_prefix,
    ann_file=ann_prefix + f'val{fold}.json',
    pred_file='full_pred.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
    pipeline=_base_.val_dataloader.dataset.pipeline[:-1] + [woll]
))

test_dataloader = dict(dataset=dict(
    type='ToothCropMultitaskDataset',
    supervise_number=supervise_number,
    data_root=data_root,
    data_prefix=data_prefix,
    ann_file=ann_prefix + f'val{fold}.json',
    pred_file='full_pred.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
    pipeline=_base_.test_dataloader.dataset.pipeline[:-1] + [woll]
))

data_preprocessor = dict(num_classes=len(attributes) - 1)

# auto_scale_lr = dict(enable=True)

val_evaluator = [
    dict(
        type='MultiTasksAggregateMetric',
        task_metrics={
            attr: [
                dict(type='PositiveLabelMetric', num_classes=2, prefix=attr),
                dict(type='PositiveLabelMetric', num_classes=2, prefix=attr, average=None),
                dict(type='ConfusionMatrix', num_classes=2),
            ]
            for attr in attributes[1:]
        }
    ),
    dict(type='BinaryConfusionMatrix', num_classes=2, prefix='binary-label'),
    dict(type='BinaryLabelMetric', num_classes=2, prefix='binary-label'),
]
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(lr=0.001, weight_decay=0.001),
    clip_grad=dict(max_norm=5.0),
    accumulative_counts=128 // batch_size,
)
train_cfg = dict(max_epochs=100)

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=1,        
        save_best='binary-label/f1-score',
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

work_dir = f'work_dirs/opg_crops_fold_multitask{fold}_swins'
