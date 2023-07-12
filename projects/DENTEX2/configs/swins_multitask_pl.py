_base_ = ['./swins.py']


custom_imports = dict(
    imports=[
        'projects.DENTEX2.datasets',
        'projects.DENTEX2.datasets.transforms.processing',
        'projects.DENTEX2.datasets.samplers.class_aware_sampler',
        'projects.DENTEX2.mlp',
        'projects.DENTEX2.models',
        'projects.DENTEX2.convnext.csra_head',
        'projects.DENTEX2.visualization.visualizer',
        'projects.DENTEX2.evaluation.metrics',
        'projects.DENTEX2.convnext.multilabel',
    ],
    allow_failed_imports=False,
)


data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
export = 'fdi-checkedv2'
fold = '_diagnosis_4'
run = 0
multilabel = False
supervise_number = False
data_prefix = data_root + 'images'
ann_prefix = data_root + f'releases/{export}/other_formats/coco/'
img_size = 256
diag_drive = False

load_from = 'work_dirs/opg_crops_fold_multitask_diagnosis_4_swins/epoch_60.pth'


_base_.model.backbone.classifiers['Caries'].pop('pretrained')
_base_.model.backbone.classifiers['Deep Caries'].pop('pretrained')
_base_.model.backbone.classifiers['Periapical Lesion'].pop('pretrained')
_base_.model.backbone.classifiers['Impacted'].pop('pretrained')
_base_.model.backbone.pop('pretrained')

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
img_size = 256
train_pipeline = [
    dict(type='RandomToothFlip', prob=0.1),
    dict(type='RandomResizedClassPreservingCrop', scale=img_size),
    # dict(type='SeparateOPGMask'),
    dict(type='NNUNetSpatialIntensityAugmentations'),
    # *([dict(type='MaskTooth')] if 'Caries' in attributes[-1] else []),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    woll,
]

batch_size = 128
train_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(
        _delete_=True,
        type='UnlabeledLabeledSampler',
        sampler='DefaultSampler',
        unlabeled_batches=50,
        batch_size=batch_size,
        shuffle=True,
    ),
    dataset=dict(
        _delete_=True,
        type='LabeledUnlabeledDatasets',
        labeled_dataset=dict(
            type='ClassBalancedDataset',
            oversample_thr=0.1,
            dataset=dict(
                type='MultiImageMixDataset',
                dataset=dict(
                    type='ToothCropMultitaskDataset',
                    supervise_number=supervise_number,
                    data_root=data_root,
                    data_prefix=data_prefix,
                    ann_file=ann_prefix + f'train{fold}.json',
                    pred_file='full_pred.json',
                    omit_file=ann_prefix + f'val{fold}.json',
                    metainfo=dict(classes=classes, attributes=attributes),
                    extend=0.1,
                    pipeline=[dict(type='LoadImageFromFile')],
                ),
                pipeline=train_pipeline,
            ),
        ),
        unlabeled_dataset=dict(
            type='ClassBalancedDataset',
            oversample_thr=0.1,
            dataset=dict(
                type='MultiImageMixDataset',
                dataset=dict(
                    type='ToothCropMultitaskDataset',
                    supervise_number=supervise_number,
                    data_root=data_root,
                    data_prefix=data_prefix.replace('images', 'crop_images'),
                    ann_file='/home/mkaailab/Documents/DENTEX/dentex/diagnosis_all.json',
                    pred_file='/home/mkaailab/Documents/DENTEX/dentex/diagnosis_all.json',
                    omit_file=ann_prefix + f'val_any_diagnosis_0.json',
                    metainfo=dict(classes=classes, attributes=attributes),
                    extend=0.1,
                    pipeline=[dict(type='LoadImageFromFile')],
                ),
                pipeline=train_pipeline,
            ),
        ),
    ),
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=img_size + 32,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=img_size),
    woll,
]
val_dataloader = dict(dataset=dict(
    type='ToothCropMultitaskDataset',
    supervise_number=supervise_number,
    data_root=data_root,
    data_prefix=data_prefix,
    ann_file=ann_prefix + f'val{fold}.json',
    pred_file=ann_prefix + f'val{fold}.json' if export == 'external' else 'full_pred.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
    pipeline=test_pipeline,
))

test_dataloader = dict(
    # sampler=dict(shuffle=True),
    dataset=dict(
        type='ToothCropMultitaskDataset',
        supervise_number=supervise_number,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file=ann_prefix + f'val{fold}.json',
        pred_file=ann_prefix + f'val{fold}.json' if export == 'external' else 'full_pred.json',
        # ann_file=data_root + 'all_pred.json',
        # pred_file=data_root + 'all_pred.json',
        metainfo=dict(classes=classes, attributes=attributes),
        extend=0.1,
        pipeline=test_pipeline,
    ),
)

data_preprocessor = dict(num_classes=len(attributes) - 1)
model = dict(type='PseudoClassifier')

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
# test_evaluator = []

optim_wrapper = dict(
    optimizer=dict(lr=2e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=5.0),
    accumulative_counts=256 // batch_size,
)
train_cfg = dict(max_epochs=60)

warmup_epochs = 5
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        by_epoch=True,
        begin=warmup_epochs,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        max_keep_ckpts=1,        
        save_best='positive-label/f1-score',
        rule='greater',
    ),
    visualization=dict(
        draw=False,
        interval=20,
    ),
)
visualizer = dict(
    type='MultiTaskVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
)

if diag_drive:
    work_dir = f'/mnt/diag/DENTEX/dentex/work_dirs/opg_crops_fold_multitask{fold}_swins'
else:
    work_dir = f'work_dirs/opg_crops_fold_multitask{fold}_swins'
