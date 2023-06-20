_base_ = '../../../configs/beitv2/benchmarks/beit-base-p16_8xb128-coslr-100e_in1k.py'

custom_imports = dict(
    imports=[
        'projects.DENTEX2.datasets',
        'projects.DENTEX2.datasets.samplers.class_aware_sampler',
        'projects.DENTEX2.beitv2.beit_swin',
        'projects.DENTEX2.datasets.transforms',
        # 'projects.DENTEX2.hooks.class_counts_hook',
        # 'projects.DENTEX2.evaluation.metrics.positive_label',
    ],
    allow_failed_imports=False,
)


model = dict(
    data_preprocessor=dict(
        mean=[118.4439122, 118.4439122, 20.45650507],
        std=[45.39504314, 45.39504314, 69.26716533],
    ),
    train_cfg=None,
)


data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
data_root = '/mnt/z/nielsvannistelrooij/dentex/darwin/'
export = 'fdi-checkedv2'
fold = '_diagnosis_0'
run = 0
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

batch_size = 128
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=0.1,
        dataset=dict(
            type='ToothCropDataset',
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_prefix + f'train{fold}.json',
            pred_file=data_root + 'full_pred.json',
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
    ),
)

val_dataloader = dict(dataset=dict(
    type='ToothCropDataset',
    data_root=data_root,
    data_prefix=data_prefix,
    ann_file=ann_prefix + f'val{fold}.json',
    pred_file=data_root + 'full_pred.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
))

test_dataloader = dict(dataset=dict(
    type='ToothCropDataset',
    data_root=data_root,
    data_prefix=data_prefix,
    ann_file=ann_prefix + f'val{fold}.json',
    pred_file=data_root + 'full_pred.json',
    metainfo=dict(classes=classes, attributes=attributes),
    extend=0.1,
))


optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.001),
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


train_cfg = dict(max_epochs=800)
