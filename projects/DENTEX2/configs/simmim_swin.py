_base_ = '../../../configs/simmim/simmim_swin-base-w6_8xb256-amp-coslr-100e_in1k-192px.py'
# _base_ = '../../../configs/simmim/simmim_swin-large-w12_16xb128-amp-coslr-800e_in1k-192px.py'


custom_imports = dict(
    imports=[
        'projects.DENTEX2.datasets',
        'projects.DENTEX2.datasets.samplers.class_aware_sampler',
        'projects.DENTEX2.datasets.transforms',
        # 'projects.DENTEX2.hooks.class_counts_hook',
        # 'projects.DENTEX2.evaluation.metrics.positive_label',
    ],
    allow_failed_imports=False,
)

img_size = 192
model=dict(    
    data_preprocessor=dict(
        mean=[115.71165283, 115.71165283, 11.35115504],
        std=[45.16004368, 45.16004368, 52.58988321],
    ),
    backbone=dict(
        # type='SimMIMSwinTransformerV2',
        img_size=img_size,
    ),
)

data_root = '/home/mkaailab/.darwin/datasets/mucoaid/dentexv2/'
# data_root = '/mnt/z/nielsvannistelrooij/dentex/darwin/'
export = 'curated-odonto'
split = 'val_all_dentex_diagnosis_0'
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

batch_size = 128
gpus = 2
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
            # ann_file='/home/mkaailab/Documents/DENTEX/dentex/pred.json',
            ann_file=ann_prefix + f'{split}.json',
            pred_file='/home/mkaailab/Documents/DENTEX/dentex/pred_odo.json',
            metainfo=dict(classes=classes, attributes=attributes),
            extend=0.1,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='RandomResizedClassPreservingCrop', scale=img_size),
                dict(type='NNUNetSpatialIntensityAugmentations'),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(
                    type='SimMIMMaskGenerator',
                    input_size=img_size,
                    mask_patch_size=32,
                    model_patch_size=4,
                    mask_ratio=0.6,
                ),
                dict(type='PackInputs'),
            ],
        ),
    ),
)

optim_wrapper = dict(
    # optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.001),
    # clip_grad=(
    #     dict(_delete_=True, max_norm=5.0)
    #     if _base_.model.backbone.type == 'ConvNeXt' else
    #     dict(max_norm=5.0)
    # ),
    accumulative_counts=2048 // batch_size // gpus,
)

visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])

default_hooks = dict(checkpoint=dict(
    max_keep_ckpts=1,
    save_best='loss',
))

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=10,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-6,
        by_epoch=True,
        begin=10,
    ),
]

train_cfg = dict(max_epochs=100)

load_from = 'checkpoints/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192_20220916-a0e931ac.pth'
# load_from = 'checkpoints/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220916-4ad216d3.pth'

work_dir = 'work_dirs/simmim_swin'
