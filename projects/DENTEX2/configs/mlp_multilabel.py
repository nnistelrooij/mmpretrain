_base_ = ['./mlp.py']


custom_imports = dict(
    imports=[
        'projects.DENTEX.datasets',
        'projects.DENTEX.mlp',
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
        pipeline=_base_.train_pipeline[:3] + _base_.train_pipeline[-1:],
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

# auto_scale_lr = dict(enable=True)

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

# model = dict(head=dict(num_classes=len(attributes) - 1))

work_dir = f'work_dirs/opg_crops_fold{fold}_multilabel'
