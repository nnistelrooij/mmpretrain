# _base_ = '../../../configs/swin_transformer_v2/swinv2-tiny-w16_16xb64_in1k-256px.py'
_base_ = '../../../configs/simmim/benchmarks/swin-base-w7_8xb256-coslr-100e_in1k.py'

binary_model = _base_.model
binary_model.backbone.drop_path_rate = 0.8
binary_model.backbone.pad_small_map = True
binary_model.backbone.img_size = 256
binary_model.head.num_classes = 2
binary_model.train_cfg = None

mix_features = False
mix_outputs = True
attributes = ['Caries', 'Deep Caries', 'Impacted', 'Periapical Lesion']
fold = 2
checkpoints = []
for attribute in attributes:
    path = f'/mnt/diag/DENTEX/dentex/work_dirs/opg_crops_fold_dentex_diagnosis_{fold}_{attribute}/epoch_80.pth'
    checkpoints.append(path)

model = dict(
    type='ImageClassifier',
    data_preprocessor=dict(
        mean=[115.71165283, 115.71165283, 11.35115504],
        std=[45.16004368, 45.16004368, 52.58988321],
    ),
    backbone=dict(
        _delete_=True,
        type='MultilabelConvNeXts',
        classifiers={
            attr: {**binary_model, 'pretrained': checkpoint}
            for attr, checkpoint in zip(attributes, checkpoints)
        },
        pretrained='work_dirs/simmim_swin/epoch_100.pth',
        mix_features=mix_features,
        mix_outputs=mix_outputs,
    ),
    neck=None,
    head=dict(
        _delete_=True,
        type='CSRAMultiTaskHead',
        task_classes={k: 1 for k in attributes},
        # loss=dict(
        #     type='LabelSmoothLoss',
        #     use_sigmoid=True,
        #     label_smooth_val=0.1,
        #     mode='multi_label',
        # ),
        # loss=dict(type='AsymmetricLoss'),
        loss=dict(type='FocalLoss'),
        use_multilabel=True,
        loss_weights=[1.0]*len(attributes),
        in_channels=1024 * (1 + len(attributes) * mix_features),
        num_heads=6,
        lam=0.1,
    ),
)

tta_model = dict(
    type='AverageClsScoreTTA',
    data_preprocessor=dict(
        mean=[115.71165283, 115.71165283, 11.35115504],
        std=[45.16004368, 45.16004368, 52.58988321],
    ),
)
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=binary_model.backbone.img_size + 32,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=binary_model.backbone.img_size),
    dict(type='TestTimeAug', transforms=[
        [
            {'type': 'RandomFlip', 'prob': 1.0},
            {'type': 'RandomFlip', 'prob': 0.0},
        ],
        [{
            'type': 'PackMultiTaskInputs',
            'multi_task_fields': ('gt_label', )
        }],
    ]),
]
