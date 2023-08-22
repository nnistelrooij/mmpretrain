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
# checkpoints = [
#     'work_dirs/opg_crops_fold_diagnosis_0_Caries_1_swin/best_positive-label_auc_epoch_56.pth',
#     'work_dirs/opg_crops_fold_diagnosis_0_Deep Caries_1_swin/best_positive-label_auc_epoch_14.pth',
#     'work_dirs/opg_crops_fold_diagnosis_0_Impacted_1_swin/best_positive-label_auc_epoch_20.pth',
#     'work_dirs/opg_crops_fold_diagnosis_0_Periapical Lesion_1_swin/best_positive-label_auc_epoch_11.pth',
# ]
# checkpoints = [
#     'work_dirs/opg_crops_fold_diagnosis_1_Caries_1_swin/best_positive-label_auc_epoch_76.pth',
#     'work_dirs/opg_crops_fold_diagnosis_1_Deep Caries_1_swin/best_positive-label_auc_epoch_32.pth',
#     'work_dirs/opg_crops_fold_diagnosis_1_Impacted_1_swin/best_positive-label_auc_epoch_45.pth',
#     'work_dirs/opg_crops_fold_diagnosis_1_Periapical Lesion_1_swin/best_positive-label_auc_epoch_19.pth',
# ]
# checkpoints = [
#     'work_dirs/opg_crops_fold_diagnosis_2_Caries_1_swin/best_positive-label_auc_epoch_55.pth',
#     'work_dirs/opg_crops_fold_diagnosis_2_Deep Caries_1_swin/best_positive-label_auc_epoch_9.pth',
#     'work_dirs/opg_crops_fold_diagnosis_2_Impacted_1_swin/best_positive-label_auc_epoch_44.pth',
#     'work_dirs/opg_crops_fold_diagnosis_2_Periapical Lesion_1_swin/best_positive-label_auc_epoch_72.pth',
# ]
# checkpoints = [
#     'work_dirs/opg_crops_fold_diagnosis_3_Caries_1_swin/best_positive-label_auc_epoch_64.pth',
#     'work_dirs/opg_crops_fold_diagnosis_3_Deep Caries_1_swin/best_positive-label_auc_epoch_20.pth',
#     'work_dirs/opg_crops_fold_diagnosis_3_Impacted_1_swin/best_positive-label_auc_epoch_37.pth',
#     'work_dirs/opg_crops_fold_diagnosis_3_Periapical Lesion_1_swin/best_positive-label_auc_epoch_13.pth',
# ]
# checkpoints = [
#     'work_dirs/opg_crops_fold_diagnosis_4_Caries_1_swin/best_positive-label_auc_epoch_71.pth',
#     'work_dirs/opg_crops_fold_diagnosis_4_Deep Caries_1_swin/best_positive-label_auc_epoch_13.pth',
#     'work_dirs/opg_crops_fold_diagnosis_4_Impacted_1_swin/best_positive-label_auc_epoch_68.pth',
#     'work_dirs/opg_crops_fold_diagnosis_4_Periapical Lesion_1_swin/best_positive-label_auc_epoch_20.pth',
# ]
# checkpoints = [
#     'work_dirs/opg_crops_fold_diagnosis_0_Caries_1_swin/epoch_80.pth',
#     'work_dirs/opg_crops_fold_diagnosis_0_Deep Caries_1_swin/epoch_80.pth',
#     'work_dirs/opg_crops_fold_diagnosis_0_Impacted_1_swin/epoch_80.pth',
#     'work_dirs/opg_crops_fold_diagnosis_0_Periapical Lesion_1_swin/epoch_80.pth',
# ]
checkpoints = [
    'work_dirs/opg_crops_folddentex_diagnosis_0_Caries_0_swin/epoch_80.pth',
    'work_dirs/opg_crops_folddentex_diagnosis_0_Deep Caries_0_swin/epoch_80.pth',
    'work_dirs/opg_crops_folddentex_diagnosis_0_Impacted_0_swin/epoch_80.pth',
    'work_dirs/opg_crops_folddentex_diagnosis_0_Periapical Lesion_0_swin/epoch_80.pth',
]


for i, ckpt in enumerate(checkpoints):
    checkpoints[i] = '/mnt/diag/DENTEX/dentex/' + ckpt

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
