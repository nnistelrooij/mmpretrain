# _base_ = '../../../configs/swin_transformer_v2/swinv2-tiny-w16_16xb64_in1k-256px.py'
_base_ = '../../../configs/swin_transformer/swin-base_16xb64_in1k.py'

binary_model = _base_.model
binary_model.backbone.pad_small_map = True
binary_model.head.num_classes = 2
binary_model.train_cfg = None

mix_features = False
mix_outputs = True
attributes = ['Caries', 'Deep Caries', 'Impacted', 'Periapical Lesion']
checkpoints = [
    'work_dirs/opg_crops_fold_diagnosis_0_Caries_1_swin/best_positive-label_auc_epoch_17.pth',
    'work_dirs/opg_crops_fold_diagnosis_0_Deep Caries_1_swin/best_positive-label_auc_epoch_28.pth',
    'work_dirs/opg_crops_fold_diagnosis_0_Impacted_1_swin/best_positive-label_auc_epoch_14.pth',
    'work_dirs/opg_crops_fold_diagnosis_0_Periapical Lesion_1_swin/best_positive-label_auc_epoch_14.pth',
]

model = dict(
    type='ImageClassifier',
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
        loss=dict(type='AsymmetricLoss'),
        # loss=dict(type='FocalLoss'),
        use_multilabel=True,
        loss_weights=[1.0]*len(attributes),
        in_channels=1024 * (1 + len(attributes) * mix_features),
        num_heads=6,
        lam=0.1,
    ),
)
