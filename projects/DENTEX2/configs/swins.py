_base_ = [
    '../../../configs/swin_transformer_v2/swinv2-tiny-w16_16xb64_in1k-256px.py'
]

binary_model = _base_.model
binary_model.backbone.pad_small_map = True
binary_model.head.num_classes = 2
binary_model.train_cfg = None

mix_features = True
mix_outputs = True
attributes = ['Caries', 'Deep Caries', 'Impacted', 'Periapical Lesion']
checkpoints = [
    'work_dirs/opg_crops_fold_diagnosis_0_Caries_1_swin/epoch_100.pth',
    'work_dirs/opg_crops_fold_diagnosis_0_Deep Caries_1_swin/best_single-label_f1-score_positive_epoch_81.pth',
    'work_dirs/opg_crops_fold_diagnosis_0_Impacted_0_swin/epoch_100.pth',
    'work_dirs/opg_crops_fold_diagnosis_0_Periapical Lesion_0_swin/best_single-label_f1-score_positive_epoch_83.pth',
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
        pretrained='checkpoints/swinv2-tiny-w16_3rdparty_in1k-256px_20220803-9651cdd7.pth',
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
        in_channels=768 * (1 + len(attributes) * mix_features),
        num_heads=6,
        lam=0.1,
    ),
)
