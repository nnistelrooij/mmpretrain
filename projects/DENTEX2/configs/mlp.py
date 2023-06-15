_base_ = [
    '../../../configs/swin_transformer_v2/swinv2-tiny-w16_16xb64_in1k-256px.py'
]

base_model = _base_.model
base_model.backbone.pad_small_map = True
base_model.head.num_classes = 2
base_model.train_cfg = None

model = dict(
    type='ImageClassifier',
    backbone=dict(
        _delete_=True,
        type='MultilabelMLP',
        classifiers={
            'Caries': {
                **base_model,
                'pretrained': 'work_dirs/opg_crops_fold_diagnosis_0_Caries_1_swin/epoch_100.pth',
                'init_cfg': None,
            },
            'Deep Caries': {
                **base_model,
                'pretrained': 'work_dirs/opg_crops_fold_diagnosis_0_Deep Caries_1_swin/best_single-label_f1-score_positive_epoch_81.pth',
                'init_cfg': None,
            },
            'Impacted': {
                **base_model,
                'pretrained': 'work_dirs/opg_crops_fold_diagnosis_0_Impacted_0_swin/epoch_100.pth',
                'init_cfg': None,
            },
            'Periapical Lesion': {
                **base_model,
                'pretrained': 'work_dirs/opg_crops_fold_diagnosis_0_Periapical Lesion_0_swin/best_single-label_f1-score_positive_epoch_83.pth',
                'init_cfg': None,
            },
        }
    ),
    neck=None,
    head=dict(
        _delete_=True,
        type='ClsMultiTaskHead',
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='multi_label', 
            use_sigmoid=True,
        ),
        use_multilabel=True,
    ),
)
