_base_ = [
    '../../../configs/convnext_v2/convnext-v2-base_32xb32_in1k.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        _delete_=True,
        type='MultilabelMLP',
        checkpoints=[
            'work_dirs/opg_crops_fold_diagnosis_0_Caries/best_single-label_f1-score_epoch_18.pth',
            'work_dirs/opg_crops_fold_diagnosis_0_Deep Caries/best_single-label_f1-score_epoch_54.pth',
            'work_dirs/opg_crops_fold_diagnosis_0_Periapical Lesion/best_single-label_f1-score_epoch_31.pth',
            'work_dirs/opg_crops_fold_diagnosis_0_Impacted/best_single-label_f1-score_epoch_30_965.pth',
        ],
        model_cfg=_base_.model,
    ),
    head=dict(
        _delete_=True,
        type='ClsHead',
        loss=dict(type='AsymmetricLoss'),
    ),
)
