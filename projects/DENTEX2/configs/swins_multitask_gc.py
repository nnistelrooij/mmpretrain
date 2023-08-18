_base_ = './swins_multitask.py'

test_dataloader = dict(
    num_workers=1,
    persistent_workers=False,
    batch_size=64,
    dataset=dict(
        _delete_=True,
        type='ToothCropGCDataset',
        ann_file='/output/pred.json',
        data_prefix='/output',
        data_root='/output',
        metainfo=dict(classes=_base_.classes, attributes=_base_.attributes),
        serialize_data=False,
        pipeline=_base_.test_pipeline[1:],
    ),
)
tta_pipeline = _base_.tta_pipeline[1:]
test_evaluator = []

_base_.model.backbone.classifiers['Caries'].pop('pretrained')
_base_.model.backbone.classifiers['Deep Caries'].pop('pretrained')
_base_.model.backbone.classifiers['Periapical Lesion'].pop('pretrained')
_base_.model.backbone.classifiers['Impacted'].pop('pretrained')
_base_.model.backbone.pop('pretrained')

log_level = 'WARN'
