_base_ = './swins_multitask.py'

test_dataloader = dict(
    num_workers=1,
    persistent_workers=False,
    batch_size=128,
    dataset=dict(
        ann_file='/output/pred.json',
        pred_file='/output/pred.json',
        data_prefix='/output',
        data_root='/output',
    ),
)

_base_.model.backbone.classifiers['Caries'].pop('pretrained')
_base_.model.backbone.classifiers['Deep Caries'].pop('pretrained')
_base_.model.backbone.classifiers['Periapical Lesion'].pop('pretrained')
_base_.model.backbone.classifiers['Impacted'].pop('pretrained')
_base_.model.backbone.pop('pretrained')

test_evaluator = []
