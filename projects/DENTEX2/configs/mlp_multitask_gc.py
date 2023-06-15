_base_ = './mlp_multitask.py'

test_dataloader = dict(dataset=dict(
    ann_file='/output/pred.json',
    pred_file='/output/pred.json',
    data_prefix='/output',
    data_root='/output',
))

_base_.model.backbone.classifiers['Caries'].pop('pretrained')
_base_.model.backbone.classifiers['Deep Caries'].pop('pretrained')
_base_.model.backbone.classifiers['Periapical Lesion'].pop('pretrained')
_base_.model.backbone.classifiers['Impacted'].pop('pretrained')
