_base_ = './convnext-v2_binary.py'

test_dataloader = dict(dataset=dict(
    ann_file='/output/pred.json',
    pred_file='/output/pred.json',
    data_prefix='/output',
    data_root='/output',
))
