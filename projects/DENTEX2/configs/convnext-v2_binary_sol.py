_base_ = './convnext-v2_binary.py'


data_root = '/input'
export = 'fdi-checkedv2'
fold = '_diagnosis_4'
data_prefix = data_root + 'images'
ann_prefix = data_root + f'releases/{export}/other_formats/coco/'

train_dataloader = dict(
    batch_size=32,
    dataset=dict(dataset=dict(
        data_root='/input',
        data_prefix='/input/images',
        ann_file=ann_prefix + f'train{fold}.json',
        pred_file='/input/full_pred.json',
    ))
)