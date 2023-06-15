_base_ = './convnext-v2_binary.py'


data_root = '/input/'
export = 'fdi-checkedv2'
fold = '_diagnosis_4'
data_prefix = data_root + 'images'
ann_prefix = data_root + f'releases/{export}/other_formats/coco/'

batch_size = 32
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_root='/input',
        data_prefix='/input/images',
        ann_file=ann_prefix + f'train{fold}.json',
        pred_file='/input/full_pred.json',
    ),
)

val_dataloader = dict(
    dataset=dict(
        data_root='/input',
        data_prefix='/input/images',
        ann_file=ann_prefix + f'val{fold}.json',
        pred_file='/input/full_pred.json',
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root='/input',
        data_prefix='/input/images',
        ann_file=ann_prefix + f'val{fold}.json',
        pred_file='/input/full_pred.json',
    )
)

param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=5 * (batch_size // 32),
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=5 * (batch_size // 32))
]

if _base_.model.backbone.type == 'ConvNeXt' and _base_.model.backbone.arch == 'base':
    load_from = 'checkpoints/convnext-base.pth'
    # load_from = 'checkpoints/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth'
elif _base_.model.backbone.type == 'ConvNeXt' and _base_.model.backbone.arch == 'tiny':
    load_from = 'checkpoints/convnext-tiny.pth'
elif _base_.model.backbone.type == 'EfficientNetV2':
    load_from = 'checkpoints/efficientnet-s.pth'
else:
    load_from = 'checkpoints/swin-t.pth'


work_dir = '/output/' + _base_.work_dir + '_eff'
