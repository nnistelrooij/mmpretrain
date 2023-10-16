_base_ = './swins_multitask_mha.py'

test_dataloader = dict(
    num_workers=1,
    persistent_workers=False,
    dataset=dict(
        type='ToothCropPNGsDataset',
        data_prefix='/input',
        data_root='/input',
    ),
)
