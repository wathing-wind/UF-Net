_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
# norm_cfg = dict(type='BN', requires_grad=True)  单卡

load_from = None

model = dict(
    data_preprocessor=data_preprocessor,
    init_cfg=dict(type='Pretrained',
                  checkpoint='/home/xietao/.cache/torch/hub/checkpoints/resnet18_v1c-b5776b93.pth'),
    backbone=dict(
        type='ResNet',
        depth=18,
        style='pytorch'
    ),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))

# optimizer
optimizer=dict(type='AdamW', lr=0.008, weight_decay=0.001, betas=(0.95, 0.99))
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, begin=0, end=2750),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        begin=2751,
        end=120000,
        by_epoch=False)
]

# training schedule for culane
train_cfg = dict(type='IterBasedTrainLoop', max_iters=120000, val_interval=200000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000, max_keep_ckpts=5),
    visualization=dict(type='SegVisualizationHook')
    )
