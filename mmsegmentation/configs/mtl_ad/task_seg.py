_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py',
]

# Inference / test config for UF-Net drivable-area segmentation.
# Override Cityscapes paths in ../_base_/datasets/cityscapes.py if needed.

crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
load_from = None

model = dict(
    data_preprocessor=data_preprocessor,
    init_cfg=None,
    backbone=dict(
        type='ResNet',
        depth=18,
        conv_cfg=dict(type='Conv2d_tip'),
        style='pytorch'),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128),
    auxiliary_head=dict(in_channels=256, channels=64))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    visualization=dict(type='SegVisualizationHook', draw=False, interval=1))
