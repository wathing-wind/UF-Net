_base_ = [
    '../_base_/datasets/culane_xt.py',
    '../_base_/default_runtime.py',
]

# Inference / test config for UF-Net lane detection (CULane).
# Override CULane paths in ../_base_/datasets/culane_xt.py if needed.

load_from = None
norm_cfg = dict(type='BN2d', requires_grad=True)
data_preprocessor = dict(type='CulaneDataPreProcessor')

model = dict(
    type='Culane_detector',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        strides=(1, 2, 2, 2),
        conv_cfg=dict(type='Conv2d_tip'),
        norm_cfg=norm_cfg,
        style='pytorch'),
    decode_head=dict(
        type='SegHead',
        backbone_depth='18',
        num_lanes=4,
        num_row=72,
        num_col=81,
        train_width=1600,
        train_height=320,
        num_cell_row=200,
        num_cell_col=100,
        fc_norm=True,
        tta=True,
        loss_decode=dict(
            type='culaneLoss',
            use_aux=False,
            sim_loss_w=0.0,
            shp_loss_w=0.0,
            mean_loss_w=0.05)))

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False))
