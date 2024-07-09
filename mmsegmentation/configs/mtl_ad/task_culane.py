_base_ = [
    '../_base_/datasets/culane_xt.py', '../_base_/default_runtime.py',
]

load_from = None

norm_cfg = dict(type='BN2d', requires_grad=True)
data_preprocessor = dict(
    # _delete_=True,
    type='CulaneDataPreProcessor',
    )

model = dict(
    type='Culane_detector',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),    # 每个阶段产生的输出特征图的索引
        strides=(1, 2, 2, 2),
        conv_cfg=dict(type='Conv2d_tip'),
        norm_cfg=norm_cfg,
        # norm_eval=True,
        style='pytorch'),
    decode_head=dict(
        type='SegHead',
        backbone_depth= '18',
        num_lanes = 4,
        num_row=72,
        num_col=81,
        train_width = 1600,
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

# optimizer
# optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
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
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000, max_keep_ckpts=5)
    )
