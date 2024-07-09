# optimizer
optimizer = dict(type='SGD', lr=0.00625, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(type='LinearLR', by_epoch=False, begin=0, end=2750),
    dict(
        type='MultiStepLR',
        milestones=[68750, 104500],
        gamma=0.1,
        begin=2750,
        end=138900,
        by_epoch=False)
]
# training schedule for culane
train_cfg = dict(type='IterBasedTrainLoop', max_iters=138900, val_interval=13890)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=27780),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
