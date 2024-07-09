# model settings 
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    _delete_=True,
    type='CulaneDataPreProcessor',
    )
model = dict(
    type='Culane_detector',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),    # 每个阶段产生的输出特征图的索引
        dilations=(1, 2, 2, 2),   
        strides=(1, 2, 2, 2),   
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch'),
    decode_head=dict(
        type='SegHead',
        backbone_depth= '18',
        num_lanes = 4,
        num_row=72,
        num_col=81,
        train_width = 1600,
        train_height=320,
        num_cell_row=20,
        num_cell_col=100,        
        fc_norm=True,
        tta=False,
        loss_decode=dict(
            type='culaneLoss',
            use_aux=False, 
            sim_loss_w=0.0,
            shp_loss_w=0.0,
            mean_loss_w=0.05)),)
    # model training and testing settings
    # train_cfg=dict(),
    # test_cfg=dict('whole'))



