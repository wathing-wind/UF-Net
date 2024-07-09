_base_ = [
    '../yolov5s/yolov5_s-v61_syncbn_8xb16-300e_coco.py'
]

max_epochs = 100  # 训练的最大 epoch
deepen_factor = _base_.deepen_factor
widen_factor = 1.0
# channels = [512, 1024, 2048]   # resnet-50
channels = [128, 256, 512]     # resnet-18
data_root = '/home/zzl/shujuji/BDD100K/bdd100k'  # 数据集目录的绝对路径
# data_root = '/root/workspace/mmyolo/data/cat/'  # Docker 容器里面数据集目录的绝对路径

# 结果保存的路径，可以省略，省略保存的文件名位于 work_dirs 下 config 同名的文件夹中
# 如果某个 config 只是修改了部分参数，修改这个变量就可以将新的训练文件保存到其他地方
work_dir = './work_dirs/yolov5_s-v61_syncbn_fast_1xb32-100e_bdd'

# load_from 可以指定本地路径或者 URL，设置了 URL 会自动进行下载，因为上面已经下载过，我们这里设置本地路径
# 因为本教程是在 cat 数据集上微调，故这里需要使用 `load_from` 来加载 MMYOLO 中的预训练模型，这样可以在加快收敛速度的同时保证精度
# load_from = '/ai/volume/mmyolo/data/bdd100k/pth/epoch_100.pth'  # noqa
load_from = None  # noqa

# 根据自己的 GPU 情况，修改 batch size，YOLOv5-s 默认为 8卡 x 16bs
train_batch_size_per_gpu = 32
# train_num_workers = 4  # 推荐使用 train_num_workers = nGPU x 4   单卡
# train_num_workers = 16  # 四卡
train_num_workers = 8  # 两卡

anchors = [  # 此处已经根据数据集特点更新了 anchor，关于 anchor 的生成，后面小节会讲解
    [(6, 7), (10, 8), (9, 19)],  # P3/8
    [(17, 15), (28, 23), (18, 44)],  # P4/16
    [(49, 34), (99, 63), (154, 156)]  # P5/32
]

# class_name = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign', )  # 根据 class_with_id.txt 类别信息，设置 class_name
# 测试只有4类会如何，依然可以训练
class_name = ('car', )  

num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60)]  # 画图时候的颜色，随便设置即可
)

# norm_cfg=dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config
norm_cfg = dict(type='BN2d', requires_grad=True)

model = dict(
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(1, 2, 3), 
        norm_cfg=norm_cfg,
        # norm_eval=True,
        style='pytorch',
        ),    
    neck=dict(
        type='YOLOv5PAFPN',
        widen_factor=widen_factor,
        in_channels=channels, 
        out_channels=channels,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=channels,
            widen_factor=widen_factor,            
            num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors),

        # loss_cls 会根据 num_classes 动态调整，但是 num_classes = 1 的时候，loss_cls 恒为 0
        loss_cls=dict(loss_weight=0.5 *
                      (num_classes / 80 * 3 / _base_.num_det_layers))))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    collate_fn=dict(type='yolov5_collate'), 
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        # 数据量太少的话，可以使用 RepeatDataset ，在每个 epoch 内重复当前数据集 n 次，这里设置 5 是重复 5 次
        times=1,
        dataset=dict(
            # type=_base_.dataset_type,
            type='YOLOv5BddDataset',
            data_root=data_root,
            metainfo=metainfo,
            ann_file='/home/zzl/shujuji/BDD100K/COCO/det_train_coco.json',
            data_prefix=dict(img='/home/zzl/shujuji/BDD100K/bdd100k/images/100k/train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            # filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='/home/zzl/shujuji/BDD100K/COCO/det_val_coco.json',
        data_prefix=dict(img='/home/zzl/shujuji/BDD100K/bdd100k/images/100k/val/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file='/home/zzl/shujuji/BDD100K/COCO/det_val_coco.json')
test_evaluator = val_evaluator

# optimizer
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
    
    )