_base_ = [
    '../yolov5s/yolov5_s-v61_syncbn_8xb16-300e_coco.py'
]

# Inference / test config for UF-Net traffic object detection.
# Override data paths below to your local BDD100K layout before running test.py.

widen_factor = 1.0
channels = [128, 256, 512]  # ResNet-18
data_root = 'data/bdd100k'
load_from = None

train_batch_size_per_gpu = 32
train_num_workers = 8

anchors = [
    [(6, 7), (10, 8), (9, 19)],
    [(17, 15), (28, 23), (18, 44)],
    [(49, 34), (99, 63), (154, 156)]
]

class_name = ('car',)
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])
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
        conv_cfg=dict(type='Conv2d_tip'),
        norm_cfg=norm_cfg,
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
        loss_cls=dict(
            loss_weight=0.5 *
            (num_classes / 80 * 3 / _base_.num_det_layers))))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/det_val_coco.json',
        data_prefix=dict(img='images/100k/val/')))

test_dataloader = val_dataloader
val_evaluator = dict(ann_file='data/bdd100k/annotations/det_val_coco.json')
test_evaluator = val_evaluator

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    visualization=dict(type='mmdet.DetVisualizationHook', draw=True, interval=1))
