_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_2xb2_80k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
# norm_cfg = dict(type='BN', requires_grad=True)  单卡

load_from = '/home/zzl/Downloads/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
    ),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))