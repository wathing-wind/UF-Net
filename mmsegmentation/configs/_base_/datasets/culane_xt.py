train_dataset_type = 'Culane_dataset'
test_dataset_type = 'Culane_test'
data_root = 'data/CULane/'
crop_size = (512, 1024)

train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=train_dataset_type,
        split='train',
        batch_size=32,
        num_threads=4,
        data_root='/home/zzl/shujuji/CULane/',
        list_path='/home/zzl/shujuji/CULane/list/train_gt.txt',
        num_row=72,
        num_col=81,
        train_width=1600,
        train_height=320,
        num_cell_row=200,
        num_cell_col=100,
        crop_ratio=0.6,
        ))

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    # sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(
        type=test_dataset_type,
        data_root='/home/zzl/shujuji/CULane',
        list_path='/home/zzl/shujuji/CULane/list/test.txt',
        train_height=320, 
        train_width=1600, 
        crop_ratio=0.6
        ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='culaneF', 
    dataset='CULane',
    tta=True,
    data_root='/home/zzl/shujuji/CULane',
    num_row=72, 
    num_col=81)
test_evaluator = val_evaluator