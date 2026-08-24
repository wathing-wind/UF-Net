# CULane dataset settings (inference / evaluation)
test_dataset_type = 'Culane_test'
data_root = 'data/CULane/'

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type=test_dataset_type,
        data_root=data_root,
        list_path='data/CULane/list/test.txt',
        train_height=320,
        train_width=1600,
        crop_ratio=0.6))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='culaneF',
    dataset='CULane',
    tta=True,
    data_root=data_root,
    num_row=72,
    num_col=81)
test_evaluator = val_evaluator
