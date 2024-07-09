task_slice_count = 3
task_groups = {'seg': [0], 'culane': [1], 'det':[2]}

task_list = [
    {
        'task_name': 'seg',
        'task_config': './task_seg.py',
        'task_group': 'seg',
        'task_prefix_group': {
            'bbox_head.conv_pred': 'seg',
        }
    },
    {
        'task_name': 'culane',
        'task_config': './task_culane.py',
        'task_group': 'culane',
        'task_prefix_group': {
            'bbox_head.conv_pred': 'culane',
        }
    },
    {
        'task_name': 'det',
        'task_config': './task_det.py',
        'task_group': 'det',
        'task_prefix_group': {
            'bbox_head.conv_pred': 'det',
        }
    }
]