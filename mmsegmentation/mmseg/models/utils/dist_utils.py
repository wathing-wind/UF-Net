import torch
import torch.distributed as dist
import tqdm

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def can_log():
    return is_main_process()


def dist_print(*args, **kwargs):
    if can_log():
        print(*args, **kwargs)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
    
def dist_tqdm(obj, *args, **kwargs):
    if can_log():
        return tqdm.tqdm(obj, *args, **kwargs)
    else:
        return obj