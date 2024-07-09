# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os
import os.path as osp
import pickle
import platform
import time
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.dist import (broadcast, get_dist_info, get_rank, init_dist,
                           is_distributed, master_only)
from mmengine.evaluator import Evaluator
from mmengine.fileio import FileClient, join_path
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, MMLogger, print_log
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
                                     set_multi_processing)

from mmengine.runner import Runner

@RUNNERS.register_module()
class Runner_culane(Runner):

    @staticmethod
    def build_dataloader(dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None,
                         diff_rank_seed: bool = False) -> DataLoader:
        if dataloader.dataset.type == 'Culane_dataset':
            if isinstance(dataloader, DataLoader):
                return dataloader

            dataloader_cfg = copy.deepcopy(dataloader)

            # build dataset
            dataset_cfg = dataloader_cfg.pop('dataset')
            if isinstance(dataset_cfg, dict):
                dataset = DATASETS.build(dataset_cfg)
                if hasattr(dataset, 'full_init'):
                    dataset.full_init()
            else:
                # fallback to raise error in dataloader
                # if `dataset_cfg` is not a valid type
                dataset = dataset_cfg
            
            return dataset
        
        else:
            # super().__init__(dataloader, seed, diff_rank_seed)
            if isinstance(dataloader, DataLoader):
                return dataloader

            dataloader_cfg = copy.deepcopy(dataloader)

            # build dataset
            dataset_cfg = dataloader_cfg.pop('dataset')
            if isinstance(dataset_cfg, dict):
                dataset = DATASETS.build(dataset_cfg)
                if hasattr(dataset, 'full_init'):
                    dataset.full_init()
            else:
                # fallback to raise error in dataloader
                # if `dataset_cfg` is not a valid type
                dataset = dataset_cfg
            import pdb; pdb.set_trace()
    
            # build sampler
            sampler_cfg = dataloader_cfg.pop('sampler')
            if isinstance(sampler_cfg, dict):
                sampler_seed = None if diff_rank_seed else seed
                sampler = DATA_SAMPLERS.build(
                    sampler_cfg,
                    default_args=dict(dataset=dataset, seed=sampler_seed))
            else:
                # fallback to raise error in dataloader
                # if `sampler_cfg` is not a valid type
                sampler = sampler_cfg

            # build batch sampler
            batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
            if batch_sampler_cfg is None:
                batch_sampler = None
            elif isinstance(batch_sampler_cfg, dict):
                batch_sampler = DATA_SAMPLERS.build(
                    batch_sampler_cfg,
                    default_args=dict(
                        sampler=sampler,
                        batch_size=dataloader_cfg.pop('batch_size')))
            else:
                # fallback to raise error in dataloader
                # if `batch_sampler_cfg` is not a valid type
                batch_sampler = batch_sampler_cfg

            # build dataloader
            init_fn: Optional[partial]

            if 'worker_init_fn' in dataloader_cfg:
                worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
                worker_init_fn_type = worker_init_fn_cfg.pop('type')
                if isinstance(worker_init_fn_type, str):
                    worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
                elif callable(worker_init_fn_type):
                    worker_init_fn = worker_init_fn_type
                else:
                    raise TypeError(
                        'type of worker_init_fn should be string or callable '
                        f'object, but got {type(worker_init_fn_type)}')
                assert callable(worker_init_fn)
                init_fn = partial(worker_init_fn,
                                **worker_init_fn_cfg)  # type: ignore
            else:
                if seed is not None:
                    disable_subprocess_warning = dataloader_cfg.pop(
                        'disable_subprocess_warning', False)
                    assert isinstance(disable_subprocess_warning, bool), (
                        'disable_subprocess_warning should be a bool, but got '
                        f'{type(disable_subprocess_warning)}')
                    init_fn = partial(
                        default_worker_init_fn,
                        num_workers=dataloader_cfg.get('num_workers'),
                        rank=get_rank(),
                        seed=seed,
                        disable_subprocess_warning=disable_subprocess_warning)
                else:
                    init_fn = None

            # `persistent_workers` requires pytorch version >= 1.7
            if ('persistent_workers' in dataloader_cfg
                    and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
                print_log(
                    '`persistent_workers` is only available when '
                    'pytorch version >= 1.7',
                    logger='current',
                    level=logging.WARNING)
                dataloader_cfg.pop('persistent_workers')

            # The default behavior of `collat_fn` in dataloader is to
            # merge a list of samples to form a mini-batch of Tensor(s).
            # However, in mmengine, if `collate_fn` is not defined in
            # dataloader_cfg, `pseudo_collate` will only convert the list of
            # samples into a dict without stacking the batch tensor.
            collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                                dict(type='pseudo_collate'))
            if isinstance(collate_fn_cfg, dict):
                collate_fn_type = collate_fn_cfg.pop('type')
                if isinstance(collate_fn_type, str):
                    collate_fn = FUNCTIONS.get(collate_fn_type)
                else:
                    collate_fn = collate_fn_type
                collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
            elif callable(collate_fn_cfg):
                collate_fn = collate_fn_cfg
            else:
                raise TypeError(
                    'collate_fn should be a dict or callable object, but got '
                    f'{collate_fn_cfg}')
            data_loader = DataLoader(
                dataset=dataset,
                sampler=sampler if batch_sampler is None else None,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                worker_init_fn=init_fn,
                **dataloader_cfg)
            return data_loader    