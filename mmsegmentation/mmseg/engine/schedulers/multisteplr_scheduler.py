# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmengine.optim.scheduler import MultiStepLR

from mmseg.registry import PARAM_SCHEDULERS

@PARAM_SCHEDULERS.register_module()
class WarmupMultiStepLR(MultiStepLR):

    def __init__(self, warmup: Optional[str] = None, warmup_iters: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.warmup = warmup
        self.warmup_iters = warmup_iters

    # 相比库函数调用增加了warmup机制
    def _get_value(self):
        """Compute value using chainable form of the scheduler."""
        
        if self.last_step not in self.milestones:   # 这里的milestone是乘过epoch_lenth的
            if self.warmup == 'linear' and self.last_step < self.warmup_iters:
                rate = self.last_step / self.warmup_iters
                return [
                    group[self.param_name] * rate for group in self.optimizer.param_groups
                ]
            else: return [
                    group[self.param_name] for group in self.optimizer.param_groups
                ]
        return [
            group[self.param_name] *
            self.gamma**self.milestones[self.last_step]
            for group in self.optimizer.param_groups
        ]