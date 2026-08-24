# Copyright (c) OpenMMLab. All rights reserved.
from .poly_ratio_scheduler import PolyLRRatio
from .multisteplr_scheduler import WarmupMultiStepLR

__all__ = ['PolyLRRatio', 'WarmupMultiStepLR']
