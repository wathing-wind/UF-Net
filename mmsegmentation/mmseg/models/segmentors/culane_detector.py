# Copyright (c) OpenMMLab. All rights reserved.
import logging
import torch
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor


@MODELS.register_module()
class Culane_detector(BaseSegmentor):
    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)  
        # self._init_auxiliary_head(auxiliary_head) 

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    # 得到decode_head的参数
    def _init_decode_head(self, decode_head: ConfigType) -> None:  
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        pass

    def forward(self,
                inputs,
                mode,
                name=None,
                seg_images=None,
                labels_row=None,
                labels_col=None,
                labels_row_float=None,
                labels_col_float=None,
                ):        

        if mode=='loss':

            outs = self.backbone(inputs)
            seg_logits = self.decode_head(outs, mode)
            
            data_label = {'inputs':inputs, 'seg_images':seg_images, 
                          'labels_row':labels_row, 'labels_col':labels_col, 
                          'labels_row_float':labels_row_float, 'labels_col_float':labels_col_float}

            losses = dict()

            loss_decode = self.decode_head._loss(seg_logits, data_label, self.train_cfg)

            losses.update(loss_decode)

            return losses
            
        elif mode=='predict':
            # inputs = torch.stack(inputs).cuda()

            outs = self.backbone(inputs)   # 这个inputs就是test_dataloader在getitem里面返回的数据
            predict = self.decode_head(outs, mode)

            predict_list = []
            predict_list.append(predict)
            return predict_list

    def _forward(self):
        pass

    def encode_decode(self):
        pass

    def extract_feat(self):
        pass

    def loss(self):
        pass

    def predict(self):
        pass
