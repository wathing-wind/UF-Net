import torch
from typing import Any, Dict
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch

@MODELS.register_module()
class CulaneDataPreProcessor(BaseDataPreprocessor):
    def __init__(self):
        super().__init__()

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:

        if training:
            return data
        else:
            # data_list = []
            # data_list.append(data)
            # return data_list
            inputs = data.get("inputs")
            inputs = torch.stack(inputs).cuda()
            data.update(inputs=inputs)
            return data
