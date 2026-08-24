import torch, os
import pdb
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from .transforms.mytransforms import find_start_pos
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
culane_col_anchor = [0.,  20.,  40.,  60.,  80., 100., 120., 140., 160., 180., 200.,
                    220., 240., 260., 280., 300., 320., 340., 360., 380., 400., 420.,
                    440., 460., 480., 500., 520., 540., 560., 580., 600., 620., 640.,
                    660., 680., 700., 720., 740., 760., 780., 800.]

@DATASETS.register_module()
class Culane_test(BaseSegDataset):
    def __init__(self, data_root, list_path, lazy_init=True,
                 train_height=None, train_width=None, crop_ratio=None):

        self.path = data_root
        self._fully_initialized = True
        self.img_transform = transforms.Compose([
            transforms.Resize((int(train_height / crop_ratio), train_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.crop_size = train_height
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane  对应txt下面从/后面第一个字母开始的所有成员组成的列表

    def __getitem__(self, idx: int) -> dict:

        name = self.list[idx].split()[0]
        img_path = os.path.join(self.path, name)
        img = self.loader_func(img_path)
        
        if self.img_transform is not None:
            img = self.img_transform(img)
        img = img[:,-self.crop_size:,:]
        import pdb; pdb.set_trace()

        return {'inputs': img, 'name': name}

    def __len__(self):
        return len(self.list)
    
    def loader_func(path):
        return Image.open(path)