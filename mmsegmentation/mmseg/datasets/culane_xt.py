# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import numpy as np
import random
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import json
from torch.utils.data import Dataset
from PIL import Image
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy
from mmseg.datasets.basesegdataset import BaseSegDataset
from typing import List
from mmseg.registry import DATASETS
import my_interp
import torchvision.transforms as transforms

culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
culane_col_anchor = [0.,  20.,  40.,  60.,  80., 100., 120., 140., 160., 180., 200.,
                    220., 240., 260., 280., 300., 320., 340., 360., 380., 400., 420.,
                    440., 460., 480., 500., 520., 540., 560., 580., 600., 620., 640.,
                    660., 680., 700., 720., 740., 760., 780., 800.]

class LaneExternalIterator(object):
    def __init__(self, path, list_path, batch_size=None, shard_id=None, num_shards=None, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode
        self.path = path
        self.list_path = list_path
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards

        if isinstance(list_path, str):
            with open(list_path, 'r') as f:
                total_list = f.readlines()
        elif isinstance(list_path, list) or isinstance(list_path, tuple):
            total_list = []
            for lst_path in list_path:
                with open(lst_path, 'r') as f:
                    total_list.extend(f.readlines())
        else:
            raise NotImplementedError

        if self.mode == 'train':
            cache_path = os.path.join(path, 'culane_anno_cache.json')
            if shard_id == 0:
                print('loading cached data')
            cache_fp = open(cache_path, 'r')
            self.cached_points = json.load(cache_fp)
            if shard_id == 0:
                print('cached data loaded')

        self.total_len = len(total_list)
        self.list = total_list
        self.n = len(self.list)

    def __iter__(self):
        self.i = 0
        if self.mode == 'train':
            random.shuffle(self.list)
        return self

    def _prepare_train_batch(self):
        images = []
        seg_images = []
        labels = []

        # image_names = []   # 添加用于查看每次加载的图片

        for _ in range(self.batch_size):
            l = self.list[self.i % self.n]
            l_info = l.split()
            img_name = l_info[0]
            seg_name = l_info[1]

            if img_name[0] == '/':
                img_name = img_name[1:]
            if seg_name[0] == '/':
                seg_name = seg_name[1:]
                
            img_name = img_name.strip()
            seg_name = seg_name.strip()

            # image_names.append(img_name)  # 添加用于查看每次加载的图片
            
            img_path = os.path.join(self.path, img_name)
            with open(img_path, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))

            img_path = os.path.join(self.path, seg_name)
            with open(img_path, 'rb') as f:
                seg_images.append(np.frombuffer(f.read(), dtype=np.uint8))

            points = np.array(self.cached_points[img_name])
            labels.append(points.astype(np.float32))

            self.i = self.i + 1

        # print(image_names)   # 添加用于查看每次加载的图片

        return (images, seg_images, labels)
 
    def _prepare_test_batch(self):
        images = []
        names = []
        for _ in range(self.batch_size):
            img_name = self.list[self.i % self.n].split()[0]

            if img_name[0] == '/':
                img_name = img_name[1:]
            img_name = img_name.strip()

            img_path = os.path.join(self.path, img_name)

            with open(img_path, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))
            names.append(np.array(list(map(ord,img_name))))
            self.i = self.i + 1
            
        return images, names

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration
        if self.mode == 'train':
            res = self._prepare_train_batch()
        elif self.mode == 'test':
            res = self._prepare_test_batch()
        else:
            raise NotImplementedError

        return res

    def __len__(self):
        return self.total_len

    next = __next__


def encoded_images_sizes(jpegs):
    shapes = fn.peek_image_shape(jpegs)  # the shapes are HWC
    h = fn.slice(shapes, 0, 1, axes=[0]) # extract height...
    w = fn.slice(shapes, 1, 1, axes=[0]) # ...and width...
    return fn.cat(w, h)               # ...and concatenate


def ExternalSourceTrainPipeline(batch_size, num_threads, device_id, external_data,
                                train_width, train_height, top_crop, 
                                normalize_image_scale=False, nscale_w=None, nscale_h=None):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, seg_images, labels = fn.external_source(source=external_data, num_outputs=3)
        images = fn.decoders.image(jpegs, device="mixed")
        seg_images = fn.decoders.image(seg_images, device="mixed")
        if normalize_image_scale:
            images = fn.resize(images, resize_x=nscale_w, resize_y=nscale_h)
            seg_images = fn.resize(seg_images, resize_x=nscale_w, resize_y=nscale_h, interp_type=types.INTERP_NN)
            # make all images at the same size

        size = encoded_images_sizes(jpegs)
        center = size / 2

        mt = fn.transforms.scale(scale = fn.random.uniform(range=(0.8, 1.2), shape=[2]), center = center)
        mt = fn.transforms.rotation(mt, angle = fn.random.uniform(range=(-6, 6)), center = center)

        off = fn.cat(fn.random.uniform(range=(-200, 200), shape = [1]), fn.random.uniform(range=(-100, 100), shape = [1]))
        mt = fn.transforms.translation(mt, offset = off)

        images = fn.warp_affine(images, matrix = mt, fill_value=0, inverse_map=False)
        seg_images = fn.warp_affine(seg_images, matrix = mt, fill_value=0, inverse_map=False)
        labels = fn.coord_transform(labels.gpu(), MT = mt)

        images = fn.resize(images, resize_x=train_width, resize_y=int(train_height/top_crop))
        seg_images = fn.resize(seg_images, resize_x=train_width, resize_y=int(train_height/top_crop), interp_type=types.INTERP_NN)

        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        seg_images = fn.crop_mirror_normalize(seg_images, 
                                            dtype=types.FLOAT, 
                                            mean = [0., 0., 0.],
                                            std = [1., 1., 1.],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        pipe.set_outputs(images, seg_images, labels)
    return pipe


@DATASETS.register_module()
class Culane_dataset(BaseSegDataset):
    def __init__(self, split, batch_size, num_threads, data_root, list_path,
                       num_row, num_col, train_width, train_height, 
                       num_cell_row, num_cell_col, crop_ratio):
    
        self.dataset = 'Culane'
        self._fully_initialized = True
        row_anchor = np.linspace(0.42,1, num_row)
        col_anchor = np.linspace(0,1, num_col)
        from mmengine.dist.utils import get_dist_info
        rank, world_size = get_dist_info()
        
        eii = LaneExternalIterator(data_root, list_path, batch_size=batch_size, 
                                   shard_id=rank, num_shards=world_size, mode=split)
        
        self.original_image_width = 1640
        self.original_image_height = 590

        pipe = ExternalSourceTrainPipeline(batch_size, num_threads, 1, eii, train_width, train_height, crop_ratio)
        self.pii = DALIGenericIterator(pipe, output_map = ['images', 'seg_images', 'points'], auto_reset=True,
                                       last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
        
        self.eii_n = eii.n
        self.batch_size = batch_size
        
        self.interp_loc_row = torch.tensor(row_anchor, dtype=torch.float32).cuda() * self.original_image_height
        self.interp_loc_col = torch.tensor(col_anchor, dtype=torch.float32).cuda() * self.original_image_width
        self.num_cell_row = num_cell_row
        self.num_cell_col = num_cell_col

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.pii)
        images = data[0]['images']
        seg_images = data[0]['seg_images']
        points = data[0]['points']
        points_row = my_interp.run(points, self.interp_loc_row, 0)
        
        points_row_extend = self._extend(points_row[:,:,:,0]).transpose(1,2)
        labels_row = (points_row_extend / self.original_image_width * (self.num_cell_row - 1)).long()
        labels_row[points_row_extend < 0] = -1
        labels_row[points_row_extend > self.original_image_width] = -1
        labels_row[labels_row < 0] = -1
        labels_row[labels_row > (self.num_cell_row - 1)] = -1

        points_col = my_interp.run(points, self.interp_loc_col, 1)
        
        points_col = points_col[:,:,:,1].transpose(1,2)
        labels_col = (points_col / self.original_image_height * (self.num_cell_col - 1)).long()
        labels_col[points_col < 0] = -1
        labels_col[points_col > self.original_image_height] = -1
        
        labels_col[labels_col < 0] = -1
        labels_col[labels_col > (self.num_cell_col - 1)] = -1

        labels_row_float = points_row_extend / self.original_image_width
        labels_row_float[labels_row_float<0] = -1
        labels_row_float[labels_row_float>1] = -1

        labels_col_float = points_col / self.original_image_height
        labels_col_float[labels_col_float<0] = -1
        labels_col_float[labels_col_float>1] = -1
        
        return {'inputs':images, 'seg_images':seg_images, 'labels_row':labels_row, 'labels_col':labels_col, 'labels_row_float':labels_row_float, 'labels_col_float':labels_col_float}
    
    def __len__(self):
        return int((self.eii_n + self.batch_size - 1) / self.batch_size)

    def reset(self):
        self.pii.reset()
    next = __next__

    def _extend(self, coords):
        # coords : n x num_lane x num_cls
        n, num_lanes, num_cls = coords.shape
        coords_np = coords.cpu().numpy()
        coords_axis = np.arange(num_cls)
        fitted_coords = coords.clone()
        for i in range(n):
            for j in range(num_lanes):
                lane = coords_np[i,j]
                if lane[-1] > 0:
                    continue

                valid = lane > 0
                num_valid_pts = np.sum(valid)
                if num_valid_pts < 6:
                    continue
                p = np.polyfit(coords_axis[valid][num_valid_pts//2:], lane[valid][num_valid_pts//2:], deg = 1)   
                start_point = coords_axis[valid][num_valid_pts//2]
                fitted_lane = np.polyval(p, np.arange(start_point, num_cls))
                
                fitted_coords[i,j,start_point:] = torch.tensor(fitted_lane, device = coords.device)
        return fitted_coords

    def _extend_col(self, coords):
        pass

def loader_func(path):
    return Image.open(path)
    
@DATASETS.register_module()
class Culane_test(Dataset):
    def __init__(self, data_root, list_path, lazy_init=True,
                 train_height=None, train_width=None, crop_ratio=None):

        self.path = data_root
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
        img = loader_func(img_path)
        
        if self.img_transform is not None:
            img = self.img_transform(img)
        img = img[:,-self.crop_size:,:]

        return {'inputs': img, 'name': name}
        # return result

    def __len__(self):
        return len(self.list)