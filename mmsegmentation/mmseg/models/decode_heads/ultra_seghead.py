import torch
import torch.nn as nn
from typing import List, Tuple

from torch import Tensor

from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.utils import ConfigType
from mmseg.models.losses import (SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis,
                                 MeanLoss)

from ..builder import build_loss

@MODELS.register_module()
class SegHead(torch.nn.Module):
    def __init__(self, backbone_depth, num_lanes, train_width, train_height,
        num_cell_row, num_row, num_cell_col, num_col, fc_norm=False, tta=False,
        loss_decode=dict(
            type='culaneLoss',
            use_sigmoid=False,
            use_aux=False,
            sim_loss_w=0.0,
            shp_loss_w=0.0,
            mean_loss_w=0.05)):
        super(SegHead, self).__init__()

        self.num_lanes_row_col = num_lanes * 2
        self.backbone_depth = backbone_depth

        input_height = train_height
        input_width = train_width
        self.num_grid_row = num_cell_row                      # Nrdim行锚上的分类维度
        self.num_cls_row = num_row                            # Nrow行锚点数量
        self.num_grid_col = num_cell_col                      # Ncdim列锚上的分类维度
        self.num_cls_col = num_col                            # Ncol列锚点数量   
        self.num_lane_on_row = num_lanes                      # Nrlane列锚点上的车道数
        self.num_lane_on_col = num_lanes                      # Nclane列锚点上的车道数
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row # Pr行锚点的定位分支
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col # Pc列锚点的定位分支
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row                 # Er行锚点的存在分支
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col                 # Ec列锚点的存在分支
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4   # P+E的预测值
        mlp_mid_dim = 2048
        self.input_dim = input_height // 32 * input_width // 32 * 8
        self.tta = tta

        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )

        self.pool = torch.nn.Conv2d(512,8,1) if self.backbone_depth in ['34','18', '34fca'] else torch.nn.Conv2d(2048,8,1)

        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
            self.use_aux = loss_decode.use_aux
            self.sim_loss_w = loss_decode.sim_loss_w
            self.shp_loss_w = loss_decode.shp_loss_w
            self.mean_loss_w = loss_decode.mean_loss_w
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
        
        if self.use_aux:    
            self.aux_header2 = torch.nn.Sequential(
                ConvModule(128, 128, kernel_size=3, stride=1, padding=1) if self.backbone_depth in ['34','18'] else ConvModule(512, 128, kernel_size=3, stride=1, padding=1),
                ConvModule(128,128,3,padding=1),
                ConvModule(128,128,3,padding=1),
                ConvModule(128,128,3,padding=1),
            )
            self.aux_header3 = torch.nn.Sequential(
                ConvModule(256, 128, kernel_size=3, stride=1, padding=1) if self.backbone_depth in ['34','18'] else ConvModule(1024, 128, kernel_size=3, stride=1, padding=1),
                ConvModule(128,128,3,padding=1),
                ConvModule(128,128,3,padding=1),
            )
            self.aux_header4 = torch.nn.Sequential(
                ConvModule(512, 128, kernel_size=3, stride=1, padding=1) if self.backbone_depth in ['34','18'] else ConvModule(2048, 128, kernel_size=3, stride=1, padding=1),
                ConvModule(128,128,3,padding=1),
            )
            self.aux_combine = torch.nn.Sequential(
                ConvModule(384, 256, 3,padding=2,dilation=2),
                ConvModule(256, 128, 3,padding=2,dilation=2),
                ConvModule(128, 128, 3,padding=2,dilation=2),
                ConvModule(128, 128, 3,padding=4,dilation=4),
                torch.nn.Conv2d(128, self.num_lanes_row_col+1, 1)
                # output : n, num_of_lanes+1, h, w
            )

            initialize_weights(self.aux_header2,self.aux_header3,self.aux_header4,self.aux_combine)
            # self.droput = torch.nn.Dropout(0.1)

        initialize_weights(self.cls)
  
    def _loss(self, pred_dict: Tuple[Tensor],  data_label:List[Tensor], 
             train_cfg: ConfigType) -> dict:

        loss_dict = self.get_loss_dict('CULane', self.sim_loss_w, self.shp_loss_w, self.mean_loss_w, self.use_aux)  # 得到用于计算损失的字典
        results = self.inference_culane(pred_dict, data_label)

        losses = self.calc_loss(loss_dict, results)
        return losses

    def get_loss_dict(self, dataset_name, sim_loss_w, shp_loss_w, mean_loss_w, use_aux):

        if dataset_name in ['Tusimple', 'CULane']:
            loss_dict = {
                'name': ['cls_loss', 'relation_loss', 'relation_dis','cls_loss_col','cls_ext','cls_ext_col', 'mean_loss_row', 'mean_loss_col'],
                'op': [SoftmaxFocalLoss(2, ignore_lb=-1), ParsingRelationLoss(), ParsingRelationDis(), SoftmaxFocalLoss(2, ignore_lb=-1), torch.nn.CrossEntropyLoss(),  torch.nn.CrossEntropyLoss(), MeanLoss(), MeanLoss(),],
                'weight': [1.0, sim_loss_w, shp_loss_w, 1.0, 1.0, 1.0, mean_loss_w, mean_loss_w,],
                'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('cls_out',), ('cls_out_col', 'cls_label_col'), 
                ('cls_out_ext', 'cls_out_ext_label'), ('cls_out_col_ext', 'cls_out_col_ext_label') , ('cls_out', 'cls_label'),('cls_out_col', 'cls_label_col'),
                ],
            }
        else:
            raise NotImplementedError
        
        if use_aux:
            loss_dict['name'].append('seg_loss')
            loss_dict['op'].append(torch.nn.CrossEntropyLoss(weight = torch.tensor([0.6, 1., 1., 1., 1.])).cuda())
            loss_dict['weight'].append(1.0)
            loss_dict['data_src'].append(('seg_out', 'seg_label'))

        assert len(loss_dict['name']) == len(loss_dict['op']) == len(loss_dict['data_src']) == len(loss_dict['weight'])
        return loss_dict

    def inference_culane(self, pred, data_label):

        cls_out_ext_label = (data_label['labels_row'] != -1).long()
        cls_out_col_ext_label = (data_label['labels_col'] != -1).long()
        res_dict = {'cls_out': pred['loc_row'], 'cls_label': data_label['labels_row'], 'cls_out_col':pred['loc_col'],'cls_label_col':data_label['labels_col'],
                'cls_out_ext':pred['exist_row'], 'cls_out_ext_label':cls_out_ext_label, 'cls_out_col_ext':pred['exist_col'],
                    'cls_out_col_ext_label':cls_out_col_ext_label, 'labels_row_float':data_label['labels_row_float'], 'labels_col_float':data_label['labels_col_float'] }
        if 'seg_out' in pred.keys():
            res_dict['seg_out'] = pred['seg_out']
            res_dict['seg_label'] = data_label['seg_images']

        return res_dict
            
    def calc_loss(self, loss_dict, results):
        
        loss = dict()
        loss_value = 0
        # import pdb; pdb.set_trace()

        for i in range(len(loss_dict['name'])):

            # loss['loss_name'] = loss_dict['name'][i]

            if loss_dict['weight'][i] == 0:
                continue
                
            data_src = loss_dict['data_src'][i]

            datas = [results[src] for src in data_src]

            loss_cur = loss_dict['op'][i](*datas)

            # loss['loss_value'] = loss_cur * loss_dict['weight'][i]
            loss_value += loss_cur * loss_dict['weight'][i]

            loss['loss_value'] = loss_value

        return loss
        
    def _forword_head(self, outs):
        x2, x3, fea = outs[1], outs[2], outs[3]

        x2 = self.aux_header2(x2)
        x3 = self.aux_header3(x3)
        x3 = torch.nn.functional.interpolate(x3,scale_factor = 2,mode='bilinear')
        x4 = self.aux_header4(fea)
        x4 = torch.nn.functional.interpolate(x4,scale_factor = 4,mode='bilinear')
        aux_seg = torch.cat([x2,x3,x4],dim=1)
        aux_seg = self.aux_combine(aux_seg)

        return aux_seg
        
    def forward(self, outs, mode):

        if mode == 'loss':
            if self.use_aux:
                seg_out = self._forword_head(outs)
            fea = outs[3]
            fea = self.pool(fea)
            # print(fea.shape)
            # print(self.coord.shape)
            # fea = torch.cat([fea, self.coord.repeat(fea.shape[0],1,1,1)], dim = 1)
            
            fea = fea.view(-1, self.input_dim)
            out = self.cls(fea)

            pred_dict = {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
                    'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                    'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                    'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}
            if self.use_aux:
                pred_dict['seg_out'] = seg_out
        
        elif mode == 'predict':
            pred_dict = self.forward_tta(outs)
        
        return pred_dict

    def forward_tta(self, outs):

        x2, x3, fea = outs[1], outs[2], outs[3]
        pooled_fea = self.pool(fea)
        n,c,h,w = pooled_fea.shape

        left_pooled_fea = torch.zeros_like(pooled_fea)
        right_pooled_fea = torch.zeros_like(pooled_fea)
        up_pooled_fea = torch.zeros_like(pooled_fea)
        down_pooled_fea = torch.zeros_like(pooled_fea)

        left_pooled_fea[:,:,:,:w-1] = pooled_fea[:,:,:,1:]
        left_pooled_fea[:,:,:,-1] = pooled_fea.mean(-1)
        
        right_pooled_fea[:,:,:,1:] = pooled_fea[:,:,:,:w-1]
        right_pooled_fea[:,:,:,0] = pooled_fea.mean(-1)

        up_pooled_fea[:,:,:h-1,:] = pooled_fea[:,:,1:,:]
        up_pooled_fea[:,:,-1,:] = pooled_fea.mean(-2)

        down_pooled_fea[:,:,1:,:] = pooled_fea[:,:,:h-1,:]
        down_pooled_fea[:,:,0,:] = pooled_fea.mean(-2)
        # 10 x 25
        fea = torch.cat([pooled_fea, left_pooled_fea, right_pooled_fea, up_pooled_fea, down_pooled_fea], dim = 0)
        fea = fea.view(-1, self.input_dim)

        out = self.cls(fea)

        predict = {'loc_row': out[:,:self.dim1].view(-1,self.num_grid_row, self.num_cls_row, self.num_lane_on_row), 
                'loc_col': out[:,self.dim1:self.dim1+self.dim2].view(-1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col),
                'exist_row': out[:,self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, self.num_cls_row, self.num_lane_on_row), 
                'exist_col': out[:,-self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}

        return predict

    
def initialize_weights(*models):
    for model in models:
        real_init_weights(model)
def real_init_weights(m):

    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)
