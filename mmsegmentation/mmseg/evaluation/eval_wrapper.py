import numpy as np
import platform
import os, json, torch, scipy
import torchvision.transforms as transforms

from typing import Dict, Optional, Sequence
from scipy.optimize import leastsq
from mmengine.evaluator import BaseMetric
from mmengine.dist import (is_main_process, master_only, is_distributed)
from mmseg.utils import dist_print, dist_tqdm, synchronize
from mmseg.datasets.culane_xt import (LaneTestDataset, SeqDistributedSampler, culane_col_anchor, culane_row_anchor)
from mmseg.registry import METRICS


def generate_lines(out, out_ext, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):

    grid = torch.arange(out.shape[1]) + 0.5
    grid = grid.view(1,-1,1,1).cuda()
    loc = (out.softmax(1) * grid).sum(1) 
    
    loc = loc / (out.shape[1]-1) * 1640
    # n, num_cls, num_lanes
    valid = out_ext.argmax(1)
    # n, num_cls, num_lanes
    valid = valid.cpu()
    loc = loc.cpu()

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            for i in [1,2]:
                if valid[j,:,i].sum() > 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            fp.write('%.3f %.3f '% ( loc[j,k,i] , culane_row_anchor[k] * 590))
                    fp.write('\n')

def generate_lines_col(out_col,out_col_ext, shape, names, output_path, griding_num, localization_type='abs', flip_updown=False):
    
    grid = torch.arange(out_col.shape[1]) + 0.5
    grid = grid.view(1,-1,1,1).cuda()
    loc = (out_col.softmax(1) * grid).sum(1) 
    
    loc = loc / (out_col.shape[1]-1) * 590
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1)
    # n, num_cls, num_lanes
    valid = valid.cpu()
    loc = loc.cpu()

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            for i in [0,3]:
                if valid[j,:,i].sum() > 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            fp.write('%.3f %.3f '% ( culane_col_anchor[k] * 1640, loc[j,k,i] ))
                    fp.write('\n')

def generate_lines_local(out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    max_indices = out.argmax(1).cpu()
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu()

    if mode == 'normal' or mode == '2row2col':
        lane_list = [1, 2]
    else:
        lane_list = range(num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            
                            out_tmp = (out[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 

                            out_tmp = out_tmp / (out.shape[1]-1) * 1640
                            fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))

                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_col_local(out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        lane_list = [0, 3]
    else:
        lane_list = range(num_lane)

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 

                            out_tmp = out_tmp / (out_col.shape[1]-1) * 590
                            fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))

                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_reg(out, out_ext, names, output_path, mode='normal', row_anchor = None):
    batch_size, num_grid_row, num_cls, num_lane = out.shape
    # n , num_cls, num_lanes
    
    valid = out_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out = out.cpu().sigmoid()

    if mode == 'normal' or mode == '2row2col':
        lane_list = [1, 2]
    else:
        lane_list = range(num_lane)

    local_width = 1
    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 2:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            # all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            
                            out_tmp = out[j,0,k,i] * 1640

                            fp.write('%.3f %.3f '% ( out_tmp , row_anchor[k] * 590))
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def generate_lines_col_reg(out_col,out_col_ext, names, output_path, mode='normal', col_anchor = None):
    batch_size, num_grid_col, num_cls, num_lane = out_col.shape
    # max_indices = out_col.argmax(1).cpu()
    # n, num_cls, num_lanes
    valid = out_col_ext.argmax(1).cpu()
    # n, num_cls, num_lanes
    out_col = out_col.cpu().sigmoid()
    local_width = 1

    if mode == 'normal' or mode == '2row2col':
        lane_list = [0, 3]
    else:
        lane_list = range(num_lane)

    for j in range(valid.shape[0]):

        name = names[j]
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for i in range(num_lane):
            for i in lane_list:
                if valid[j,:,i].sum() > num_cls / 4:
                    for k in range(valid.shape[1]):
                        if valid[j,k,i]:
                            # all_ind = torch.tensor(list(range(max(0,max_indices[j,k,i] - local_width), min(out_col.shape[1]-1, max_indices[j,k,i] + local_width) + 1)))
                            # out_tmp = (out_col[j,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_col[j,0,k,i] * 590
                            fp.write('%.3f %.3f '% ( col_anchor[k] * 1640, out_tmp ))
                    fp.write('\n')
                elif mode == 'all':
                    fp.write('\n')

def coordinate_parse(line):
    if line == '\n':
        return [], []

    items = line.split(' ')[:-1]
    x = [float(items[2*i]) for i in range(len(items)//2)]
    y = [float(items[2*i+1]) for i in range(len(items)//2)]

    return x, y

def func(p, x):
    f = np.poly1d(p)
    return f(x)

def resudual(p, x, y):
    error = y - func(p, x)
    return error

def revise_lines(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        if not os.path.exists(line_save_path):
            continue
        with open(line_save_path, 'r') as fp:
            lines = fp.readlines()
        flag = True
        for i in range(4):
            x1, y1 = coordinate_parse(lines[i])
            x2, y2 = coordinate_parse(lines[i+4])
            x = x1 + x2
            y = y1 + y2
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 18)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()           

def rectify_lines(names, output_path):
    for name in names:
        line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
        if not os.path.exists(line_save_path):
            continue
        with open(line_save_path, 'r') as fp:
            lines = fp.readlines()
        flag = True
        for line in lines:
            x, y = coordinate_parse(line)
            if x == [] or y == []:
                continue
            x = np.array(x)
            y = np.array(y)

            p_init = np.random.randn(3)
            para_x = leastsq(resudual, p_init, args=(x, y))
            y_temp = func(para_x[0], x)
            y_error = np.mean(np.square(y_temp-y))

            para_y = leastsq(resudual, p_init, args=(y, x))
            x_temp = func(para_y[0], y)
            x_error = np.mean(np.square(x_temp-x))

            if x_error > y_error:
                x_new = np.linspace(min(x), max(x), 18)
                y_new = func(para_x[0], x_new)
            else:
                y_new = np.linspace(min(y), max(y), 41)
                x_new = func(para_y[0], y_new)

            if flag:
                fp = open(line_save_path, 'w')
                flag = False
            else:
                fp = open(line_save_path, 'a')
            for i in range(x_new.shape[0]):
                fp.write('%.3f %.3f '% ( x_new[i], y_new[i] ))
            fp.write('\n')
            fp.close()
        if flag:
            fp = open(line_save_path, 'w')
            fp.close()

def run_test(data_samples, data_batch, exp_name, work_dir, row_anchor = None, col_anchor = None):
    # torch.backends.cudnn.benchmark = True
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()

    imgs, names = data_batch
    imgs = imgs.cuda()
    
    pred = data_samples 
    
    generate_lines_local(pred['loc_row'],pred['exist_row'], names, output_path, 'normal', row_anchor=row_anchor)
    generate_lines_col_local(pred['loc_col'],pred['exist_col'], names, output_path, 'normal', col_anchor=col_anchor)

def generate_lines_local_tta(loc_row, loc_row_left, loc_row_right, exist_row, exist_row_left, exist_row_right, names, output_path, row_anchor):

    local_width = 1

    max_indices = loc_row.argmax(1).cpu()
    valid = exist_row.argmax(1).cpu()
    loc_row = loc_row.cpu()

    max_indices_left = loc_row_left.argmax(1).cpu()
    valid_left = exist_row_left.argmax(1).cpu()
    loc_row_left = loc_row_left.cpu()

    max_indices_right = loc_row_right.argmax(1).cpu()
    valid_right = exist_row_right.argmax(1).cpu()
    loc_row_right = loc_row_right.cpu()

    batch_size, num_grid, num_cls, num_lane = loc_row.shape

    min_lane_length = num_cls / 2

    for batch_idx in range(batch_size):

        name = names[batch_idx]
        line_save_path = os.path.join(output_path, name.replace('jpg', 'lines.txt'))
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'w') as fp:
            # for lane_idx in range(num_lane):
            for lane_idx in [1,2]:
                if valid[batch_idx,:,lane_idx].sum() >= min_lane_length:
                    pt_all = []
                    for cls_idx in range(num_cls):
                        cnt = 0
                        out_tmp_all = 0
                        if valid[batch_idx,cls_idx,lane_idx]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp = (loc_row[batch_idx,all_ind,cls_idx,lane_idx].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_tmp / (num_grid-1) * 1640
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp

                        if valid_left[batch_idx,cls_idx,lane_idx]:
                            all_ind_left = torch.tensor(list(range(max(0,max_indices_left[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_left[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                        
                            out_tmp_left = (loc_row_left[batch_idx,all_ind_left,cls_idx,lane_idx].softmax(0) * all_ind_left.float()).sum() + 0.5 
                            out_tmp_left = out_tmp_left / (num_grid-1) * 1640 + 1640./25
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_left

                        if valid_right[batch_idx,cls_idx,lane_idx]:
                            all_ind_right = torch.tensor(list(range(max(0,max_indices_right[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_right[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                        
                            out_tmp_right = (loc_row_right[batch_idx,all_ind_right,cls_idx,lane_idx].softmax(0) * all_ind_right.float()).sum() + 0.5 
                            out_tmp_right = out_tmp_right / (num_grid-1) * 1640 - 1640./25
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_right


                        if cnt >= 2:
                            pt_all.append(( out_tmp_all/cnt , row_anchor[cls_idx] * 590))
                    if len(pt_all) < min_lane_length:
                            continue
                    for pt in pt_all:
                        fp.write('%.3f %.3f '% pt)
                    fp.write('\n')

def generate_lines_col_local_tta(loc_col, loc_col_up, loc_col_down, exist_col, exist_col_up, exist_col_down, names, output_path, col_anchor):
    local_width = 1
    
    max_indices = loc_col.argmax(1).cpu()
    valid = exist_col.argmax(1).cpu()
    loc_col = loc_col.cpu()

    max_indices_up = loc_col_up.argmax(1).cpu()
    valid_up = exist_col_up.argmax(1).cpu()
    loc_col_up = loc_col_up.cpu()

    max_indices_down = loc_col_down.argmax(1).cpu()
    valid_down = exist_col_down.argmax(1).cpu()
    loc_col_down = loc_col_down.cpu()

    batch_size, num_grid, num_cls, num_lane = loc_col.shape

    min_lane_length = num_cls / 4

    for batch_idx in range(batch_size):

        name = names[batch_idx]
        line_save_path = os.path.join(output_path, name.replace('jpg','lines.txt'))
        save_dir, _ = os.path.split(line_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(line_save_path, 'a') as fp:
            # for lane_idx in range(num_lane):
            for lane_idx in [0,3]:
                if valid[batch_idx,:,lane_idx].sum() >= min_lane_length:
                    pt_all = []
                    for cls_idx in range(num_cls):
                        cnt = 0
                        out_tmp_all = 0
                        if valid[batch_idx,cls_idx,lane_idx]:
                            all_ind = torch.tensor(list(range(max(0,max_indices[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp = (loc_col[batch_idx,all_ind,cls_idx,lane_idx].softmax(0) * all_ind.float()).sum() + 0.5 
                            out_tmp = out_tmp / (num_grid-1) * 590
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp

                        if valid_up[batch_idx,cls_idx,lane_idx]:
                            all_ind_up = torch.tensor(list(range(max(0,max_indices_up[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_up[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp_up = (loc_col_up[batch_idx,all_ind_up,cls_idx,lane_idx].softmax(0) * all_ind_up.float()).sum() + 0.5 
                            out_tmp_up = out_tmp_up / (num_grid-1) * 590 + 32./534*590
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_up
                        if valid_down[batch_idx,cls_idx,lane_idx]:
                            all_ind_down = torch.tensor(list(range(max(0,max_indices_down[batch_idx,cls_idx,lane_idx] - local_width), min(num_grid-1, max_indices_down[batch_idx,cls_idx,lane_idx] + local_width) + 1)))
                            out_tmp_down = (loc_col_down[batch_idx,all_ind_down,cls_idx,lane_idx].softmax(0) * all_ind_down.float()).sum() + 0.5 
                            out_tmp_down = out_tmp_down / (num_grid-1) * 590 - 32./534*590     
                            cnt += 1
                            out_tmp_all = out_tmp_all + out_tmp_down

                        if cnt >= 2:
                            pt_all.append(( col_anchor[cls_idx] * 1640, out_tmp_all/cnt ))
                    if len(pt_all) < min_lane_length:
                        continue
                    for pt in pt_all:
                        fp.write('%.3f %.3f '% pt)
                    fp.write('\n')

def run_test_tta(data_samples, data_batch, exp_name, work_dir, row_anchor = None, col_anchor = None):
    
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()

    imgs, names = data_batch
    imgs = imgs.cuda()

    pred = data_samples

    loc_row, loc_row_left, loc_row_right, _, _ = torch.chunk(pred['loc_row'], 5)
    loc_col, _, _, loc_col_up, loc_col_down = torch.chunk(pred['loc_col'], 5)

    exist_row, exist_row_left, exist_row_right, _, _ = torch.chunk(pred['exist_row'], 5)
    exist_col, _, _, exist_col_up, exist_col_down = torch.chunk(pred['exist_col'], 5)

    generate_lines_local_tta(loc_row, loc_row_left, loc_row_right, exist_row, exist_row_left, exist_row_right, names, output_path, row_anchor)
    generate_lines_col_local_tta(loc_col, loc_col_up, loc_col_down, exist_col, exist_col_up, exist_col_down, names, output_path, col_anchor)

def eval_lane(data_samples, data_batch, dataset='CULane', tta=None, data_root=None, 
              test_work_dir=None, num_row=None, num_col=None):
    
    row_anchor = np.linspace(0.42,1, num_row)
    col_anchor = np.linspace(0,1, num_col)
    if dataset == 'CULane':
        if not tta:
            run_test(data_samples, data_batch, 'culane_eval_tmp', test_work_dir, row_anchor, col_anchor)
        else:
            run_test_tta(data_samples, data_batch, 'culane_eval_tmp', test_work_dir, row_anchor, col_anchor)
        synchronize()    # wait for all results
        if is_main_process():
            res = call_culane_eval(data_root, 'culane_eval_tmp', test_work_dir)
            TP,FP,FN = 0,0,0
            for k, v in res.items():
                val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
                val_tp,val_fp,val_fn = int(v['tp']),int(v['fp']),int(v['fn'])
                TP += val_tp
                FP += val_fp
                FN += val_fn
                # dist_print(k,val)

            if TP + FP == 0:
                P = 0
                print("nearly no results!")
            else:
                P = TP * 1.0/(TP + FP)
            if TP + FN == 0:
                R = 0
                print("nearly no results!")
            else:
                R = TP * 1.0/(TP + FN)
            if (P+R) == 0:
                F = 0
            else:
                F = 2*P*R/(P + R)
            # dist_print(F)
              
        synchronize()
        if is_main_process():
            return F
        else:
            return None

def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k : v for k,v in zip(keys,values)}
    return res

def call_culane_eval(data_dir, exp_name, output_path):

    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir=os.path.join(output_path,exp_name)+'/'

    w_lane=30
    iou=0.5  # Set iou to 0.3 or 0.5
    im_w=1640
    im_h=590
    frame=1
    list0 = os.path.join(data_dir,'list/test_split/test0_normal.txt')
    list1 = os.path.join(data_dir,'list/test_split/test1_crowd.txt')
    list2 = os.path.join(data_dir,'list/test_split/test2_hlight.txt')
    list3 = os.path.join(data_dir,'list/test_split/test3_shadow.txt')
    list4 = os.path.join(data_dir,'list/test_split/test4_noline.txt')
    list5 = os.path.join(data_dir,'list/test_split/test5_arrow.txt')
    list6 = os.path.join(data_dir,'list/test_split/test6_curve.txt')
    list7 = os.path.join(data_dir,'list/test_split/test7_cross.txt')
    list8 = os.path.join(data_dir,'list/test_split/test8_night.txt')
    if not os.path.exists(os.path.join(output_path,'txt')):
        os.mkdir(os.path.join(output_path,'txt'))
    out0 = os.path.join(output_path,'txt','out0_normal.txt')
    out1=os.path.join(output_path,'txt','out1_crowd.txt')
    out2=os.path.join(output_path,'txt','out2_hlight.txt')
    out3=os.path.join(output_path,'txt','out3_shadow.txt')
    out4=os.path.join(output_path,'txt','out4_noline.txt')
    out5=os.path.join(output_path,'txt','out5_arrow.txt')
    out6=os.path.join(output_path,'txt','out6_curve.txt')
    out7=os.path.join(output_path,'txt','out7_cross.txt')
    out8=os.path.join(output_path,'txt','out8_night.txt')

    eval_cmd = './evaluation/culane/evaluate'
    if platform.system() == 'Windows':
        eval_cmd = eval_cmd.replace('/', os.sep)

    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(eval_cmd,data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    res_all = {}
    res_all['res_normal'] = read_helper(out0)
    res_all['res_crowd']= read_helper(out1)
    res_all['res_night']= read_helper(out8)
    res_all['res_noline'] = read_helper(out4)
    res_all['res_shadow'] = read_helper(out3)
    res_all['res_arrow']= read_helper(out5)
    res_all['res_hlight'] = read_helper(out2)
    res_all['res_curve']= read_helper(out6)
    res_all['res_cross']= read_helper(out7)
    return res_all

@METRICS.register_module()
class culaneF(BaseMetric):

    def __init__(self,
                 dataset='CULane',tta=False, data_root=None, test_work_dir=None,
                 num_row=72, num_col=81, **kwargs) -> None:
        super().__init__()
        self.dataset = dataset
        self.tta = tta
        self.data_root = data_root
        self.test_work_dir = test_work_dir
        self.num_row = num_row
        self.num_col = num_col

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.  直接从val_dataloader过来的原始数据
            data_samples (Sequence[dict]): A batch of outputs from 
                the model.  在loss里面是 predict
        """
        res=eval_lane(data_samples, data_batch, self.dataset, self.tta, self.data_root, self.test_work_dir,
                    self.num_row, self.num_col)
        
        return self.results.append(res)
