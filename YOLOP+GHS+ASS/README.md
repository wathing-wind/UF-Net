
<div align="left">   

## YOLOP+GHS+ASS

>
> Zilong Zhou, Ping Liu, Haibo Huang
>
> 
>
> 


### Contributions

* 

* 
  
* 

### Results

#### Traffic Object Detection Result

| Model          | Recall(%) | mAP50(%) | Speed(fps) |
| -------------- | --------- | -------- | ---------- |
| `Multinet`     | 81.3      | 60.2     | 8.6        |
| `DLT-Net`      | 89.4      | 68.4     | 9.3        |
| `Faster R-CNN` | 81.2      | 64.9     | 8.8        |
| `YOLOv5s`      | 86.8      | 77.2     | 82         |
| `YOLOP`        | 89.2      | 76.5     | 41         |
| `YOLOP+GHS`    | 91.7      | 83.6     | 43.7       |
#### Drivable Area Segmentation Result

| Model         | mIOU(%) | Speed(fps) |
| ------------- | ------- | ---------- |
| `Multinet`    | 71.6    | 8.6        |
| `DLT-Net`     | 71.3    | 9.3        |
| `PSPNet`      | 89.6    | 11.1       |
| `YOLOP`       | 91.5    | 41         |
| `YOLOP+GHS`   | 92.5    | 43.7       |
#### Lane Detection Result:

| Model         |  IOU(%) |
| ------------- |  ------ |
| `ENet`        |  14.64  |
| `SCNN`        |  15.84  |
| `ENet-SAD`    |  16.02  |
| `YOLOP`       |  26.20  |
| `YOLOP+GHS`   |   27.5  |


### Visualization

#### Traffic Object Detection Result



#### Drivable Area Segmentation Result



#### Lane Detection Result



**Notes**: 

- 

---

---

### Requirement

This codebase has been developed with python version 3.7, PyTorch 1.7+ and torchvision 0.8+:

```
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
```

See `requirements.txt` for additional dependencies and version requirements.

```setup
pip install -r requirements.txt
```

### Data preparation

#### Download

- Download the images from [images](https://bdd-data.berkeley.edu/).

- Download the annotations of detection from [det_annotations](https://drive.google.com/file/d/1Ge-R8NTxG1eqd4zbryFo-1Uonuh0Nxyl/view?usp=sharing). 
- Download the annotations of drivable area segmentation from [da_seg_annotations](https://drive.google.com/file/d/1xy_DhUZRHR8yrZG3OwTQAHhYTnXn7URv/view?usp=sharing). 
- Download the annotations of lane line segmentation from [ll_seg_annotations](https://drive.google.com/file/d/1lDNTPIQj_YLNZVkksKM25CvCHuquJ8AP/view?usp=sharing). 

We recommend the dataset directory structure to be the following:

```
# The id represent the correspondence relation
├─dataset root
│ ├─images
│ │ ├─train
│ │ ├─val
│ ├─det_annotations
│ │ ├─train
│ │ ├─val
│ ├─da_seg_annotations
│ │ ├─train
│ │ ├─val
│ ├─ll_seg_annotations
│ │ ├─train
│ │ ├─val
```

Update the your dataset path in the `./lib/config/default.py`.

### Training

You can set the training configuration in the `./lib/config/default.py`. (Including:  the loading of preliminary model,  loss,  data augmentation, optimizer, warm-up and cosine annealing, auto-anchor, training epochs, batch_size).

If you want try alternating optimization or train model for single task, please modify the corresponding configuration in `./lib/config/default.py` to `True`. (As following, all configurations is `False`, which means training multiple tasks end to end).

```python
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task
```

Start training:

```shell
python tools/train.py
```
Multi GPU mode:
```shell
python -m torch.distributed.launch --nproc_per_node=N tools/train.py  # N: the number of GPUs
```


### Evaluation

You can set the evaluation configuration in the `./lib/config/default.py`. (Including： batch_size and threshold value for nms).

Start evaluating:

```shell
python tools/test.py --weights weights/End-to-end.pth
```



### Demo Test

We provide two testing method.

#### Folder

You can store the image or video in `--source`, and then save the reasoning result to `--save-dir`

```shell
python tools/demo.py --source inference/images
```



#### Camera

If there are any camera connected to your computer, you can set the `source` as the camera number(The default is 0).

```shell
python tools/demo.py --source 0
```




