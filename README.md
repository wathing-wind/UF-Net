# UF-Net

- `mmyolo`: contains the code for UF-Net's traffic object detection task.
- `mmsegmentation`: contains code for lane line detection and segmentation of the drivable area.
- `YOLOP+GHS+ASS`: code containing the supplementary experimental part of the article.
## Requirement
 `pip install -r requirements.txt`
## Download pre-training weights
1. Download our weights at the following address：`https://pan.baidu.com/s/1XErEnN991n5xBT5vzkbVFQ?pwd=wh6a`
`password：wh6a `
--来自百度网盘超级会员V5的分享

### traffic object detection task of UF-Net 

test：`python mmsegmentation/tools/test.py mmsegmentation/configs/mtl_ad/task_det.py path/to/your/weights`

### drivable area segmentation of UF-Net

test：`python mmsegmentation/tools/test.py mmsegmentation/configs/mtl_ad/task_culane.py path/to/your/weights`

### lane line detection of UF-Net

test：`python mmsegmentation/tools/test.py mmsegmentation/configs/mtl_ad/task_seg.py path/to/your/weights`

### YOLOP+GHS+ASS

test：`python tools/test.py --weights path/to/your/weights`

