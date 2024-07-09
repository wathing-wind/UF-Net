import copy
import os.path as osp
from typing import Any, Optional
from typing import List

from mmengine.fileio import get_local_path

# from .yolov5_coco import BatchShapePolicyDataset
from mmdet.datasets import BaseDetDataset, CocoDataset
from mmdet.datasets.api_wrappers import COCO

from ..registry import DATASETS, TASK_UTILS

single_cls = True   # just detect vehicle

class BatchShapePolicyDataset(BaseDetDataset):
    """Dataset with the batch shape policy that makes paddings with least
    pixels during batch inference process, which does not require the image
    scales of all batches to be the same throughout validation."""

    def __init__(self,
                 *args,
                 batch_shapes_cfg: Optional[dict] = None,
                 **kwargs):
        self.batch_shapes_cfg = batch_shapes_cfg
        super().__init__(*args, **kwargs)

    def full_init(self):
        """rewrite full_init() to be compatible with serialize_data in
        BatchShapePolicy."""
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()

        # batch_shapes_cfg
        if self.batch_shapes_cfg:
            batch_shapes_policy = TASK_UTILS.build(self.batch_shapes_cfg)
            self.data_list = batch_shapes_policy(self.data_list)
            del batch_shapes_policy

        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def prepare_data(self, idx: int) -> Any:
        """Pass the dataset to the pipeline during training to support mixed
        data augmentation, such as Mosaic and MixUp."""
        if self.test_mode is False:
            data_info = self.get_data_info(idx)
            data_info['dataset'] = self
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)

class BDDCocoDataset(CocoDataset):

    METAINFO = {
        'classes':
        ('car', 'bus', 'train', 'truck')
    }    
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501 把标签数据地址替换为local_path
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])   # 取了设定的类别
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}    # 将类别编ID号
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()   # 取了所有加载图片数组
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]   # 将图片加载入空数组
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])   # 将标签的id和图片id对应上
            raw_ann_info = self.coco.load_anns(ann_ids)    # 将标签加载入空数组
            total_ann_ids.extend(ann_ids)
            # 将加载的标签和图片修改成对应格式
            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:   # 保证标签里面的id是单一的
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list
    
    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)  # 判断config给的filter_empty_gt
        min_size = self.filter_cfg.get('min_size', 0)   # 判断config给的min_size

        # obtain images that contain annotation获取包含标注的图像
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories获取包含所需类别注释的图像
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []   # 有效数据信息数组
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if single_cls:
                self.cat_id = 0   # 将所有类别都归为0类
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
    
@DATASETS.register_module()
class YOLOv5BddDataset(BatchShapePolicyDataset, BDDCocoDataset):
    """按照yolov5_coco.py继承两个父类
    父类BatchShapePolicyDataset为BaseDetDataset的子类
    父类BDDCocoDatasetCocoDataset
        function:为实现筛选标签类别,使得car,bus,train,truck四类归类为car类
    """
    pass