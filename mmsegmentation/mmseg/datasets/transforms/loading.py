# Copyright (c) OpenMMLab. All rights reserved.
import torch
import warnings
from typing import Dict, Optional, Union, List, Sequence, Tuple

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile

from mmdet.structures.mask import PolygonMasks, polygon_to_bitmap
from mmdet.datasets.transforms import LoadAnnotations as MMDET_LoadAnnotations
from mmdet.structures.bbox import (HorizontalBoxes, autocast_box_type,
                                   get_box_type)
from .keypoint_structure import Keypoints

from mmseg.registry import TRANSFORMS
from mmseg.utils import datafrombytes

try:
    from osgeo import gdal
except ImportError:
    gdal = None


@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
        self,
        reduce_zero_label=None,
        backend_args=None,
        imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromNDArray(LoadImageFromFile):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def transform(self, results: dict) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = results['img']
        if self.to_float32:
            img = img.astype(np.float32)

        results['img_path'] = None
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadBiomedicalImageFromFile(BaseTransform):
    """Load an biomedical mage from file.

    Required Keys:

    - img_path

    Added Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities, and data type is float32
        if set to_float32 = True, or float64 if decode_backend is 'nifti' and
        to_float32 is False.
    - img_shape
    - ori_shape

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an float64 array.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 to_float32: bool = True,
                 backend_args: Optional[dict] = None) -> None:
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']

        data_bytes = fileio.get(filename, self.backend_args)
        img = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            img = img.astype(np.float32)

        if len(img.shape) == 3:
            img = img[None, ...]

        if self.decode_backend == 'nifti':
            img = img.transpose(0, 3, 2, 1)

        if self.to_xyz:
            img = img.transpose(0, 3, 2, 1)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadBiomedicalAnnotation(BaseTransform):
    """Load ``seg_map`` annotation provided by biomedical dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'gt_seg_map': np.ndarray (X, Y, Z) or (Z, Y, X)
        }

    Required Keys:

    - seg_map_path

    Added Keys:

    - gt_seg_map (np.ndarray): Biomedical seg map with shape (Z, Y, X) by
        default, and data type is float32 if set to_float32 = True, or
        float64 if decode_backend is 'nifti' and to_float32 is False.

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        to_float32 (bool): Whether to convert the loaded seg map to a float32
            numpy array. If set to False, the loaded image is an float64 array.
            Defaults to True.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See :class:`mmengine.fileio` for details.
            Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'nifti',
                 to_xyz: bool = False,
                 to_float32: bool = True,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.to_float32 = to_float32
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['seg_map_path'], self.backend_args)
        gt_seg_map = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            gt_seg_map = gt_seg_map.astype(np.float32)

        if self.decode_backend == 'nifti':
            gt_seg_map = gt_seg_map.transpose(2, 1, 0)

        if self.to_xyz:
            gt_seg_map = gt_seg_map.transpose(2, 1, 0)

        results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadBiomedicalData(BaseTransform):
    """Load an biomedical image and annotation from file.

    The loading data format is as the following:

    .. code-block:: python

        {
            'img': np.ndarray data[:-1, X, Y, Z]
            'seg_map': np.ndarray data[-1, X, Y, Z]
        }


    Required Keys:

    - img_path

    Added Keys:

    - img (np.ndarray): Biomedical image with shape (N, Z, Y, X) by default,
        N is the number of modalities.
    - gt_seg_map (np.ndarray, optional): Biomedical seg map with shape
        (Z, Y, X) by default.
    - img_shape
    - ori_shape

    Args:
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        decode_backend (str): The data decoding backend type. Options are
            'numpy'and 'nifti', and there is a convention that when backend is
            'nifti' the axis of data loaded is XYZ, and when backend is
            'numpy', the the axis is ZYX. The data will be transposed if the
            backend is 'nifti'. Defaults to 'nifti'.
        to_xyz (bool): Whether transpose data from Z, Y, X to X, Y, Z.
            Defaults to False.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 with_seg=False,
                 decode_backend: str = 'numpy',
                 to_xyz: bool = False,
                 backend_args: Optional[dict] = None) -> None:  # noqa
        self.with_seg = with_seg
        self.decode_backend = decode_backend
        self.to_xyz = to_xyz
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        data_bytes = fileio.get(results['img_path'], self.backend_args)
        data = datafrombytes(data_bytes, backend=self.decode_backend)
        # img is 4D data (N, X, Y, Z), N is the number of protocol
        img = data[:-1, :]

        if self.decode_backend == 'nifti':
            img = img.transpose(0, 3, 2, 1)

        if self.to_xyz:
            img = img.transpose(0, 3, 2, 1)

        results['img'] = img
        results['img_shape'] = img.shape[1:]
        results['ori_shape'] = img.shape[1:]

        if self.with_seg:
            gt_seg_map = data[-1, :]
            if self.decode_backend == 'nifti':
                gt_seg_map = gt_seg_map.transpose(2, 1, 0)

            if self.to_xyz:
                gt_seg_map = gt_seg_map.transpose(2, 1, 0)
            results['gt_seg_map'] = gt_seg_map
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'with_seg={self.with_seg}, '
                    f"decode_backend='{self.decode_backend}', "
                    f'to_xyz={self.to_xyz}, '
                    f'backend_args={self.backend_args})')
        return repr_str


@TRANSFORMS.register_module()
class InferencerLoader(BaseTransform):
    """Load an image from ``results['img']``.

    Similar with :obj:`LoadImageFromFile`, but the image has been loaded as
    :obj:`np.ndarray` in ``results['img']``. Can be used when loading image
    from webcam.

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_path
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.from_file = TRANSFORMS.build(
            dict(type='LoadImageFromFile', **kwargs))
        self.from_ndarray = TRANSFORMS.build(
            dict(type='LoadImageFromNDArray', **kwargs))

    def transform(self, single_input: Union[str, np.ndarray, dict]) -> dict:
        """Transform function to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        if isinstance(single_input, str):
            inputs = dict(img_path=single_input)
        elif isinstance(single_input, np.ndarray):
            inputs = dict(img=single_input)
        elif isinstance(single_input, dict):
            inputs = single_input
        else:
            raise NotImplementedError

        if 'img' in inputs:
            return self.from_ndarray(inputs)
        return self.from_file(inputs)


@TRANSFORMS.register_module()
class LoadSingleRSImageFromFile(BaseTransform):
    """Load a Remote Sensing mage from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        self.to_float32 = to_float32

        if gdal is None:
            raise RuntimeError('gdal is not installed')

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        ds = gdal.Open(filename)
        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        img = np.einsum('ijk->jki', ds.ReadAsArray())

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str


@TRANSFORMS.register_module()
class LoadMultipleRSImageFromFile(BaseTransform):
    """Load two Remote Sensing mage from file.

    Required Keys:

    - img_path
    - img_path2

    Modified Keys:

    - img
    - img2
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is a float64 array.
            Defaults to True.
    """

    def __init__(self, to_float32: bool = True):
        if gdal is None:
            raise RuntimeError('gdal is not installed')
        self.to_float32 = to_float32

    def transform(self, results: Dict) -> Dict:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        filename2 = results['img_path2']

        ds = gdal.Open(filename)
        ds2 = gdal.Open(filename2)

        if ds is None:
            raise Exception(f'Unable to open file: {filename}')
        if ds2 is None:
            raise Exception(f'Unable to open file: {filename2}')

        img = np.einsum('ijk->jki', ds.ReadAsArray())
        img2 = np.einsum('ijk->jki', ds2.ReadAsArray())

        if self.to_float32:
            img = img.astype(np.float32)
            img2 = img2.astype(np.float32)

        if img.shape != img2.shape:
            raise Exception(f'Image shapes do not match:'
                            f' {img.shape} vs {img2.shape}')

        results['img'] = img
        results['img2'] = img2
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str


@TRANSFORMS.register_module()
class LoadDepthAnnotation(BaseTransform):
    """Load ``depth_map`` annotation provided by depth estimation dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'gt_depth_map': np.ndarray [Y, X]
        }

    Required Keys:

    - seg_depth_path

    Added Keys:

    - gt_depth_map (np.ndarray): Depth map with shape (Y, X) by
        default, and data type is float32 if set to_float32 = True.
    - depth_rescale_factor (float): The rescale factor of depth map, which
        can be used to recover the original value of depth map.

    Args:
        decode_backend (str): The data decoding backend type. Options are
            'numpy', 'nifti', and 'cv2'. Defaults to 'cv2'.
        to_float32 (bool): Whether to convert the loaded depth map to a float32
            numpy array. If set to False, the loaded image is an uint16 array.
            Defaults to True.
        depth_rescale_factor (float): Factor to rescale the depth value to
            limit the range. Defaults to 1.0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See :class:`mmengine.fileio` for details.
            Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 decode_backend: str = 'cv2',
                 to_float32: bool = True,
                 depth_rescale_factor: float = 1.0,
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        self.decode_backend = decode_backend
        self.to_float32 = to_float32
        self.depth_rescale_factor = depth_rescale_factor
        self.backend_args = backend_args.copy() if backend_args else None

    def transform(self, results: Dict) -> Dict:
        """Functions to load depth map.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded depth map.
        """
        data_bytes = fileio.get(results['depth_map_path'], self.backend_args)
        gt_depth_map = datafrombytes(data_bytes, backend=self.decode_backend)

        if self.to_float32:
            gt_depth_map = gt_depth_map.astype(np.float32)

        gt_depth_map *= self.depth_rescale_factor
        results['gt_depth_map'] = gt_depth_map
        results['seg_fields'].append('gt_depth_map')
        results['depth_rescale_factor'] = self.depth_rescale_factor
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str

# @TRANSFORMS.register_module()
# class LoadAnnotations_yolo(MMDET_LoadAnnotations):
#     """Because the yolo series does not need to consider ignore bboxes for the
#     time being, in order to speed up the pipeline, it can be excluded in
#     advance.

#     Args:
#         mask2bbox (bool): Whether to use mask annotation to get bbox.
#             Defaults to False.
#         poly2mask (bool): Whether to transform the polygons to bitmaps.
#             Defaults to False.
#         merge_polygons (bool): Whether to merge polygons into one polygon.
#             If merged, the storage structure is simpler and training is more
#             effcient, especially if the mask inside a bbox is divided into
#             multiple polygons. Defaults to True.
#     """

#     def __init__(self,
#                  mask2bbox: bool = False,
#                  poly2mask: bool = False,
#                  merge_polygons: bool = True,
#                  **kwargs):
#         self.mask2bbox = mask2bbox
#         self.merge_polygons = merge_polygons
#         assert not poly2mask, 'Does not support BitmapMasks considering ' \
#                               'that bitmap consumes more memory.'
#         super().__init__(poly2mask=poly2mask, **kwargs)
#         if self.mask2bbox:
#             assert self.with_mask, 'Using mask2bbox requires ' \
#                                    'with_mask is True.'
#         self._mask_ignore_flag = None

#     def transform(self, results: dict) -> dict:
#         """Function to load multiple types annotations.

#         Args:
#             results (dict): Result dict from :obj:``mmengine.BaseDataset``.

#         Returns:
#             dict: The dict contains loaded bounding box, label and
#             semantic segmentation.
#         """
#         if self.mask2bbox:
#             self._load_masks(results)
#             if self.with_label:
#                 self._load_labels(results)
#                 self._update_mask_ignore_data(results)
#             gt_bboxes = results['gt_masks'].get_bboxes(dst_type='hbox')
#             results['gt_bboxes'] = gt_bboxes
#         elif self.with_keypoints:
#             self._load_kps(results)
#             _, box_type_cls = get_box_type(self.box_type)
#             results['gt_bboxes'] = box_type_cls(
#                 results.get('bbox', []), dtype=torch.float32)
#         else:
#             results = super().transform(results)
#             self._update_mask_ignore_data(results)
#         return results

#     def _update_mask_ignore_data(self, results: dict) -> None:
#         if 'gt_masks' not in results:
#             return

#         if 'gt_bboxes_labels' in results and len(
#                 results['gt_bboxes_labels']) != len(results['gt_masks']):
#             assert len(results['gt_bboxes_labels']) == len(
#                 self._mask_ignore_flag)
#             results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
#                 self._mask_ignore_flag]

#         if 'gt_bboxes' in results and len(results['gt_bboxes']) != len(
#                 results['gt_masks']):
#             assert len(results['gt_bboxes']) == len(self._mask_ignore_flag)
#             results['gt_bboxes'] = results['gt_bboxes'][self._mask_ignore_flag]

#     def _load_bboxes(self, results: dict):
#         """Private function to load bounding box annotations.
#         Note: BBoxes with ignore_flag of 1 is not considered.
#         Args:
#             results (dict): Result dict from :obj:``mmengine.BaseDataset``.

#         Returns:
#             dict: The dict contains loaded bounding box annotations.
#         """
#         gt_bboxes = []
#         gt_ignore_flags = []
#         for instance in results.get('instances', []):
#             if instance['ignore_flag'] == 0:
#                 gt_bboxes.append(instance['bbox'])
#                 gt_ignore_flags.append(instance['ignore_flag'])
#         results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

#         if self.box_type is None:
#             results['gt_bboxes'] = np.array(
#                 gt_bboxes, dtype=np.float32).reshape((-1, 4))
#         else:
#             _, box_type_cls = get_box_type(self.box_type)
#             results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)

#     def _load_labels(self, results: dict):
#         """Private function to load label annotations.

#         Note: BBoxes with ignore_flag of 1 is not considered.
#         Args:
#             results (dict): Result dict from :obj:``mmengine.BaseDataset``.
#         Returns:
#             dict: The dict contains loaded label annotations.
#         """
#         gt_bboxes_labels = []
#         for instance in results.get('instances', []):
#             if instance['ignore_flag'] == 0:
#                 gt_bboxes_labels.append(instance['bbox_label'])
#         results['gt_bboxes_labels'] = np.array(
#             gt_bboxes_labels, dtype=np.int64)

#     def _load_masks(self, results: dict) -> None:
#         """Private function to load mask annotations.

#         Args:
#             results (dict): Result dict from :obj:``mmengine.BaseDataset``.
#         """
#         gt_masks = []
#         gt_ignore_flags = []
#         self._mask_ignore_flag = []
#         for instance in results.get('instances', []):
#             if instance['ignore_flag'] == 0:
#                 if 'mask' in instance:
#                     gt_mask = instance['mask']
#                     if isinstance(gt_mask, list):
#                         gt_mask = [
#                             np.array(polygon) for polygon in gt_mask
#                             if len(polygon) % 2 == 0 and len(polygon) >= 6
#                         ]
#                         if len(gt_mask) == 0:
#                             # ignore
#                             self._mask_ignore_flag.append(0)
#                         else:
#                             if len(gt_mask) > 1 and self.merge_polygons:
#                                 gt_mask = self.merge_multi_segment(gt_mask)
#                             gt_masks.append(gt_mask)
#                             gt_ignore_flags.append(instance['ignore_flag'])
#                             self._mask_ignore_flag.append(1)
#                     else:
#                         raise NotImplementedError(
#                             'Only supports mask annotations in polygon '
#                             'format currently')
#                 else:
#                     # TODO: Actually, gt with bbox and without mask needs
#                     #  to be retained
#                     self._mask_ignore_flag.append(0)
#         self._mask_ignore_flag = np.array(self._mask_ignore_flag, dtype=bool)
#         results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

#         h, w = results['ori_shape']
#         gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
#         results['gt_masks'] = gt_masks

#     def merge_multi_segment(self,
#                             gt_masks: List[np.ndarray]) -> List[np.ndarray]:
#         """Merge multi segments to one list.

#         Find the coordinates with min distance between each segment,
#         then connect these coordinates with one thin line to merge all
#         segments into one.
#         Args:
#             gt_masks(List(np.array)):
#                 original segmentations in coco's json file.
#                 like [segmentation1, segmentation2,...],
#                 each segmentation is a list of coordinates.
#         Return:
#             gt_masks(List(np.array)): merged gt_masks
#         """
#         s = []
#         segments = [np.array(i).reshape(-1, 2) for i in gt_masks]
#         idx_list = [[] for _ in range(len(gt_masks))]

#         # record the indexes with min distance between each segment
#         for i in range(1, len(segments)):
#             idx1, idx2 = self.min_index(segments[i - 1], segments[i])
#             idx_list[i - 1].append(idx1)
#             idx_list[i].append(idx2)

#         # use two round to connect all the segments
#         # first round: first to end, i.e. A->B(partial)->C
#         # second round: end to first, i.e. C->B(remaining)-A
#         for k in range(2):
#             # forward first round
#             if k == 0:
#                 for i, idx in enumerate(idx_list):
#                     # middle segments have two indexes
#                     # reverse the index of middle segments
#                     if len(idx) == 2 and idx[0] > idx[1]:
#                         idx = idx[::-1]
#                         segments[i] = segments[i][::-1, :]
#                     # add the idx[0] point for connect next segment
#                     segments[i] = np.roll(segments[i], -idx[0], axis=0)
#                     segments[i] = np.concatenate(
#                         [segments[i], segments[i][:1]])
#                     # deal with the first segment and the last one
#                     if i in [0, len(idx_list) - 1]:
#                         s.append(segments[i])
#                     # deal with the middle segment
#                     # Note that in the first round, only partial segment
#                     # are appended.
#                     else:
#                         idx = [0, idx[1] - idx[0]]
#                         s.append(segments[i][idx[0]:idx[1] + 1])
#             # forward second round
#             else:
#                 for i in range(len(idx_list) - 1, -1, -1):
#                     # deal with the middle segment
#                     # append the remaining points
#                     if i not in [0, len(idx_list) - 1]:
#                         idx = idx_list[i]
#                         nidx = abs(idx[1] - idx[0])
#                         s.append(segments[i][nidx:])
#         return [np.concatenate(s).reshape(-1, )]

#     def min_index(self, arr1: np.ndarray, arr2: np.ndarray) -> Tuple[int, int]:
#         """Find a pair of indexes with the shortest distance.

#         Args:
#             arr1: (N, 2).
#             arr2: (M, 2).
#         Return:
#             tuple: a pair of indexes.
#         """
#         dis = ((arr1[:, None, :] - arr2[None, :, :])**2).sum(-1)
#         return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

#     def _load_kps(self, results: dict) -> None:
#         """Private function to load keypoints annotations.

#         Args:
#             results (dict): Result dict from
#                 :class:`mmengine.dataset.BaseDataset`.

#         Returns:
#             dict: The dict contains loaded keypoints annotations.
#         """
#         results['height'] = results['img_shape'][0]
#         results['width'] = results['img_shape'][1]
#         num_instances = len(results.get('bbox', []))

#         if num_instances == 0:
#             results['keypoints'] = np.empty(
#                 (0, len(results['flip_indices']), 2), dtype=np.float32)
#             results['keypoints_visible'] = np.empty(
#                 (0, len(results['flip_indices'])), dtype=np.int32)
#             results['category_id'] = []

#         results['gt_keypoints'] = Keypoints(
#             keypoints=results['keypoints'],
#             keypoints_visible=results['keypoints_visible'],
#             flip_indices=results['flip_indices'],
#         )

#         results['gt_ignore_flags'] = np.array([False] * num_instances)
#         results['gt_bboxes_labels'] = np.array(results['category_id']) - 1

#     def __repr__(self) -> str:
#         repr_str = self.__class__.__name__
#         repr_str += f'(with_bbox={self.with_bbox}, '
#         repr_str += f'with_label={self.with_label}, '
#         repr_str += f'with_mask={self.with_mask}, '
#         repr_str += f'with_seg={self.with_seg}, '
#         repr_str += f'mask2bbox={self.mask2bbox}, '
#         repr_str += f'poly2mask={self.poly2mask}, '
#         repr_str += f"imdecode_backend='{self.imdecode_backend}', "
#         repr_str += f'backend_args={self.backend_args})'
#         return repr_str
