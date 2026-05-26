"""
自定义Transform：标注值映射

用于将DeepCrack数据集的标注值255映射为类别索引1
"""

import copy
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS
from .transforms import RandomCrop


@TRANSFORMS.register_module()
class ConvertSegMapToGrayscale(BaseTransform):
    """将多通道标注图转换为单通道灰度图
    
    用于处理RGBA格式的PNG标注图，提取第一个通道或使用最大值投影。
    """

    def __init__(self, method='first'):
        """初始化
        
        Args:
            method (str): 转换方法
                - 'first': 使用第一个通道（默认）
                - 'max': 使用所有通道的最大值
                - 'mean': 使用所有通道的平均值
        """
        super().__init__()
        self.method = method

    def transform(self, results):
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            
            # 如果已经是2D，直接返回
            if gt_seg_map.ndim == 2:
                return results
            
            # 如果是多通道，转换为单通道
            if gt_seg_map.ndim == 3:
                if self.method == 'first':
                    gt_seg_map = gt_seg_map[:, :, 0]
                elif self.method == 'max':
                    gt_seg_map = np.max(gt_seg_map, axis=2)
                elif self.method == 'mean':
                    gt_seg_map = np.mean(gt_seg_map, axis=2).astype(np.uint8)
                else:
                    raise ValueError(f'Unknown method: {self.method}')
                
                results['gt_seg_map'] = gt_seg_map.astype(np.uint8)
        
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(method={self.method})'


@TRANSFORMS.register_module()
class BinarizeLabels(BaseTransform):
    """二值化标注：0=背景，>0=裂缝"""

    def __init__(self):
        """初始化，无需参数"""
        super().__init__()

    def transform(self, results):
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']
            # 简单粗暴：所有非零值都是裂缝
            results['gt_seg_map'] = (gt_seg_map > 0).astype(np.uint8)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@TRANSFORMS.register_module()
class MapLabels(BaseTransform):
    """将标注值映射为类别索引

    Args:
        mapping (dict): 映射字典，例如 {0: 0, 255: 1}
    """

    def __init__(self, mapping):
        super().__init__()  # ← 关键！必须调用父类初始化
        self.mapping = mapping

    def transform(self, results):
        """执行标注映射

        Args:
            results (dict): 包含标注的结果字典

        Returns:
            dict: 映射后的结果
        """
        # 获取标注
        if 'gt_seg_map' in results:
            gt_seg_map = results['gt_seg_map']

            # 创建映射后的标注
            mapped_seg = np.zeros_like(gt_seg_map)
            for old_val, new_val in self.mapping.items():
                mapped_seg[gt_seg_map == old_val] = new_val

            # 更新结果
            results['gt_seg_map'] = mapped_seg

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(mapping={self.mapping})'


@TRANSFORMS.register_module()
class ColorMapToLabel(BaseTransform):
    """将 RGB 颜色掩码转换为类别索引。

    典型用法：
        - 处理 RGBA / RGB 标注图（H x W x 3/4）
        - 根据指定的 (R, G, B) -> class_id 字典进行映射

    Args:
        color_map (dict): 颜色到类别 ID 的映射，
            例如 {(0, 0, 0): 0, (255, 255, 255): 1, ...}
    """

    def __init__(self, color_map):
        super().__init__()
        # 将 key 统一为 tuple，避免列表/元组混用问题
        self.color_map = {tuple(k): int(v) for k, v in color_map.items()}

    def transform(self, results):
        if 'gt_seg_map' not in results:
            return results

        gt = results['gt_seg_map']

        # 如果是 2D，说明已经是类别 ID 了，直接返回
        if gt.ndim == 2:
            return results

        # 只取前三个通道 (RGB)，忽略 alpha 通道
        if gt.ndim == 3 and gt.shape[2] >= 3:
            rgb = gt[:, :, :3].astype(np.uint8)
        else:
            raise ValueError(
                f'ColorMapToLabel 期望输入为 HxWx3/4，当前形状为 {gt.shape}')

        h, w, _ = rgb.shape
        label_map = np.zeros((h, w), dtype=np.uint8)

        # 逐颜色映射
        for (r, g, b), cls_id in self.color_map.items():
            mask = (
                (rgb[:, :, 0] == r) &
                (rgb[:, :, 1] == g) &
                (rgb[:, :, 2] == b)
            )
            label_map[mask] = cls_id

        results['gt_seg_map'] = label_map
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(num_colors={len(self.color_map)})'


@TRANSFORMS.register_module()
class ClassBalancedRandomCrop(BaseTransform):
    """优先裁剪包含指定类别的随机裁剪。

    典型用法：
        - 针对极少数类（如 crack=1），提高其在 crop 中出现的概率
        - 如果多次尝试仍找不到包含目标类的区域，则退化为普通 RandomCrop

    Args:
        crop_size (tuple): 裁剪尺寸 (h, w)。
        class_ids (Sequence[int]): 期望在裁剪区域中出现的类别 ID 列表。
        num_retry (int): 最多尝试次数。
        cat_max_ratio (float): 透传给 RandomCrop，控制单类最大占比。
        ignore_index (int): 透传给 RandomCrop。
    """

    def __init__(
        self,
        crop_size,
        class_ids=(1,),
        num_retry=10,
        cat_max_ratio=1.0,
        ignore_index=255,
    ):
        super().__init__()
        self.class_ids = tuple(int(c) for c in class_ids)
        self.num_retry = int(num_retry)
        self.crop_op = RandomCrop(
            crop_size=crop_size,
            cat_max_ratio=cat_max_ratio,
            ignore_index=ignore_index,
        )

    def transform(self, results):
        # 如果还没有标签，直接走普通裁剪
        if 'gt_seg_map' not in results:
            return self.crop_op.transform(results)

        last = None
        for _ in range(self.num_retry):
            tmp = copy.deepcopy(results)
            tmp = self.crop_op.transform(tmp)
            gt = tmp.get('gt_seg_map', None)
            if gt is None:
                last = tmp
                continue
            # 如果 crop 中包含任一目标类别，则接受本次裁剪
            if np.isin(gt, self.class_ids).any():
                return tmp
            last = tmp

        # 多次尝试均未命中目标类，退化为普通 RandomCrop 结果
        return last if last is not None else self.crop_op.transform(results)

    def __repr__(self):
        return (f'{self.__class__.__name__}(class_ids={self.class_ids}, '
                f'num_retry={self.num_retry})')