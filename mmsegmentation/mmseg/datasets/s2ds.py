import mmengine.fileio as fileio
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class S2DSDataset(BaseSegDataset):
    """S2DS dataset for multi-class concrete surface defect segmentation.

    数据集结构：
        - train_img: 训练图像
        - train_lab: 训练标注（RGBA, 颜色掩码）
        - val_img: 验证图像
        - val_lab: 验证标注
        - test_img: 测试图像
        - test_lab: 测试标注

    标注为 RGBA 彩色掩码，但语义只在 RGB 三通道中：

        (0, 0, 0)       -> 0  background
        (255, 255, 255) -> 1  crack
        (255, 0, 0)     -> 2  spalling
        (255, 255, 0)   -> 3  corrosion
        (0, 255, 255)   -> 4  efflorescence
        (0, 255, 0)     -> 5  vegetation
        (0, 0, 255)     -> 6  control_point
    """

    METAINFO = dict(
        classes=(
            'background',
            'crack',
            'spalling',
            'corrosion',
            'efflorescence',
            'vegetation',
            'control_point',
        ),
        palette=[
            [0, 0, 0],          # background
            [255, 255, 255],    # crack
            [255, 0, 0],        # spalling
            [255, 255, 0],      # corrosion
            [0, 255, 255],      # efflorescence
            [0, 255, 0],        # vegetation
            [0, 0, 255],        # control_point
        ],
    )

    def __init__(
        self,
        img_suffix='.png',
        # 标注文件形如 xxx_lab.png：015.png -> 015_lab.png
        seg_map_suffix='_lab.png',
        reduce_zero_label=False,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )

        # 确认图像路径存在，避免路径写错
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args
        ), f"图像路径不存在: {self.data_prefix['img_path']}"


