import mmengine.fileio as fileio
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class S2DSDeepCrackDataset(BaseSegDataset):
    """Merged dam-defect dataset for 7-class semantic segmentation (mmseg style).

    数据集结构（data_root 下）：
        - train_img: 训练图像（RGB .png）
        - train_lab: 训练标注（单通道 class-id .png）
        - val_img:   验证图像
        - val_lab:   验证标注
        - test_img:  测试图像
        - test_lab:  测试标注

    配对规则：
        在任意 split 中，图像与标注按“同名文件”配对：
            <split>_img/<stem>.png  <->  <split>_lab/<stem>.png

    标注格式：
        - 标注为单通道 PNG（L 模式），像素值为类别 id（uint8）。
        - 有效类别 id 范围：0..6
        - ignore_index = 255：像素值为 255 的位置不参与 loss 与评估。

    类别定义（id -> class）：
        0 background
        1 crack
        2 spalling
        3 corrosion
        4 efflorescence
        5 vegetation
        6 control_point
    """

    METAINFO = dict(
        classes=(
            "background",
            "crack",
            "spalling",
            "corrosion",
            "efflorescence",
            "vegetation",
            "control_point",
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
        img_suffix=".png",
        # 标注文件与图像同名：xxx.png -> xxx.png
        seg_map_suffix=".png",
        reduce_zero_label=False,
        **kwargs,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )

        # 确认图像路径存在，避免路径写错（保持与你的示例一致）
        assert fileio.exists(
            self.data_prefix["img_path"], backend_args=self.backend_args
        ), f"图像路径不存在: {self.data_prefix['img_path']}"
