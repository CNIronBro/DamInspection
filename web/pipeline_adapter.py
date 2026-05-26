"""Pipeline 适配器 — 将 InferencePipeline 封装为 Web API 可调用的函数"""

import base64
import tempfile
import os
import uuid
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "/root/work/pic"))


def _to_native(obj):
    """递归将 numpy 类型转为 Python 原生类型，确保 JSON 可序列化"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj

# 将项目根目录加入 sys.path，以便 import infer_and_measure
import sys
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from infer_and_measure import InferencePipeline, Instance


def load_pipeline(config_path: str) -> InferencePipeline:
    """加载推理 pipeline（单例由 app.py 管理）"""
    overrides = {
        'io': {'output_dir': tempfile.mkdtemp(prefix='dam_web_')},
        'model': {'show_progress': False},
    }
    return InferencePipeline(config_path, overrides=overrides)


def get_classes_config(pipeline: InferencePipeline) -> dict:
    """从 pipeline 配置中提取类别信息供前端使用"""
    palette_bgr = pipeline.cfg['classes']['palette']
    color_space = pipeline.cfg['classes'].get('color_space', 'BGR')

    palette_rgb = {}
    for name, color in palette_bgr.items():
        if color_space == 'BGR':
            palette_rgb[name] = [color[2], color[1], color[0]]
        else:
            palette_rgb[name] = list(color)

    return {
        "classes": {
            "crack": {"color_rgb": palette_rgb.get("crack", [255, 255, 255]), "label_cn": "裂缝"},
            "spalling": {"color_rgb": palette_rgb.get("spalling", [0, 255, 255]), "label_cn": "剥落"},
            "efflorescence": {"color_rgb": palette_rgb.get("efflorescence", [255, 255, 0]), "label_cn": "泛碱"},
        },
        "mm_per_px": pipeline.cfg['measurement']['mm_per_px'],
    }


def _extract_contour(binary_mask: np.ndarray) -> List[List[int]]:
    """从二值 mask 中提取轮廓点列表 [[x,y], ...]"""
    mask_u8 = binary_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    # 取最大轮廓
    contour = max(contours, key=cv2.contourArea)
    # 裂缝等细长区域可能有大量点，用 approxPolyDP 简化
    epsilon = 1.0
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    # 转为 [[x,y], ...] 列表
    return [[int(pt[0][0]), int(pt[0][1])] for pt in approx]


def _bgr_to_rgb(color: list) -> list:
    """BGR → RGB 转换"""
    return [color[2], color[1], color[0]]


def detect_from_bytes_sync(pipeline: InferencePipeline, image_bytes: bytes) -> dict:
    """
    核心函数：接收图片 bytes，运行推理，返回 JSON 可序列化的结果 dict。
    由 app.py 通过 asyncio.to_thread 调用。
    """
    # 写入临时文件
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)

        # ---- 推理 ----
        pred_raw, img_orig = pipeline._inference_single(tmp_path)
        h, w = img_orig.shape[:2]

        # 类别重映射
        pred_out = pipeline._apply_remap(pred_raw)

        # 颜色配置（BGR→RGB）
        palette_bgr = pipeline.cfg['classes']['palette']
        color_space = pipeline.cfg['classes'].get('color_space', 'BGR')

        class_colors_rgb = {}
        for name, color in palette_bgr.items():
            class_colors_rgb[name] = _bgr_to_rgb(color) if color_space == 'BGR' else list(color)

        # ---- 提取实例 ----
        instances_out = []
        summary_by_class = {}

        for class_id, class_name in [(1, 'crack'), (2, 'spalling'), (3, 'efflorescence')]:
            mask = (pred_out == class_id)
            if mask.sum() == 0:
                summary_by_class[class_name] = 0
                continue

            instance_list = pipeline.measurer.extract_instances(mask, class_name)
            summary_by_class[class_name] = len(instance_list)

            for inst in instance_list:
                # 量化
                if class_name == 'crack':
                    metrics = pipeline.measurer.measure_crack_instance(inst)
                else:
                    metrics = pipeline.measurer.measure_spalling_efflorescence_instance(inst, class_name)

                # 轮廓
                contour = _extract_contour(inst.mask.astype(np.uint8))

                # RLE
                rle = pipeline._encode_rle_simple(inst.mask.astype(np.uint8))
                # 确保 counts 是原生 int 列表
                rle['counts'] = [int(c) for c in rle['counts']]
                rle['start_with'] = int(rle['start_with'])

                # bbox
                bx, by, bw, bh = [int(v) for v in inst.bbox]

                instances_out.append({
                    "class_name": class_name,
                    "class_name_cn": pipeline.CLASS_NAMES_CN[class_name],
                    "instance_id": int(inst.instance_id),
                    "bbox": {"xmin": bx, "ymin": by, "xmax": bx + bw, "ymax": by + bh},
                    "centroid": {"x": int(inst.centroid[0]), "y": int(inst.centroid[1])},
                    "metrics": {k: float(v) for k, v in metrics.items()},
                    "contour": [[int(x), int(y)] for x, y in contour],
                    "rle": rle,
                })

        # ---- 保存原图到磁盘 ----
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.png"
        image_path = IMAGE_DIR / filename
        cv2.imwrite(str(image_path), img_orig)

        result = {
            "image": {
                "width": int(w),
                "height": int(h),
                "path": f"/images/{filename}",
            },
            "instances": instances_out,
            "classes": {
                "crack": {"color_rgb": class_colors_rgb.get("crack", [255, 255, 255]), "label_cn": "裂缝"},
                "spalling": {"color_rgb": class_colors_rgb.get("spalling", [0, 255, 255]), "label_cn": "剥落"},
                "efflorescence": {"color_rgb": class_colors_rgb.get("efflorescence", [255, 255, 0]), "label_cn": "泛碱"},
            },
            "summary": {
                "total_instances": len(instances_out),
                "by_class": summary_by_class,
            },
        }

        return _to_native(result)

    finally:
        if tmp_path and tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
