"""FastAPI 路由 — 图片拼接 + 检测"""

import asyncio
import json
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .pipeline_adapter import detect_from_bytes_sync

router = APIRouter(prefix="/api/stitch", tags=["stitch"])


def _stitch_images(images: List[np.ndarray], rows: int, cols: int, overlap: int = 0) -> np.ndarray:
    """将图片按网格拼接"""
    # 先按行拼接
    row_images = []
    for r in range(rows):
        row_imgs = images[r * cols : (r + 1) * cols]
        if overlap > 0 and len(row_imgs) > 1:
            # 裁掉重叠区域（每张图右侧裁掉 overlap，最后一张不裁）
            trimmed = []
            for i, img in enumerate(row_imgs):
                if i < len(row_imgs) - 1:
                    trimmed.append(img[:, :-overlap])
                else:
                    trimmed.append(img)
            row_images.append(np.hstack(trimmed))
        else:
            row_images.append(np.hstack(row_imgs))

    # 再按列拼接
    if overlap > 0 and len(row_images) > 1:
        trimmed = []
        for i, img in enumerate(row_images):
            if i < len(row_images) - 1:
                trimmed.append(img[:-overlap, :])
            else:
                trimmed.append(img)
        return np.vstack(trimmed)
    else:
        return np.vstack(row_images)


@router.post("/detect")
async def stitch_and_detect(
    files: List[UploadFile] = File(...),
    grid: str = Form(...),
    model_id: Optional[str] = Form(None),
    overlap: int = Form(0),
):
    """
    接收多张图片 + 网格配置，拼接后执行检测。
    grid JSON 格式: {"rows": 2, "cols": 3, "order": [0,1,2,3,4,5]}
    order 是 files 列表的索引顺序，按 row-major 排列。
    """
    # 解析网格配置
    try:
        grid_cfg = json.loads(grid)
        rows = grid_cfg["rows"]
        cols = grid_cfg["cols"]
        order = grid_cfg.get("order", list(range(len(files))))
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(400, f"网格配置格式错误: {e}")

    if len(files) != rows * cols:
        raise HTTPException(400, f"图片数量({len(files)})与网格大小({rows}x{cols}={rows * cols})不匹配")

    # 读取所有图片
    images_bytes = []
    for f in files:
        data = await f.read()
        images_bytes.append(data)

    # 按 order 排序并解码
    images = []
    for idx in order:
        data = images_bytes[idx]
        img_array = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"无法解码图片: {files[idx].filename}")
        images.append(img)

    # 检查所有图片尺寸一致
    h0, w0 = images[0].shape[:2]
    for i, img in enumerate(images):
        if img.shape[:2] != (h0, w0):
            raise HTTPException(400, f"图片尺寸不一致: {files[0].filename}({h0}x{w0}) vs {files[order[i]].filename}({img.shape[0]}x{img.shape[1]})")

    # 拼接
    stitched = _stitch_images(images, rows, cols, overlap)

    # 编码为 PNG bytes
    _, encoded = cv2.imencode('.png', stitched)
    stitched_bytes = encoded.tobytes()

    # 调用检测（detect_from_bytes_sync 内部会保存图片到磁盘）
    from .app import _get_pipeline
    pipeline = _get_pipeline(model_id)

    try:
        result = await asyncio.to_thread(
            detect_from_bytes_sync, pipeline, stitched_bytes
        )
    except Exception as e:
        raise HTTPException(500, f"推理失败: {str(e)}")

    return result
