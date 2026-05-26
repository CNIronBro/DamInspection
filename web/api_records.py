"""FastAPI 路由 — 检测记录 CRUD + 统计 + 量化测量"""

from typing import Optional, List

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .database import (
    delete_record,
    get_record,
    get_stats_summary,
    list_records,
    save_record,
    update_record,
)

router = APIRouter(prefix="/api/records", tags=["records"])


# --------------- 请求模型 ---------------

class SaveRecordRequest(BaseModel):
    record_id: int = 0  # 0 = 新建，>0 = 更新已有记录
    image_name: str
    image_width: int = 0
    image_height: int = 0
    image_path: str = ""
    model_id: str = "segformer-b2"
    mm_per_px: float = 1.0
    total_instances: int = 0
    by_class: dict = {}
    instances: list = []


class MeasureRequest(BaseModel):
    points: List[List[int]]  # 多边形顶点 [[x,y], [x,y], ...]
    class_name: str           # crack / spalling / efflorescence
    image_width: int
    image_height: int
    mm_per_px: float = 1.0


# --------------- 量化测量 ---------------

def _mask_from_polygon(points: list, width: int, height: int) -> np.ndarray:
    """从多边形顶点生成布尔 mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _measure_crack(mask: np.ndarray, mm_per_px: float) -> dict:
    """裂缝量化 — 与 DefectMeasurer.measure_crack_instance 逻辑一致"""
    from skimage.morphology import skeletonize
    from scipy.ndimage import distance_transform_edt

    area_px = int(mask.sum())
    area_mm2 = area_px * (mm_per_px ** 2)

    skeleton = skeletonize(mask).astype(np.uint8)
    length_mm = int(skeleton.sum()) * mm_per_px

    dist = distance_transform_edt(mask)
    widths = 2 * dist[skeleton > 0] * mm_per_px

    width_mean_mm = float(np.mean(widths)) if len(widths) > 0 else 0.0
    width_max_mm = float(np.max(widths)) if len(widths) > 0 else 0.0
    width_p95_mm = float(np.percentile(widths, 95)) if len(widths) > 0 else 0.0

    return {
        "area_mm2": round(area_mm2, 2),
        "length_mm": round(length_mm, 2),
        "width_mean_mm": round(width_mean_mm, 2),
        "width_max_mm": round(width_max_mm, 2),
        "width_p95_mm": round(width_p95_mm, 2),
    }


def _measure_spalling_efflorescence(mask: np.ndarray, mm_per_px: float) -> dict:
    """剥落/泛碱量化 — 与 DefectMeasurer.measure_spalling_efflorescence_instance 逻辑一致"""
    area_px = int(mask.sum())
    area_mm2 = area_px * (mm_per_px ** 2)
    eq_diameter_mm = np.sqrt(4 * area_px / np.pi) * mm_per_px

    major_axis_mm = 0.0
    minor_axis_mm = 0.0
    coords = np.column_stack(np.where(mask))
    if len(coords) >= 3:
        rect = cv2.minAreaRect(coords.astype(np.float32))
        w, h = rect[1]
        major_axis_mm = max(w, h) * mm_per_px
        minor_axis_mm = min(w, h) * mm_per_px

    return {
        "area_mm2": round(area_mm2, 2),
        "eq_diameter_mm": round(eq_diameter_mm, 2),
        "major_axis_mm": round(major_axis_mm, 2),
        "minor_axis_mm": round(minor_axis_mm, 2),
    }


# --------------- 路由 ---------------

@router.post("/save")
async def api_save_record(req: SaveRecordRequest):
    """保存或更新检测记录（含所有实例）"""
    record = {
        "image_name": req.image_name,
        "image_width": req.image_width,
        "image_height": req.image_height,
        "image_path": req.image_path,
        "model_id": req.model_id,
        "mm_per_px": req.mm_per_px,
        "total_instances": req.total_instances,
        "by_class": req.by_class,
    }
    if req.record_id > 0:
        record_id = update_record(req.record_id, record, req.instances)
        return {"id": record_id, "message": "更新成功"}
    else:
        record_id = save_record(record, req.instances)
        return {"id": record_id, "message": "保存成功"}


@router.get("")
async def api_list_records(page: int = 1, size: int = 20):
    """分页查询记录列表"""
    return list_records(page=page, size=size)


@router.get("/stats/summary")
async def api_stats_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    class_name: Optional[str] = None,
):
    """聚合统计"""
    return get_stats_summary(
        start_date=start_date,
        end_date=end_date,
        class_name=class_name,
    )


@router.post("/measure")
async def api_measure(req: MeasureRequest):
    """从多边形顶点计算量化指标（与 DefectMeasurer 一致的算法）"""
    if len(req.points) < 3:
        raise HTTPException(400, "多边形至少需要3个顶点")
    if req.class_name not in ("crack", "spalling", "efflorescence"):
        raise HTTPException(400, f"不支持的类别: {req.class_name}")

    mask = _mask_from_polygon(req.points, req.image_width, req.image_height)
    if not mask.any():
        raise HTTPException(400, "多边形区域为空")

    if req.class_name == "crack":
        metrics = _measure_crack(mask, req.mm_per_px)
    else:
        metrics = _measure_spalling_efflorescence(mask, req.mm_per_px)

    return {"metrics": metrics}


@router.get("/{record_id}")
async def api_get_record(record_id: int):
    """获取单条记录详情"""
    record = get_record(record_id)
    if not record:
        raise HTTPException(404, "记录不存在")
    return record


@router.delete("/{record_id}")
async def api_delete_record(record_id: int):
    """删除记录"""
    if not delete_record(record_id):
        raise HTTPException(404, "记录不存在")
    return {"message": "删除成功"}
