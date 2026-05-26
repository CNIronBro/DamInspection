"""FastAPI 应用 — 大坝缺陷检测 Web 服务"""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline_adapter import detect_from_bytes_sync, load_pipeline, get_classes_config, IMAGE_DIR
from .database import init_db
from .api_records import router as records_router
from .api_stitch import router as stitch_router

app = FastAPI(title="大坝缺陷检测系统", version="1.0.0")

# 注册记录/统计路由
app.include_router(records_router)
app.include_router(stitch_router)

# CORS — 开发模式允许 Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- 模型管理 ---------------

MODELS = {
    "segformer-b2": {
        "id": "segformer-b2",
        "name": "SegFormer-B2",
        "description": "7类语义分割，适用于大坝表面缺陷检测",
        "config": "config.example.yml",
    },
}

_pipelines: dict = {}  # model_id -> InferencePipeline
_current_model: str = "segformer-b2"
_model_ready: bool = False


def _get_pipeline(model_id: Optional[str] = None):
    """获取指定模型的 pipeline（懒加载）"""
    global _pipelines, _current_model
    mid = model_id or _current_model
    if mid not in MODELS:
        raise HTTPException(400, f"Unknown model: {mid}")
    if mid not in _pipelines:
        cfg_path = Path(__file__).resolve().parent.parent / MODELS[mid]["config"]
        _pipelines[mid] = load_pipeline(str(cfg_path))
    return _pipelines[mid]


@app.on_event("startup")
async def startup_event():
    """启动时初始化数据库 + 预加载默认模型"""
    global _model_ready
    # 初始化 MySQL
    try:
        init_db()
        print("[INFO] MySQL database initialized")
    except Exception as e:
        print(f"[WARNING] MySQL init failed (records/stats will be unavailable): {e}")
    # 预加载模型
    try:
        _get_pipeline(_current_model)
        _model_ready = True
    except Exception as e:
        print(f"[WARNING] Failed to preload model: {e}")


# --------------- API 路由 ---------------

@app.get("/api/models")
async def get_models():
    """返回可用模型列表"""
    return {
        "models": [
            {**m, "available": (mid in _pipelines or mid == _current_model)}
            for mid, m in MODELS.items()
        ],
        "current": _current_model,
        "ready": _model_ready,
    }


@app.get("/api/config")
async def get_config():
    """返回类别定义和配置"""
    pipeline = _get_pipeline()
    return get_classes_config(pipeline)


@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
):
    """上传图片并执行缺陷检测"""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "上传文件必须是图片格式")

    image_bytes = await file.read()
    if len(image_bytes) > 50 * 1024 * 1024:
        raise HTTPException(413, "图片文件过大（最大 50MB）")

    pipeline = _get_pipeline(model_id)

    try:
        result = await asyncio.to_thread(
            detect_from_bytes_sync, pipeline, image_bytes
        )
    except Exception as e:
        raise HTTPException(500, f"推理失败: {str(e)}")

    return result


# --------------- 静态文件 & 前端 ---------------

# 图片文件服务
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")

# 生产模式：挂载 Vue 构建产物
_dist_dir = Path(__file__).resolve().parent / "frontend" / "dist"
_static_dir = Path(__file__).resolve().parent / "static"

if _dist_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_dist_dir / "assets")), name="assets")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(_dist_dir / "index.html"))

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """SPA fallback — 非 /api 路径都返回 index.html"""
        file_path = _dist_dir / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_dist_dir / "index.html"))
elif _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
