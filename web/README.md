# 大坝缺陷检测 Web 服务 — 本地启动指南

## 前置条件

| 项目 | 要求 |
|------|------|
| Python | 3.9 |
| CUDA | 11.8（有 GPU 时） |
| Node.js | >= 18（[下载地址](https://nodejs.org/)） |
| 模型权重 | 训练好的 `.pth` 文件（如 `best_mIoU_iter_38000.pth`） |

---

## Step 1：配置模型权重路径

编辑 `config.example.yml`，将 `checkpoint` 改为本地权重的实际路径：

```yaml
model:
  mmseg_config: "config/segformer_mit-b2_1xb8-40k_s2dsdeepcrack-512x512.py"
  checkpoint: "E:/your/path/to/best_mIoU_iter_38000.pth"   # ← 改成你的本地路径
  device: "cuda:0"   # 无 GPU 改为 "cpu"
```

> 路径可以用绝对路径或相对于项目根目录的相对路径。

---

## Step 2：创建 Python 环境并安装依赖

```bash
# 创建虚拟环境
conda create --prefix ./venv python=3.9 -y
conda activate ./venv

# PyTorch（必须 2.1.2 + cu118）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# OpenMMLab
pip install "mmengine>=0.5.0,<1.0.0"
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# mmsegmentation + 自定义文件覆盖
pip install mmsegmentation==1.2.2
cp -r mmsegmentation/mmseg/* ./venv/lib/python3.9/site-packages/mmseg/

# 推理依赖
pip install "numpy<2" opencv-python scikit-image scipy pyyaml pandas tqdm Pillow scikit-learn ftfy

# Web 依赖
pip install -r requirements-web.txt
```

> **`numpy<2` 是必须的**，mmcv 2.1.0 与 NumPy 2.x 不兼容。
> **`ftfy` 是必须的**，mmsegmentation 的 tokenizer 模块依赖它。

---

## Step 3：安装前端依赖

```bash
cd web/frontend
npm install
cd ../..
```

---

## Step 4：启动服务

需要 **两个终端**：

### 终端 1 — 后端（FastAPI）

```bash
conda activate ./venv   # 或你的虚拟环境
python run_web.py
```

看到 `Model loaded from ...` 和 `Uvicorn running on http://0.0.0.0:8000` 即启动成功。
首次启动会加载模型（约 5-15 秒）。

### 终端 2 — 前端（Vite）

```bash
cd web/frontend
npm run dev
```

看到 `Local: http://localhost:5173/` 即启动成功。

---

## Step 5：使用

浏览器打开 **http://localhost:5173**

1. 左侧侧边栏查看功能导航
2. 顶部工具栏选择模型、上传图片
3. 等待检测完成，查看 Canvas overlay 渲染和右侧量化指标

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'mmseg.models.decode_heads.decode_head'` | 自定义 mmseg 目录不完整 | 重新执行 `pip install mmsegmentation==1.2.2` + `cp -r` 覆盖 |
| `_ARRAY_API not found` | NumPy 2.x 不兼容 | `pip install "numpy<2"` |
| `No module named 'ftfy'` | 缺少依赖 | `pip install ftfy` |
| `Config file not found` | 配置路径错误 | 检查 `config.example.yml` 中的 `mmseg_config` 路径 |
| `Checkpoint not found` | 权重路径错误 | 检查 `config.example.yml` 中的 `checkpoint` 路径 |
| `npm: command not found` | 未安装 Node.js | 安装 Node.js >= 18 |
| 后端启动后前端请求 500 | 模型加载失败 | 检查终端 1 的报错信息 |
| 端口 8000 被占用 | 其他进程占用 | `lsof -i :8000` 查找并关闭，或修改 `run_web.py` 中的端口 |

---

## 文件结构

```
DamInspection/
├── config.example.yml          # 推理配置（需修改 checkpoint 路径）
├── config/                     # MMSeg 训练配置
├── infer_and_measure.py        # 推理引擎（不修改）
├── run_web.py                  # 后端启动入口
├── requirements-web.txt        # 后端 Python 依赖
├── web/
│   ├── app.py                  # FastAPI 应用
│   ├── pipeline_adapter.py     # 推理适配器
│   └── frontend/               # Vue 3 前端
│       ├── package.json
│       ├── vite.config.js
│       └── src/
│           ├── main.js
│           ├── App.vue
│           └── components/
└── mmsegmentation/             # 自定义 mmseg（不修改）
```
