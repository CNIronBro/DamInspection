# MMSegmentation 推理与缺陷量化工具（实例级版本）

用于 SegFormer 模型的离线推理、可视化标注和缺陷量化测量的一体化工具。

## 功能特性

- ✅ 使用 MMSegmentation 官方 API 进行推理
- ✅ 支持类别重映射（7类 → 4类）
- ✅ **实例级缺陷量化**：对每个缺陷实例分别计算指标（支持一张图多个同类缺陷）
- ✅ **可配置后处理**：噪点过滤、形态学操作、孔洞填充（支持按类单独配置）
- ✅ **中文显示**：所有文字（类别名、指标名、单位）全部中文
- ✅ **CSV 表头中文化**：支持中文/英文表头切换
- ✅ **RLE 编码输出**：生成 mask 的 RLE 编码文件，供前端渲染使用
- ✅ 可视化 overlay 标注图（半透明叠加、轮廓、实例标签、信息框）
- ✅ 输出单通道 ID mask 和彩色 mask
- ✅ 批量处理支持（单张图片或目录）
- ✅ YAML 配置文件驱动，参数可调

## 安装依赖

```bash
pip install mmsegmentation mmengine mmcv opencv-python numpy scikit-image scipy pyyaml pandas tqdm Pillow scikit-learn
```

**可选依赖**（用于 COCO RLE 格式）：
```bash
pip install pycocotools
```

**注意**：
- `Pillow` 是中文文字渲染的必需依赖
- `scikit-learn` 是裂缝端点方向估计（PCA）的必需依赖
- `pycocotools` 是 COCO RLE 格式的可选依赖（未安装时自动使用 simple 格式）

**注意**：`Pillow` 是中文文字渲染的必需依赖，用于正确显示中文文字（避免出现 "????"）。

## 快速开始

### 1. 准备配置文件

复制示例配置文件并修改：

```bash
cp config.example.yml config.yml
```

编辑 `config.yml`，至少需要设置：
- `model.mmseg_config`: MMSegmentation 配置文件路径
- `model.checkpoint`: 模型权重路径
- `io.input`: 输入图片路径或目录
- `io.output_dir`: 输出目录

### 2. 运行推理

```bash
python infer_and_measure.py --cfg config.yml
```

### 3. 命令行参数覆盖（可选）

```bash
# 覆盖输入路径
python infer_and_measure.py --cfg config.yml --input /path/to/images

# 覆盖输出目录
python infer_and_measure.py --cfg config.yml --output-dir ./results

# 覆盖设备
python infer_and_measure.py --cfg config.yml --device cuda:1
```

## 输出结构

```
output/
├── vis/
│   ├── image1_overlay.png          # Overlay 标注图（含实例标签和信息框）
│   ├── image1_debug_raw7.png        # 原始 7 类 debug overlay（可选）
│   └── ...
├── mask/
│   ├── image1_id.png                # 单通道 ID mask (0-3)
│   ├── image1_color.png             # 彩色 mask
│   └── ...
└── metrics_instances.csv            # 实例级量化结果汇总（CSV）
```

## 核心改进说明

### A. 实例级划分与量化

**改进前**：每个类别整体统计（所有区域合并），无法区分多个实例。

**改进后**：
- 对每个类别进行连通域分析，提取独立实例
- 每个实例分别计算指标（面积、长度、宽度等）
- 在 overlay 图上标注每个实例（如"裂缝1"、"裂缝2"、"剥落1"等）
- 信息框显示每类前 N 个实例的详细指标

**输出格式**：
- `metrics_instances.csv`：每行一个实例，包含 `image_name`, `class_name`, `instance_id`, `area_mm2`, `length_mm` 等字段

### B. 可配置后处理

**新增功能**：
- **总开关**：`postprocess.enable` 可完全禁用后处理
- **按类配置**：`postprocess.per_class` 支持为不同类别设置不同参数
- **后处理选项**：
  - `min_area_px`：小连通域过滤（关键降噪）
  - `morph`：形态学开闭运算（去毛刺、填洞、连通断裂）
  - `hole_filling`：孔洞填充（可选）

**配置示例**：
```yaml
postprocess:
  enable: true
  min_area_px: 80
  morph:
    enable: true
    kernel: 3
    open_iter: 1
    close_iter: 1
  per_class:
    crack:
      min_area_px: 50
      morph:
        close_iter: 2  # 裂缝用更多闭运算以连通断裂线段
    spalling:
      min_area_px: 120
      morph:
        kernel: 5  # 剥落用更大的核
```

### C. 中文显示

**改进内容**：
- 类别名称：crack → 裂缝，spalling → 剥落，efflorescence → 泛碱
- 指标名称：area → 面积，length → 长度，width_mean → 平均宽度等
- 单位显示：mm²（面积）、mm（长度/宽度/直径/轴长）
- 信息框：全部中文，格式如"裂缝(2处): 裂缝1 面积=... 长度=..."

## 配置文件字段说明

### model

- `mmseg_config`: MMSegmentation 配置文件路径（必须）
- `checkpoint`: 模型权重路径（必须）
- `device`: 推理设备，`"cuda:0"` 或 `"cpu"`
- `fp16`: 是否启用半精度推理
- `show_progress`: 是否显示进度条

### io

- `input`: 输入图片路径或目录
- `output_dir`: 输出目录
- `exts`: 允许的图片后缀列表
- `save`: 保存选项
  - `overlay`: 是否保存 overlay 图
  - `id_mask`: 是否保存 ID mask
  - `color_mask`: 是否保存彩色 mask
  - `debug_raw_7class_overlay`: 是否保存原始 7 类 debug overlay
  - `csv`: 是否保存 CSV 汇总
  - `json`: 是否保存 JSON 汇总

### classes

- `original_classes`: 原始 7 类 ID 映射（训练时的类别）
- `keep_classes`: 需要保留的类别列表
- `output_classes`: 输出类别 ID（重映射后的 4 类）
- `class_remap`: 类别重映射规则（非目标类映射到 background）
- `palette`: 三类缺陷的可视化颜色（BGR 格式）
- `color_space`: 颜色空间，`"BGR"` 或 `"RGB"`

### postprocess（新增/增强）

- `enable`: **总开关**，false 则完全不做任何后处理
- `min_area_px`: 小连通域过滤阈值（像素数）
- `morph`: 形态学处理配置
  - `enable`: 是否启用
  - `kernel`: 核大小
  - `open_iter`: 开运算迭代次数
  - `close_iter`: 闭运算迭代次数
- `hole_filling`: 孔洞填充配置（可选）
  - `enable`: 是否启用
  - `max_hole_area_px`: 最大孔洞面积
- `per_class`: **按类单独配置**（可选）
  - `crack`: 裂缝专用配置
  - `spalling`: 剥落专用配置
  - `efflorescence`: 泛碱专用配置

### measurement

- `mm_per_px`: 像素到毫米的转换比例（需要根据实际图像分辨率手动填写）
- `decimals`: 数值保留小数位（默认 2）
- `compute`: 各类缺陷的测量项配置
  - `crack`: 裂缝测量项（area, length, width, width_stats）
  - `spalling`: 剥落测量项（area, eq_diameter, bbox_axes）
  - `efflorescence`: 泛碱测量项（area, eq_diameter, bbox_axes）
- `output`: 输出选项
  - `save_instance_csv`: 是否保存实例级 CSV
  - `save_instance_json`: 是否保存实例级 JSON

### visualize（新增字段）

- `alpha`: Overlay 透明度（0-1）
- `draw_contours`: 是否绘制轮廓
- `contour_thickness`: 轮廓线宽
- `font`: **字体设置（中文渲染）**（新增）
  - `font_path`: 字体文件路径（为空则自动探测常见路径）
  - `font_size`: 字体大小
  - `font_color`: 字体颜色（BGR 格式）
  - `line_spacing`: 行间距（像素）
  - `stroke`: 描边设置
    - `enable`: 是否启用描边（提高可读性）
    - `width`: 描边宽度
    - `color`: 描边颜色（BGR 格式）
- `info_box`: 信息框设置
  - `enable`: 是否启用
  - `position`: 位置（topleft/topright/bottomleft/bottomright）
  - `padding`: 内边距
  - `bg_alpha`: 背景透明度
  - `text_line_spacing`: 行间距
  - `max_instances_per_class`: **每类最多展示几个实例**（避免信息框过长）
  - `show_summary`: **是否展示类别汇总**（总面积/总长度等）
- `instance_label`: **实例标签设置**（新增）
  - `enable`: 是否在实例上绘制标签（如"裂缝1"）
  - `draw_at`: 标签位置（centroid 或 bbox_center）
  - `text_scale`: 标签文字大小（已废弃，使用 `font.font_size`）
  - `text_thickness`: 标签文字粗细（已废弃，使用 `font.stroke.width`）

## 量化指标说明

### 实例级输出格式

CSV 文件 `metrics_instances.csv` 包含以下字段：

- `image_name`: 图片名称
- `class_name`: 类别名称（英文）
- `class_name_cn`: 类别名称（中文）
- `instance_id`: 实例编号（从 1 开始）

**Crack（裂缝）实例**：
- `area_mm2`: 面积（平方毫米）
- `length_mm`: 长度（毫米，基于 skeletonize）
- `width_mean_mm`: 平均宽度（毫米）
- `width_max_mm`: 最大宽度（毫米）
- `width_p95_mm`: 95 百分位宽度（毫米）

**Spalling / Efflorescence（剥落/泛碱）实例**：
- `area_mm2`: 面积（平方毫米）
- `eq_diameter_mm`: 等效直径（毫米，sqrt(4*area/π)）
- `major_axis_mm`: 长轴（毫米，最小外接矩形）
- `minor_axis_mm`: 短轴（毫米，最小外接矩形）

### 信息框显示格式

信息框会显示每类前 N 个实例的详细指标（N 由 `visualize.info_box.max_instances_per_class` 配置），例如：

```
裂缝(2处):
  裂缝1 面积=123.45 mm² 长度=56.78 mm 平均宽度=2.34 mm 最大宽度=3.45 mm
  裂缝2 面积=98.76 mm² 长度=43.21 mm 平均宽度=2.12 mm 最大宽度=3.01 mm
  汇总: 总面积=222.21 mm², 总长度=99.99 mm
剥落(3处):
  剥落1 面积=456.78 mm² 等效直径=24.12 mm 长轴=28.90 mm 短轴=20.15 mm
  剥落2 面积=234.56 mm² 等效直径=17.28 mm 长轴=19.50 mm 短轴=15.30 mm
  剥落3 面积=345.67 mm² 等效直径=20.98 mm 长轴=23.40 mm 短轴=18.80 mm
  汇总: 总面积=1037.01 mm²
泛碱(1处):
  泛碱1 面积=567.89 mm² 等效直径=26.88 mm 长轴=30.20 mm 短轴=23.90 mm
  汇总: 总面积=567.89 mm²
```

## 注意事项

1. **尺寸对齐**: 脚本会自动处理推理输出尺寸与原图不一致的情况，使用最近邻插值将预测结果 resize 回原图尺寸。

2. **类别重映射**: 非目标类别（corrosion, vegetation, control_point）会被映射到 background，不会出现在最终输出中。

3. **像素尺度**: `measurement.mm_per_px` 需要根据实际图像分辨率手动填写，用于将像素单位转换为物理单位（毫米）。

4. **后处理开关**: 如果遇到后处理过度或不足的问题，可以通过 `postprocess.enable` 关闭，或通过 `postprocess.per_class` 为不同类别调整参数。

5. **实例数量**: 如果一张图中有很多实例，信息框可能过长。可以通过 `visualize.info_box.max_instances_per_class` 限制显示数量。

6. **设备兼容性**: 如果 CUDA 不可用，会自动回退到 CPU（但速度较慢）。

7. **内存管理**: 批量处理大图片时注意 GPU 内存，如遇 OOM 可减小 batch_size 或使用 CPU。

## 故障排查

- **模型加载失败**: 检查 `mmseg_config` 和 `checkpoint` 路径是否正确
- **推理尺寸不匹配**: 脚本会自动处理，如仍有问题检查 test_pipeline 配置
- **量化结果异常**: 检查 `mm_per_px` 是否正确设置
- **可视化颜色不对**: 检查 `color_space` 设置（BGR vs RGB）
- **实例提取失败**: 检查 `postprocess.min_area_px` 是否过大，导致所有实例被过滤
- **后处理效果不佳**: 尝试调整 `postprocess.per_class` 中的参数，或关闭后处理查看原始结果
- **中文显示为 "????"**: 
  - 确保已安装 `Pillow`: `pip install Pillow`
  - 检查 `visualize.font.font_path` 配置是否正确
  - 如果为空，确保系统安装了中文字体（如 NotoSansCJK）
  - 可以将字体文件放到项目 `fonts/` 目录下
  - 查看控制台错误信息，按照提示配置字体路径

## 中文字体配置

**重要**：为了正确显示中文文字（避免出现 "????"），需要配置中文字体。

### 方法 1：自动探测（推荐）

如果 `visualize.font.font_path` 为空，程序会自动尝试以下常见路径：

**Linux**：
- `/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc`
- `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
- `/usr/share/fonts/truetype/wqy/wqy-microhei.ttc`
- `/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc`

**Windows**：
- `C:/Windows/Fonts/simhei.ttf`（黑体）
- `C:/Windows/Fonts/msyh.ttc`（微软雅黑）
- `C:/Windows/Fonts/simsun.ttc`（宋体）

**macOS**：
- `/System/Library/Fonts/PingFang.ttc`
- `/System/Library/Fonts/STHeiti Light.ttc`

**项目本地**：
- `./fonts/NotoSansCJK-Regular.ttc`
- `./fonts/simhei.ttf`
- `./fonts/msyh.ttc`

### 方法 2：手动配置

在 `config.yml` 中设置 `visualize.font.font_path`：

```yaml
visualize:
  font:
    font_path: "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
```

### 方法 3：使用项目本地字体

将字体文件放到项目根目录的 `fonts/` 目录下，程序会自动探测。

### 字体缺失处理

如果程序找不到任何中文字体，会：
1. 在控制台输出明确的错误信息
2. 提示用户配置 `font_path`
3. 程序直接退出（不会生成包含 "????" 的图像）

**安装字体（Linux）**：

```bash
# Ubuntu/Debian
sudo apt-get install fonts-noto-cjk

# CentOS/RHEL
sudo yum install google-noto-cjk-fonts
```

## 裂缝断裂实例合并（端点桥接）

### 问题背景

由于语义分割模型的预测结果可能存在断裂，一条完整的裂缝可能被分割成多个不连通的区域，导致实例提取时被识别为多个独立的实例。这会影响实例级量化的准确性。

### 解决方案

本工具实现了**端点桥接（Endpoint Bridging）**算法，专门用于裂缝实例的智能合并：

1. **去噪**：先移除小噪点，避免噪点参与桥接
2. **轻量形态学**：轻微 close 操作，连接非常小的缺口
3. **骨架化**：提取裂缝的中心线（skeleton）
4. **端点检测**：识别 skeleton 的端点（8邻域度为1的点）
5. **方向估计**：通过 PCA 估计每个端点的方向向量
6. **智能配对**：基于距离和方向一致性进行端点配对
7. **桥接绘制**：在配对的端点之间绘制连接线
8. **实例划分**：对桥接后的 mask 进行连通域分析

### 配置参数

在 `config.yml` 的 `postprocess.per_class.crack.bridge` 中配置：

```yaml
postprocess:
  per_class:
    crack:
      bridge:
        enable: true             # 总开关
        max_gap_px: 30            # 最大端点连接距离（像素）
        max_angle_deg: 25         # 方向一致性阈值（度）
        line_thickness: 1         # 连接线粗细（1~2 推荐）
        max_pairs_per_endpoint: 1 # 每个端点最多连接几个
        prefer_nearest: true      # 优先连接最近端点
        min_component_length_px: 20  # 太短的裂缝段不参与桥接
        debug_save: true          # 是否保存debug可视化
        safe_check:               # 安全检查（可选）
          enable: false
          sample_points: 20
          max_bg_ratio: 0.7
```

### 关键参数说明

#### `max_gap_px`（最大端点连接距离）

- **含义**：两个端点之间的距离必须小于等于此值才会被考虑连接
- **调参建议**：
  - 太小（< 20）：可能无法连接较远的断裂
  - 太大（> 50）：可能误连接不相关的裂缝
  - **推荐值**：20-40 像素，根据图像分辨率和裂缝断裂程度调整

#### `max_angle_deg`（方向一致性阈值）

- **含义**：两个端点的方向向量夹角必须小于等于此值才会被连接
- **调参建议**：
  - 太小（< 15°）：只能连接方向非常一致的端点
  - 太大（> 45°）：可能误连接垂直或反向的端点
  - **推荐值**：20-30°，根据裂缝的弯曲程度调整

#### `line_thickness`（连接线粗细）

- **含义**：桥接线的像素宽度
- **调参建议**：
  - 1-2 像素通常足够
  - 太粗可能影响后续的宽度测量

#### `min_component_length_px`（最小组件长度）

- **含义**：小于此长度的裂缝段不参与桥接（避免噪点干扰）
- **调参建议**：
  - 10-30 像素，根据图像分辨率调整

### Debug 可视化

启用 `bridge.debug_save: true` 和 `visualize.debug.save_bridge_debug: true` 后，会在输出目录的 `debug_bridge/` 子目录下生成可视化图像：

- **红色**：原始裂缝 mask
- **白色**：Skeleton（中心线）
- **蓝色圆点**：检测到的端点
- **青色线条**：桥接连接线

通过查看 debug 图像，可以直观地了解桥接效果，便于调参。

### 调参流程

1. **初始设置**：使用默认参数（`max_gap_px=30`, `max_angle_deg=25`）
2. **查看 debug 图像**：检查是否有断裂未被连接，或是否有误连接
3. **调整距离阈值**：
   - 如果断裂较远未被连接：适当增大 `max_gap_px`
   - 如果误连接了不相关的裂缝：减小 `max_gap_px`
4. **调整角度阈值**：
   - 如果弯曲裂缝未被连接：适当增大 `max_angle_deg`
   - 如果误连接了垂直裂缝：减小 `max_angle_deg`
5. **迭代优化**：根据实际效果反复调整，直到满意

### 注意事项

- **仅对 crack 生效**：桥接功能只应用于裂缝类别，spalling 和 efflorescence 仍使用常规连通域分析
- **性能考虑**：端点数量过多时，算法会自动使用 KDTree 加速距离查询
- **安全检查**：可选启用 `safe_check`，防止桥接线跨越大面积背景区域

## 信息面板可视化（Panel 版）

### 功能说明

除了标准的 overlay 图（`out/vis/xxx_overlay.png`），程序还可以生成带信息面板的可视化图，完整展示所有实例的量化数据，避免信息框空间不足的问题。

### 输出位置

Panel 版图片保存在 `out/vis_panel/` 目录下（目录名可在配置中修改）。

### 配置方法

在 `config.yml` 中设置：

```yaml
visualize:
  panel:
    enable: true                # 是否生成 panel 版
    output_dir: "vis_panel"      # 输出目录名
    placement: "left"            # "left"（左侧）或 "bottom"（底部）
    width_px: 520               # 左侧面板宽度
    height_px: 300              # 底部面板高度
    bg_color: [30, 30, 30]      # 面板背景颜色（BGR）
    padding: 16                 # 内边距
    line_spacing: 8             # 行间距
    auto_shrink: true           # 自动缩小字体适配
    paginate:
      enable: true              # 启用分页
      lines_per_page: 35        # 每页最大行数
```

### 面板内容

Panel 版包含：
- **图片名**
- **三类缺陷的完整列表**（裂缝/剥落/泛碱）
- **每个实例的详细指标**：
  - 裂缝：面积、长度、平均宽度、最大宽度、P95宽度
  - 剥落/泛碱：面积、等效直径、长轴、短轴
- **每类汇总**：总面积、总长度（裂缝）

### 分页处理

当实例数量过多，单页无法容纳时：
- 如果 `paginate.enable=true`：自动分页，输出多张图片（`xxx_panel_p1.png`, `xxx_panel_p2.png`, ...）
- 如果 `auto_shrink=true`：自动缩小字体，直到可容纳或达到最小字号阈值

### 与标准 Overlay 的区别

- **标准 Overlay**：在原图上叠加信息框，最多显示每类 top-3 个实例
- **Panel 版**：扩展画布，将原图和信息面板拼接，显示所有实例的完整数据

**注意**：Panel 版是额外输出，不影响现有的 `out/vis/xxx_overlay.png`。

## 实例标签显示优化

### 功能说明

默认情况下，实例编号（如"裂缝1"）不再绘制白色背景块，仅保留文字和描边，减少对原图的遮挡。

### 配置方法

在 `config.yml` 中设置：

```yaml
visualize:
  instance_label:
    draw_bg: false    # false：不绘制背景（默认，仅文字+描边）
                      # true：绘制白色背景（旧行为）
```

### 效果对比

- **draw_bg=false**（默认）：文字 + 描边，不遮挡原图
- **draw_bg=true**：文字 + 白色背景块（旧行为）

**注意**：此设置仅影响实例编号标签，不影响信息框（信息框仍使用半透明背景以提高可读性）。

## 许可证

与 MMSegmentation 保持一致。
