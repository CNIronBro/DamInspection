#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMSegmentation 推理与缺陷量化脚本（实例级版本）
支持类别重映射、实例分割、可视化、量化测量
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import yaml
import json
import pandas as pd
from tqdm import tqdm
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import torch

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL (Pillow) not found. Chinese text rendering will not work.")

try:
    import pycocotools.mask as mask_util
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("Warning: pycocotools not found. RLE encoding will use simple format.")

try:
    from mmseg.apis import init_model, inference_model
    from mmseg.utils import register_all_modules
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
except ImportError as e:
    print(f"Error: Missing required package. Please install mmsegmentation, mmengine, mmcv.")
    print(f"Import error: {e}")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL (Pillow) not found. Chinese text rendering will not work.")


class ChineseTextRenderer:
    """中文文字渲染器（使用 PIL）"""
    
    # 常见中文字体路径（自动探测）
    COMMON_FONT_PATHS = [
        # Linux 常见路径
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        # Windows 常见路径
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
        'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
        'C:/Windows/Fonts/msyhbd.ttc',  # 微软雅黑 Bold
        # macOS 常见路径
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        # 项目本地字体目录
        './fonts/NotoSansCJK-Regular.ttc',
        './fonts/simhei.ttf',
        './fonts/msyh.ttc',
    ]
    
    def __init__(self, font_path: str = "", font_size: int = 26, 
                 font_color: List[int] = [255, 255, 255],
                 line_spacing: int = 6,
                 stroke_enable: bool = True,
                 stroke_width: int = 2,
                 stroke_color: List[int] = [0, 0, 0],
                 color_space: str = "BGR"):
        """
        初始化中文文字渲染器
        
        Args:
            font_path: 字体文件路径，为空则自动探测
            font_size: 字体大小
            font_color: 字体颜色 [B, G, R] 或 [R, G, B]
            line_spacing: 行间距
            stroke_enable: 是否启用描边
            stroke_width: 描边宽度
            stroke_color: 描边颜色
            color_space: 颜色空间 "BGR" 或 "RGB"
        """
        if not HAS_PIL:
            raise ImportError("PIL (Pillow) is required for Chinese text rendering. "
                            "Please install: pip install Pillow")
        
        self.font_size = font_size
        self.font_color = font_color
        self.line_spacing = line_spacing
        self.stroke_enable = stroke_enable
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color
        self.color_space = color_space
        
        # 加载字体
        self.font = self._load_font(font_path)
    
    def _load_font(self, font_path: str) -> ImageFont.FreeTypeFont:
        """加载字体文件"""
        # 如果指定了路径，直接使用
        if font_path and os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, self.font_size)
            except Exception as e:
                print(f"Warning: Failed to load font from {font_path}: {e}")
        
        # 自动探测
        for path in self.COMMON_FONT_PATHS:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, self.font_size)
                except Exception as e:
                    continue
        
        # 如果都找不到，报错退出
        print("\n" + "="*60)
        print("错误：未找到中文字体文件！")
        print("="*60)
        print("请执行以下操作之一：")
        print("1. 在配置文件中设置 visualize.font_path 指向字体文件路径")
        print("2. 将字体文件放到项目 fonts/ 目录下")
        print("\n常见字体路径示例（Linux）：")
        print("  /usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc")
        print("  /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
        print("\n常见字体路径示例（Windows）：")
        print("  C:/Windows/Fonts/simhei.ttf")
        print("  C:/Windows/Fonts/msyh.ttc")
        print("="*60 + "\n")
        sys.exit(1)
    
    def _bgr_to_rgb(self, color: List[int]) -> Tuple[int, int, int]:
        """BGR 转 RGB"""
        if self.color_space == "BGR":
            return tuple(color[::-1])
        return tuple(color)
    
    def get_text_size(self, text: str) -> Tuple[int, int]:
        """获取文本尺寸（支持多行）"""
        lines = text.split('\n')
        max_w = 0
        total_h = 0
        
        for i, line in enumerate(lines):
            if not line.strip():
                bbox = self.font.getbbox("A")  # 使用占位符
            else:
                bbox = self.font.getbbox(line)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            max_w = max(max_w, w)
            if i == 0:
                total_h = h
            else:
                total_h += self.line_spacing + h
        
        return (max_w, total_h)
    
    def draw_text(self, img: np.ndarray, text: str, pos: Tuple[int, int],
                  bg_color: Optional[List[int]] = None,
                  bg_padding: int = 0) -> np.ndarray:
        """
        在图像上绘制中文文本
        
        Args:
            img: BGR 格式的 OpenCV 图像
            text: 文本内容（支持多行，用 \n 分隔）
            pos: 左上角坐标 (x, y)
            bg_color: 背景颜色 [B, G, R] 或 [R, G, B]，None 则不绘制背景
            bg_padding: 背景内边距
        
        Returns:
            绘制后的 BGR 图像
        """
        # 转换 BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # 颜色转换
        text_color_rgb = self._bgr_to_rgb(self.font_color)
        stroke_color_rgb = self._bgr_to_rgb(self.stroke_color) if self.stroke_enable else None
        
        # 绘制背景（如果需要）
        if bg_color is not None:
            bg_color_rgb = self._bgr_to_rgb(bg_color)
            text_w, text_h = self.get_text_size(text)
            x, y = pos
            bg_rect = [
                x - bg_padding,
                y - bg_padding,
                x + text_w + bg_padding,
                y + text_h + bg_padding
            ]
            draw.rectangle(bg_rect, fill=bg_color_rgb)
        
        # 绘制文本（支持多行）
        lines = text.split('\n')
        x, y = pos
        
        for line in lines:
            if not line.strip():
                y += self.font_size + self.line_spacing
                continue
            
            # 获取当前行尺寸
            bbox = self.font.getbbox(line)
            line_h = bbox[3] - bbox[1]
            
            # 绘制描边（如果启用）
            if self.stroke_enable and self.stroke_width > 0:
                # 检查 PIL 版本是否支持 stroke_width
                try:
                    draw.text((x, y), line, font=self.font, fill=text_color_rgb,
                            stroke_width=self.stroke_width, stroke_fill=stroke_color_rgb)
                except TypeError:
                    # 旧版本不支持 stroke_width，手动模拟描边
                    offsets = [
                        (-self.stroke_width, -self.stroke_width),
                        (-self.stroke_width, 0),
                        (-self.stroke_width, self.stroke_width),
                        (0, -self.stroke_width),
                        (0, self.stroke_width),
                        (self.stroke_width, -self.stroke_width),
                        (self.stroke_width, 0),
                        (self.stroke_width, self.stroke_width),
                    ]
                    for dx, dy in offsets:
                        draw.text((x + dx, y + dy), line, font=self.font, fill=stroke_color_rgb)
                    draw.text((x, y), line, font=self.font, fill=text_color_rgb)
            else:
                draw.text((x, y), line, font=self.font, fill=text_color_rgb)
            
            y += line_h + self.line_spacing
        
        # 转换回 BGR
        img_bgr = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        
        return img_bgr


class Instance(NamedTuple):
    """实例数据结构"""
    instance_id: int
    mask: np.ndarray
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


class DefectMeasurer:
    """缺陷量化测量器（实例级）"""
    
    # 中文类别名称映射
    CLASS_NAMES_CN = {
        'crack': '裂缝',
        'spalling': '剥落',
        'efflorescence': '泛碱',
    }
    
    # 中文指标名称映射
    METRIC_NAMES_CN = {
        'area': '面积',
        'length': '长度',
        'width_mean': '平均宽度',
        'width_max': '最大宽度',
        'width_p95': 'P95宽度',
        'eq_diameter': '等效直径',
        'major_axis': '长轴',
        'minor_axis': '短轴',
    }
    
    def __init__(self, config: dict):
        self.config = config
        self.mm_per_px = config['measurement']['mm_per_px']
        self.compute_cfg = config['measurement']['compute']
        self.decimals = config['measurement'].get('decimals', 2)
        
    def _get_postprocess_cfg(self, class_name: str) -> dict:
        """获取后处理配置（支持 per_class 覆盖）"""
        post_cfg = self.config['postprocess']
        if not post_cfg.get('enable', True):
            return {'enable': False}
        
        # 全局配置
        cfg = {
            'enable': True,
            'min_area_px': post_cfg.get('min_area_px', 50),
            'morph': post_cfg.get('morph', {}),
            'hole_filling': post_cfg.get('hole_filling', {}),
            'denoise': post_cfg.get('denoise', {}),
            'bridge': post_cfg.get('bridge', {}),
        }
        
        # 按类覆盖
        per_class = post_cfg.get('per_class', {})
        if class_name in per_class:
            class_cfg = per_class[class_name]
            if 'min_area_px' in class_cfg:
                cfg['min_area_px'] = class_cfg['min_area_px']
            if 'morph' in class_cfg:
                cfg['morph'] = {**cfg.get('morph', {}), **class_cfg['morph']}
            if 'hole_filling' in class_cfg:
                cfg['hole_filling'] = {**cfg.get('hole_filling', {}), **class_cfg['hole_filling']}
            if 'denoise' in class_cfg:
                cfg['denoise'] = {**cfg.get('denoise', {}), **class_cfg['denoise']}
            if 'bridge' in class_cfg:
                cfg['bridge'] = {**cfg.get('bridge', {}), **class_cfg['bridge']}
        
        return cfg
    
    def _apply_postprocess(self, mask: np.ndarray, class_name: str) -> np.ndarray:
        """应用后处理"""
        cfg = self._get_postprocess_cfg(class_name)
        if not cfg.get('enable', True):
            return mask
        
        result = mask.copy().astype(np.uint8)
        
        # 形态学处理
        morph = cfg.get('morph', {})
        if morph.get('enable', True):
            kernel_size = morph.get('kernel', 3)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
            if morph.get('open_iter', 0) > 0:
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, 
                                         kernel, iterations=morph['open_iter'])
            if morph.get('close_iter', 0) > 0:
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, 
                                         kernel, iterations=morph['close_iter'])
        
        # 孔洞填充
        hole_filling = cfg.get('hole_filling', {})
        if hole_filling.get('enable', False):
            max_hole_area = hole_filling.get('max_hole_area_px', 50)
            # 使用 findContours 找到所有孔洞并填充
            h, w = result.shape
            mask_filled = result.copy()
            # 创建边界，确保背景连通
            mask_with_border = np.zeros((h + 2, w + 2), dtype=np.uint8)
            mask_with_border[1:-1, 1:-1] = result
            
            # 找到所有轮廓（包括孔洞）
            contours, hierarchy = cv2.findContours(
                mask_with_border, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if hierarchy is not None:
                for i, h in enumerate(hierarchy[0]):
                    # 如果是孔洞（有父轮廓）
                    if h[3] >= 0:
                        area = cv2.contourArea(contours[i])
                        if area <= max_hole_area:
                            # 填充孔洞
                            cv2.drawContours(mask_with_border, [contours[i]], -1, 255, -1)
            
            result = mask_with_border[1:-1, 1:-1].astype(bool)
        
        return result.astype(bool)
    
    def _denoise_crack_mask(self, mask: np.ndarray, cfg: dict) -> np.ndarray:
        """对裂缝 mask 进行去噪"""
        denoise_cfg = cfg.get('denoise', {})
        if not denoise_cfg.get('enable', True):
            return mask
        
        min_area = denoise_cfg.get('min_area_px', 80)
        if min_area <= 0:
            return mask
        
        # 移除小连通域
        labeled = label(mask > 0)
        result = mask.copy()
        for region in regionprops(labeled):
            if region.area < min_area:
                result[labeled == region.label] = 0
        
        return result.astype(bool)
    
    def _apply_crack_morphology(self, mask: np.ndarray, cfg: dict) -> np.ndarray:
        """对裂缝 mask 应用轻量形态学处理"""
        morph_cfg = cfg.get('morph', {})
        if not morph_cfg.get('enable', True):
            return mask
        
        result = mask.copy().astype(np.uint8)
        
        # Close（连接小缺口）
        close_cfg = morph_cfg.get('close', {})
        if close_cfg.get('kernel', 0) > 0 and close_cfg.get('iter', 0) > 0:
            kernel = np.ones((close_cfg['kernel'], close_cfg['kernel']), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, 
                                     kernel, iterations=close_cfg['iter'])
        
        # Open（去毛刺，可选）
        open_cfg = morph_cfg.get('open', {})
        if open_cfg.get('kernel', 0) > 0 and open_cfg.get('iter', 0) > 0:
            kernel = np.ones((open_cfg['kernel'], open_cfg['kernel']), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, 
                                     kernel, iterations=open_cfg['iter'])
        
        return result.astype(bool)
    
    def _detect_skeleton_endpoints(self, skeleton: np.ndarray) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
        """检测 skeleton 的端点并估计方向"""
        h, w = skeleton.shape
        endpoints = []
        directions = []
        
        # 8邻域偏移
        neighbors_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # 找到所有 skeleton 点
        skel_points = np.column_stack(np.where(skeleton > 0))
        
        if len(skel_points) == 0:
            return endpoints, directions
        
        # 构建 KDTree 用于快速查找邻居
        tree = cKDTree(skel_points)
        
        for idx, (y, x) in enumerate(skel_points):
            # 计算8邻域中 skeleton 邻居数
            neighbor_count = 0
            neighbor_coords = []
            
            for dy, dx in neighbors_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0:
                    neighbor_count += 1
                    neighbor_coords.append((ny, nx))
            
            # 端点是邻居数为1的点
            if neighbor_count == 1:
                endpoints.append((x, y))  # OpenCV 使用 (x, y)
                
                # 估计方向：沿 skeleton 回溯 L 个点做 PCA
                direction = self._estimate_endpoint_direction(
                    (y, x), skel_points, tree, backtrack_length=10
                )
                directions.append(direction)
        
        return endpoints, directions
    
    def _estimate_endpoint_direction(self, point: Tuple[int, int], 
                                    skel_points: np.ndarray, 
                                    tree: cKDTree, 
                                    backtrack_length: int = 10) -> np.ndarray:
        """估计端点方向（通过回溯 skeleton 点做 PCA）"""
        py, px = point
        
        # 找到最近的邻居（沿 skeleton 的方向）
        dists, indices = tree.query([point], k=min(backtrack_length + 1, len(skel_points)))
        if len(indices[0]) < 2:
            return np.array([1.0, 0.0])  # 默认方向
        
        # 获取回溯点（排除自身）
        backtrack_points = skel_points[indices[0][1:min(backtrack_length + 1, len(indices[0]))]]
        
        if len(backtrack_points) < 2:
            # 如果点太少，直接用最近邻的方向
            if len(backtrack_points) == 0:
                return np.array([1.0, 0.0])  # 默认方向
            nearest = backtrack_points[0]
            direction = np.array([nearest[1] - px, nearest[0] - py], dtype=float)
        else:
            # 使用 PCA 估计主方向
            try:
                pca = PCA(n_components=1)
                pca.fit(backtrack_points[:, ::-1])  # 转为 (x, y) 格式
                direction = pca.components_[0]
            except:
                # PCA 失败，使用最近邻方向
                nearest = backtrack_points[0]
                direction = np.array([nearest[1] - px, nearest[0] - py], dtype=float)
        
        # 归一化
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        else:
            direction = np.array([1.0, 0.0])
        
        return direction
    
    def _bridge_crack_endpoints(self, mask: np.ndarray, skeleton: np.ndarray,
                                endpoints: List[Tuple[int, int]], 
                                directions: List[np.ndarray],
                                cfg: dict) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """桥接裂缝端点"""
        bridge_cfg = cfg.get('bridge', {})
        if not bridge_cfg.get('enable', False):
            return mask, []
        
        max_gap = bridge_cfg.get('max_gap_px', 30)
        max_angle_deg = bridge_cfg.get('max_angle_deg', 25)
        line_thickness = bridge_cfg.get('line_thickness', 1)
        max_pairs = bridge_cfg.get('max_pairs_per_endpoint', 1)
        prefer_nearest = bridge_cfg.get('prefer_nearest', True)
        min_length = bridge_cfg.get('min_component_length_px', 20)
        safe_check = bridge_cfg.get('safe_check', {})
        
        if len(endpoints) < 2:
            return mask, []
        
        # 过滤太短的组件（可选）
        if min_length > 0:
            labeled = label(skeleton > 0)
            valid_endpoints = []
            valid_directions = []
            for i, (x, y) in enumerate(endpoints):
                component_id = labeled[y, x]
                if component_id > 0:
                    component_size = (labeled == component_id).sum()
                    if component_size >= min_length:
                        valid_endpoints.append((x, y))
                        valid_directions.append(directions[i])
            endpoints = valid_endpoints
            directions = valid_directions
        
        if len(endpoints) < 2:
            return mask, []
        
        # 构建 KDTree 加速距离查询
        endpoint_array = np.array(endpoints)
        tree = cKDTree(endpoint_array)
        
        # 构建候选对
        max_angle_rad = np.deg2rad(max_angle_deg)
        candidate_pairs = []
        
        for i, (p1, d1) in enumerate(zip(endpoints, directions)):
            # 查找距离内的端点
            dists, indices = tree.query([p1], k=min(len(endpoints), 20))
            dists = dists[0]
            indices = indices[0]
            
            for j, (dist, idx) in enumerate(zip(dists, indices)):
                if i >= idx or dist > max_gap or dist < 1:
                    continue
                
                p2 = endpoints[idx]
                d2 = directions[idx]
                
                # 计算方向夹角
                dot_product = abs(np.dot(d1, d2))
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                
                if angle <= max_angle_rad:
                    # 安全检查（可选）
                    if safe_check.get('enable', False):
                        if not self._safe_bridge_check(p1, p2, mask, safe_check):
                            continue
                    
                    candidate_pairs.append((i, idx, dist, p1, p2))
        
        # 贪心匹配
        matched = set()
        bridge_pairs = []
        
        if prefer_nearest:
            candidate_pairs.sort(key=lambda x: x[2])  # 按距离排序
        
        for i, idx, dist, p1, p2 in candidate_pairs:
            if i in matched or idx in matched:
                continue
            
            # 检查端点匹配次数限制
            i_count = sum(1 for pair in bridge_pairs if pair[0] == p1 or pair[1] == p1)
            idx_count = sum(1 for pair in bridge_pairs if pair[0] == p2 or pair[1] == p2)
            
            if i_count >= max_pairs or idx_count >= max_pairs:
                continue
            
            bridge_pairs.append((p1, p2))
            matched.add(i)
            matched.add(idx)
        
        # 绘制桥接线
        bridge_canvas = np.zeros_like(mask, dtype=np.uint8)
        for p1, p2 in bridge_pairs:
            cv2.line(bridge_canvas, p1, p2, 1, thickness=line_thickness)
        
        # 融合到 mask
        result = mask.copy().astype(bool) | (bridge_canvas > 0)
        
        # 可选：再做一次轻微 close
        morph_cfg = cfg.get('morph', {})
        close_cfg = morph_cfg.get('close', {})
        if close_cfg.get('kernel', 0) > 0 and close_cfg.get('iter', 0) > 0:
            kernel = np.ones((close_cfg['kernel'], close_cfg['kernel']), np.uint8)
            result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, 
                                     kernel, iterations=1)
            result = result.astype(bool)
        
        return result, bridge_pairs
    
    def _safe_bridge_check(self, p1: Tuple[int, int], p2: Tuple[int, int],
                          mask: np.ndarray, safe_cfg: dict) -> bool:
        """安全检查：防止跨越大面积背景"""
        sample_points = safe_cfg.get('sample_points', 20)
        max_bg_ratio = safe_cfg.get('max_bg_ratio', 0.7)
        
        # 在 p1 和 p2 之间采样点
        x1, y1 = p1
        x2, y2 = p2
        
        bg_count = 0
        for i in range(sample_points + 1):
            t = i / sample_points
            x = int(x1 * (1 - t) + x2 * t)
            y = int(y1 * (1 - t) + y2 * t)
            
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                if not mask[y, x]:
                    bg_count += 1
        
        bg_ratio = bg_count / (sample_points + 1)
        return bg_ratio <= max_bg_ratio
    
    def _visualize_bridge_debug(self, mask_morphed: np.ndarray, skeleton: np.ndarray,
                               endpoints: List[Tuple[int, int]],
                               bridge_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                               output_path: Path):
        """可视化桥接过程（debug）"""
        h, w = mask_morphed.shape
        
        # 创建三通道图像
        debug_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 原始 mask（红色）
        debug_img[mask_morphed > 0] = [0, 0, 255]  # BGR: 红色
        
        # Skeleton（白色）
        debug_img[skeleton > 0] = [255, 255, 255]
        
        # 端点（蓝色圆点）
        for x, y in endpoints:
            cv2.circle(debug_img, (x, y), 3, (255, 0, 0), -1)  # BGR: 蓝色
        
        # 桥接线（青色）
        for p1, p2 in bridge_pairs:
            cv2.line(debug_img, p1, p2, (255, 255, 0), 2)  # BGR: 青色
        
        # 保存
        cv2.imwrite(str(output_path), debug_img)
    
    def extract_instances(self, mask: np.ndarray, class_name: str, 
                         debug_output_dir: Optional[Path] = None,
                         debug_img_name: Optional[str] = None) -> List[Instance]:
        """提取实例（连通域分析），对 crack 特殊处理（端点桥接）"""
        if mask.sum() == 0:
            return []
        
        cfg = self._get_postprocess_cfg(class_name)
        
        # 对 crack 进行特殊处理（桥接）
        if class_name == 'crack' and cfg.get('bridge', {}).get('enable', False):
            # 1. 去噪
            mask_denoised = self._denoise_crack_mask(mask, cfg)
            
            # 2. 轻量形态学
            mask_morphed = self._apply_crack_morphology(mask_denoised, cfg)
            
            # 3. Skeletonize
            skeleton = skeletonize(mask_morphed).astype(np.uint8)
            
            # 4. 检测端点
            endpoints, directions = self._detect_skeleton_endpoints(skeleton)
            
            # 5. 桥接端点
            mask_bridged, bridge_pairs = self._bridge_crack_endpoints(
                mask_morphed, skeleton, endpoints, directions, cfg
            )
            
            # 6. Debug 可视化
            bridge_cfg = cfg.get('bridge', {})
            if bridge_cfg.get('debug_save', False) and debug_output_dir and debug_img_name:
                debug_dir = debug_output_dir / bridge_cfg.get('debug_dirname', 'debug_bridge')
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_path = debug_dir / f'{debug_img_name}_bridge.png'
                self._visualize_bridge_debug(
                    mask_morphed, skeleton, endpoints, bridge_pairs, debug_path
                )
            
            processed_mask = mask_bridged
        else:
            # 非 crack 或桥接未启用：使用常规后处理
            processed_mask = self._apply_postprocess(mask, class_name)
        
        if processed_mask.sum() == 0:
            return []
        
        # 连通域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            processed_mask.astype(np.uint8), connectivity=8
        )
        
        # 过滤小连通域
        min_area = cfg.get('min_area_px', 50)
        instances = []
        
        for i in range(1, num_labels):  # 跳过背景 (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            
            instance_mask = (labels == i)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            centroid = (int(centroids[i, 0]), int(centroids[i, 1]))
            
            instances.append(Instance(
                instance_id=len(instances) + 1,
                mask=instance_mask,
                centroid=centroid,
                bbox=(x, y, w, h)
            ))
        
        return instances
    
    def measure_crack_instance(self, instance: Instance) -> Dict:
        """测量单个 crack 实例"""
        results = {
            'area_mm2': 0.0,
            'length_mm': 0.0,
            'width_mean_mm': 0.0,
            'width_max_mm': 0.0,
            'width_p95_mm': 0.0,
        }
        
        mask = instance.mask
        if mask.sum() == 0:
            return results
        
        # Area
        if self.compute_cfg['crack']['area']:
            results['area_mm2'] = round(mask.sum() * (self.mm_per_px ** 2), self.decimals)
        
        # Length (skeletonize)
        if self.compute_cfg['crack']['length']:
            skeleton = skeletonize(mask).astype(np.uint8)
            results['length_mm'] = round(skeleton.sum() * self.mm_per_px, self.decimals)
        
        # Width (distance transform)
        if self.compute_cfg['crack']['width']:
            dist = distance_transform_edt(mask)
            skeleton = skeletonize(mask)
            widths = 2 * dist[skeleton > 0] * self.mm_per_px
            
            if len(widths) > 0:
                results['width_mean_mm'] = round(float(np.mean(widths)), self.decimals)
                results['width_max_mm'] = round(float(np.max(widths)), self.decimals)
                if 'p95' in self.compute_cfg['crack']['width_stats']:
                    results['width_p95_mm'] = round(float(np.percentile(widths, 95)), self.decimals)
        
        return results
    
    def measure_spalling_efflorescence_instance(self, instance: Instance, defect_type: str) -> Dict:
        """测量单个 spalling 或 efflorescence 实例"""
        results = {
            'area_mm2': 0.0,
            'eq_diameter_mm': 0.0,
            'major_axis_mm': 0.0,
            'minor_axis_mm': 0.0,
        }
        
        mask = instance.mask
        if mask.sum() == 0:
            return results
        
        area_px = mask.sum()
        
        # Area
        if self.compute_cfg[defect_type]['area']:
            results['area_mm2'] = round(area_px * (self.mm_per_px ** 2), self.decimals)
        
        # Equivalent diameter
        if self.compute_cfg[defect_type]['eq_diameter']:
            results['eq_diameter_mm'] = round(np.sqrt(4 * area_px / np.pi) * self.mm_per_px, self.decimals)
        
        # Bounding box axes (minAreaRect)
        if self.compute_cfg[defect_type]['bbox_axes']:
            coords = np.column_stack(np.where(mask))
            if len(coords) >= 3:
                coords = coords.astype(np.float32)
                rect = cv2.minAreaRect(coords)
                width, height = rect[1]
                results['major_axis_mm'] = round(max(width, height) * self.mm_per_px, self.decimals)
                results['minor_axis_mm'] = round(min(width, height) * self.mm_per_px, self.decimals)
        
        return results


class InferencePipeline:
    """推理与后处理流水线"""
    
    # 中文类别名称映射
    CLASS_NAMES_CN = {
        'crack': '裂缝',
        'spalling': '剥落',
        'efflorescence': '泛碱',
    }
    
    # CSV 表头中英文映射
    CSV_HEADER_MAP = {
        'image_name': '图片名',
        'class_name': '缺陷类别',
        'class_name_cn': '缺陷类别(中文)',
        'instance_id': '实例编号',
        'area_mm2': '面积(mm²)',
        'length_mm': '长度(mm)',
        'width_mean_mm': '平均宽度(mm)',
        'width_max_mm': '最大宽度(mm)',
        'width_p95_mm': 'P95宽度(mm)',
        'eq_diameter_mm': '等效直径(mm)',
        'major_axis_mm': '长轴(mm)',
        'minor_axis_mm': '短轴(mm)',
        'bbox_xmin': '外接框xmin',
        'bbox_ymin': '外接框ymin',
        'bbox_xmax': '外接框xmax',
        'bbox_ymax': '外接框ymax',
        'bbox_width': '外接框宽度',
        'bbox_height': '外接框高度',
    }
    
    def __init__(self, config_path: str, overrides: Optional[dict] = None):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        
        # 命令行覆盖
        if overrides:
            self._update_config(self.cfg, overrides)
        
        # 初始化模型
        self._init_model()
        
        # 初始化测量器
        self.measurer = DefectMeasurer(self.cfg)
        
        # 构建类别映射 LUT
        self._build_remap_lut()
        
        # 初始化中文渲染器
        vis_cfg = self.cfg.get('visualize', {})
        font_cfg = vis_cfg.get('font', {})
        if not font_cfg:
            # 兼容旧配置（如果没有 font 字段，使用默认值）
            font_cfg = {}
        stroke_cfg = font_cfg.get('stroke', {})
        if not stroke_cfg:
            stroke_cfg = {}
        
        self.text_renderer = ChineseTextRenderer(
            font_path=font_cfg.get('font_path', ''),
            font_size=font_cfg.get('font_size', 26),
            font_color=font_cfg.get('font_color', [255, 255, 255]),
            line_spacing=font_cfg.get('line_spacing', 6),
            stroke_enable=stroke_cfg.get('enable', True),
            stroke_width=stroke_cfg.get('width', 2),
            stroke_color=stroke_cfg.get('color', [0, 0, 0]),
            color_space=self.cfg['classes'].get('color_space', 'BGR')
        )
        
        # 创建输出目录
        self.output_dir = Path(self.cfg['io']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'vis').mkdir(exist_ok=True)
        (self.output_dir / 'mask').mkdir(exist_ok=True)
        
        # 创建 panel 输出目录（如果启用）
        panel_cfg = self.cfg.get('visualize', {}).get('panel', {})
        if panel_cfg.get('enable', False):
            panel_dir = panel_cfg.get('output_dir', 'vis_panel')
            (self.output_dir / panel_dir).mkdir(exist_ok=True)
        
        # RLE 记录列表（用于批量保存）
        self.rle_records = []
    
    def _encode_rle_coco(self, binary_mask: np.ndarray) -> dict:
        """使用 COCO 格式编码 RLE（Fortran order）"""
        if not HAS_PYCOCOTOOLS:
            raise ImportError("pycocotools not available. Use simple format instead.")
        
        # COCO RLE 使用 Fortran order (column-major)
        rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')  # 转为字符串
        return rle
    
    def _encode_rle_simple(self, binary_mask: np.ndarray, order: str = 'C') -> dict:
        """使用简单格式编码 RLE（纯 numpy 实现）"""
        h, w = binary_mask.shape
        
        # 按指定顺序 flatten
        if order == 'C':
            flat = binary_mask.flatten(order='C')
        else:  # 'F'
            flat = binary_mask.flatten(order='F')
        
        # RLE 编码：交替表示连续 0/1 的长度
        counts = []
        current_val = flat[0]
        current_count = 1
        
        for val in flat[1:]:
            if val == current_val:
                current_count += 1
            else:
                counts.append(current_count)
                current_val = val
                current_count = 1
        counts.append(current_count)
        
        return {
            'order': order,
            'start_with': int(flat[0]),
            'counts': counts,
            'size': [h, w]
        }
    
    def _update_config(self, cfg: dict, overrides: dict):
        """递归更新配置"""
        for key, value in overrides.items():
            if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
                self._update_config(cfg[key], value)
            else:
                cfg[key] = value
    
    def _init_model(self):
        """初始化 MMSegmentation 模型"""
        register_all_modules(init_default_scope('mmseg'))
        
        mmseg_config = self.cfg['model']['mmseg_config']
        checkpoint = self.cfg['model']['checkpoint']
        device = self.cfg['model']['device']
        
        if not os.path.exists(mmseg_config):
            raise FileNotFoundError(f"Config file not found: {mmseg_config}")
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        
        self.model = init_model(
            mmseg_config,
            checkpoint,
            device=device
        )
        
        print(f"Model loaded from {checkpoint}")
        print(f"Device: {device}")
    
    def _build_remap_lut(self):
        """构建类别重映射查找表"""
        original = self.cfg['classes']['original_classes']
        output = self.cfg['classes']['output_classes']
        remap = self.cfg['classes']['class_remap']
        
        # 创建 LUT: 原始 id -> 输出 id
        max_orig_id = max(original.values())
        self.remap_lut = np.zeros(max_orig_id + 1, dtype=np.int32)
        
        for orig_name, orig_id in original.items():
            target_name = remap[orig_name]
            if target_name in output:
                self.remap_lut[orig_id] = output[target_name]
            else:
                # 映射到 background
                self.remap_lut[orig_id] = output['background']
    
    def _get_image_files(self, input_path: str) -> List[Path]:
        """获取输入图片文件列表"""
        input_path = Path(input_path)
        exts = self.cfg['io']['exts']
        
        if input_path.is_file():
            if input_path.suffix.lower() in exts:
                return [input_path]
            else:
                raise ValueError(f"Unsupported file extension: {input_path.suffix}")
        elif input_path.is_dir():
            files = []
            for ext in exts:
                files.extend(input_path.glob(f'*{ext}'))
                files.extend(input_path.glob(f'*{ext.upper()}'))
            return sorted(files)
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")
    
    def _inference_single(self, img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """单张图片推理"""
        # 读取原图（用于尺寸对齐）
        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            raise ValueError(f"Failed to read image: {img_path}")
        h_orig, w_orig = img_orig.shape[:2]
        
        # 使用 MMSegmentation 推理（严格按照 test_pipeline）
        result = inference_model(self.model, str(img_path))
        
        # 获取预测结果（兼容不同版本的 API）
        if hasattr(result, 'pred_sem_seg'):
            pred_raw = result.pred_sem_seg.data[0].cpu().numpy().astype(np.int32)
        elif hasattr(result, 'seg_logits'):
            pred_raw = result.seg_logits.data[0].cpu().numpy().argmax(axis=0).astype(np.int32)
        else:
            pred_raw = result.data[0].cpu().numpy().astype(np.int32)
        
        # 如果尺寸不一致，resize 回原图尺寸
        h_pred, w_pred = pred_raw.shape
        if (h_pred, w_pred) != (h_orig, w_orig):
            pred_raw = cv2.resize(
                pred_raw.astype(np.uint8),
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
        
        return pred_raw, img_orig
    
    def _apply_remap(self, pred_raw: np.ndarray) -> np.ndarray:
        """应用类别重映射"""
        return self.remap_lut[pred_raw]
    
    def _draw_instances_overlay(self, img: np.ndarray, instances_by_class: Dict[str, List[Instance]], 
                               measurements_by_instance: Dict[Tuple[str, int], Dict]) -> np.ndarray:
        """绘制实例 overlay"""
        vis_cfg = self.cfg['visualize']
        palette = self.cfg['classes']['palette']
        color_space = self.cfg['classes']['color_space']
        instance_label_cfg = vis_cfg.get('instance_label', {})
        
        overlay = img.copy()
        alpha = vis_cfg['alpha']
        
        # 绘制每个实例
        for class_name, instances in instances_by_class.items():
            if not instances:
                continue
            
            color = palette[class_name]
            if color_space == 'RGB':
                color = color[::-1]  # RGB -> BGR for OpenCV
            
            class_name_cn = self.CLASS_NAMES_CN[class_name]
            
            for instance in instances:
                mask = instance.mask
                
                # 半透明叠加
                overlay[mask] = (
                    overlay[mask] * (1 - alpha) + 
                    np.array(color) * alpha
                ).astype(np.uint8)
                
                # 绘制轮廓
                if vis_cfg['draw_contours']:
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(
                        overlay,
                        contours,
                        -1,
                        color,
                        vis_cfg['contour_thickness']
                    )
                
                # 绘制实例标签
                if instance_label_cfg.get('enable', True):
                    label_text = f"{class_name_cn}{instance.instance_id}"
                    draw_at = instance_label_cfg.get('draw_at', 'centroid')
                    
                    if draw_at == 'centroid':
                        pos = instance.centroid
                    else:  # bbox_center
                        x, y, w, h = instance.bbox
                        pos = (x + w // 2, y + h // 2)
                    
                    # 使用中文渲染器绘制标签
                    # 临时保存并修改颜色
                    original_color = self.text_renderer.font_color
                    self.text_renderer.font_color = color  # 使用实例对应的颜色
                    
                    # 计算文本尺寸以居中显示
                    text_w, text_h = self.text_renderer.get_text_size(label_text)
                    text_x = pos[0] - text_w // 2
                    text_y = pos[1] - text_h // 2
                    
                    # 根据配置决定是否绘制背景
                    draw_bg = instance_label_cfg.get('draw_bg', False)
                    bg_color = None
                    bg_padding = 0
                    if draw_bg:
                        bg_color = [255, 255, 255]
                        bg_padding = 2
                    
                    # 绘制文字（带或不带背景）
                    overlay = self.text_renderer.draw_text(
                        overlay,
                        label_text,
                        (text_x, text_y),
                        bg_color=bg_color,
                        bg_padding=bg_padding
                    )
                    
                    # 恢复原始颜色
                    self.text_renderer.font_color = original_color
        
        return overlay
    
    def _format_chinese_report(self, instances_by_class: Dict[str, List[Instance]],
                               measurements_by_instance: Dict[Tuple[str, int], Dict]) -> List[str]:
        """格式化中文报告"""
        lines = []
        info_cfg = self.cfg['visualize']['info_box']
        max_instances = info_cfg.get('max_instances_per_class', 3)
        show_summary = info_cfg.get('show_summary', True)
        decimals = self.cfg['measurement'].get('decimals', 2)
        
        for class_name in ['crack', 'spalling', 'efflorescence']:
            instances = instances_by_class.get(class_name, [])
            if not instances:
                continue
            
            class_name_cn = self.CLASS_NAMES_CN[class_name]
            lines.append(f"{class_name_cn}({len(instances)}处):")
            
            # 显示前 N 个实例
            for instance in instances[:max_instances]:
                key = (class_name, instance.instance_id)
                m = measurements_by_instance.get(key, {})
                
                if class_name == 'crack':
                    area = m.get('area_mm2', 0)
                    length = m.get('length_mm', 0)
                    width_mean = m.get('width_mean_mm', 0)
                    width_max = m.get('width_max_mm', 0)
                    
                    line = f"  {class_name_cn}{instance.instance_id} "
                    parts = []
                    if area > 0:
                        parts.append(f"面积={area:.{decimals}f} mm²")
                    if length > 0:
                        parts.append(f"长度={length:.{decimals}f} mm")
                    if width_mean > 0:
                        parts.append(f"平均宽度={width_mean:.{decimals}f} mm")
                    if width_max > 0:
                        parts.append(f"最大宽度={width_max:.{decimals}f} mm")
                    line += " ".join(parts)
                    lines.append(line)
                else:
                    area = m.get('area_mm2', 0)
                    eq_d = m.get('eq_diameter_mm', 0)
                    major = m.get('major_axis_mm', 0)
                    minor = m.get('minor_axis_mm', 0)
                    
                    line = f"  {class_name_cn}{instance.instance_id} "
                    parts = []
                    if area > 0:
                        parts.append(f"面积={area:.{decimals}f} mm²")
                    if eq_d > 0:
                        parts.append(f"等效直径={eq_d:.{decimals}f} mm")
                    if major > 0:
                        parts.append(f"长轴={major:.{decimals}f} mm")
                    if minor > 0:
                        parts.append(f"短轴={minor:.{decimals}f} mm")
                    line += " ".join(parts)
                    lines.append(line)
            
            # 汇总信息
            if show_summary and len(instances) > 0:
                total_area = sum(measurements_by_instance.get((class_name, inst.instance_id), {}).get('area_mm2', 0) 
                               for inst in instances)
                if class_name == 'crack':
                    total_length = sum(measurements_by_instance.get((class_name, inst.instance_id), {}).get('length_mm', 0) 
                                     for inst in instances)
                    if total_area > 0 or total_length > 0:
                        lines.append(f"  汇总: 总面积={total_area:.{decimals}f} mm², 总长度={total_length:.{decimals}f} mm")
                else:
                    if total_area > 0:
                        lines.append(f"  汇总: 总面积={total_area:.{decimals}f} mm²")
        
        return lines
    
    def _format_full_chinese_report(self, instances_by_class: Dict[str, List[Instance]],
                                   measurements_by_instance: Dict[Tuple[str, int], Dict],
                                   img_name: str) -> List[str]:
        """格式化完整中文报告（用于 panel，显示所有实例）"""
        lines = []
        decimals = self.cfg['measurement'].get('decimals', 2)
        
        # 图片名
        lines.append(f"图片: {img_name}")
        lines.append("")  # 空行
        
        # 三类缺陷
        for class_name in ['crack', 'spalling', 'efflorescence']:
            instances = instances_by_class.get(class_name, [])
            if not instances:
                continue
            
            class_name_cn = self.CLASS_NAMES_CN[class_name]
            lines.append(f"{class_name_cn}({len(instances)}处):")
            
            # 显示所有实例
            for instance in instances:
                key = (class_name, instance.instance_id)
                m = measurements_by_instance.get(key, {})
                
                if class_name == 'crack':
                    area = m.get('area_mm2', 0)
                    length = m.get('length_mm', 0)
                    width_mean = m.get('width_mean_mm', 0)
                    width_max = m.get('width_max_mm', 0)
                    width_p95 = m.get('width_p95_mm', 0)
                    
                    line = f"  {class_name_cn}{instance.instance_id}: "
                    parts = []
                    if area > 0:
                        parts.append(f"面积={area:.{decimals}f} mm²")
                    if length > 0:
                        parts.append(f"长度={length:.{decimals}f} mm")
                    if width_mean > 0:
                        parts.append(f"平均宽度={width_mean:.{decimals}f} mm")
                    if width_max > 0:
                        parts.append(f"最大宽度={width_max:.{decimals}f} mm")
                    if width_p95 > 0:
                        parts.append(f"P95宽度={width_p95:.{decimals}f} mm")
                    line += " | ".join(parts)
                    lines.append(line)
                else:
                    area = m.get('area_mm2', 0)
                    eq_d = m.get('eq_diameter_mm', 0)
                    major = m.get('major_axis_mm', 0)
                    minor = m.get('minor_axis_mm', 0)
                    
                    line = f"  {class_name_cn}{instance.instance_id}: "
                    parts = []
                    if area > 0:
                        parts.append(f"面积={area:.{decimals}f} mm²")
                    if eq_d > 0:
                        parts.append(f"等效直径={eq_d:.{decimals}f} mm")
                    if major > 0:
                        parts.append(f"长轴={major:.{decimals}f} mm")
                    if minor > 0:
                        parts.append(f"短轴={minor:.{decimals}f} mm")
                    line += " | ".join(parts)
                    lines.append(line)
            
            # 汇总信息
            total_area = sum(m.get('area_mm2', 0) for m in [
                measurements_by_instance.get((class_name, inst.instance_id), {})
                for inst in instances
            ])
            if class_name == 'crack':
                total_length = sum(m.get('length_mm', 0) for m in [
                    measurements_by_instance.get((class_name, inst.instance_id), {})
                    for inst in instances
                ])
                if total_length > 0:
                    lines.append(f"  汇总: 总面积={total_area:.{decimals}f} mm² | 总长度={total_length:.{decimals}f} mm")
                else:
                    lines.append(f"  汇总: 总面积={total_area:.{decimals}f} mm²")
            else:
                lines.append(f"  汇总: 总面积={total_area:.{decimals}f} mm²")
            
            lines.append("")  # 空行分隔
        
        return lines
    
    def _draw_info_box(self, img: np.ndarray, lines: List[str], vis_cfg: dict) -> np.ndarray:
        """绘制信息框（使用中文渲染器）"""
        if not lines:
            return img
        
        info_cfg = vis_cfg['info_box']
        padding = info_cfg['padding']
        bg_alpha = info_cfg['bg_alpha']
        
        # 合并所有行文本（用 \n 分隔）
        text = '\n'.join(lines)
        
        # 计算文本尺寸
        text_w, text_h = self.text_renderer.get_text_size(text)
        
        # 确定位置
        h_img, w_img = img.shape[:2]
        box_w = text_w + 2 * padding
        box_h = text_h + 2 * padding
        
        pos = info_cfg['position']
        if pos == 'topleft':
            x, y = padding, padding
        elif pos == 'topright':
            x, y = w_img - box_w - padding, padding
        elif pos == 'bottomleft':
            x, y = padding, h_img - box_h - padding
        else:  # bottomright
            x, y = w_img - box_w - padding, h_img - box_h - padding
        
        # 绘制半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
        img = cv2.addWeighted(img, 1 - bg_alpha, overlay, bg_alpha, 0)
        
        # 使用中文渲染器绘制文本
        img = self.text_renderer.draw_text(
            img,
            text,
            (x + padding, y + padding)
        )
        
        return img
    
    def _save_panel_visualizations(self, overlay: np.ndarray, instances_by_class: Dict[str, List[Instance]],
                                   measurements_by_instance: Dict[Tuple[str, int], Dict], img_name: str):
        """保存带信息面板的可视化图"""
        panel_cfg = self.cfg.get('visualize', {}).get('panel', {})
        if not panel_cfg.get('enable', False):
            return
        
        # 获取配置
        placement = panel_cfg.get('placement', 'left')
        width_px = panel_cfg.get('width_px', 520)
        height_px = panel_cfg.get('height_px', 300)
        bg_color = panel_cfg.get('bg_color', [30, 30, 30])
        padding = panel_cfg.get('padding', 16)
        line_spacing = panel_cfg.get('line_spacing', 8)
        title_font_scale = panel_cfg.get('title_font_scale', 1.0)
        text_font_scale = panel_cfg.get('text_font_scale', 0.8)
        auto_shrink = panel_cfg.get('auto_shrink', True)
        paginate_cfg = panel_cfg.get('paginate', {})
        paginate_enable = paginate_cfg.get('enable', True)
        lines_per_page = paginate_cfg.get('lines_per_page', 35)
        min_font_size = panel_cfg.get('min_font_size', 12)
        output_dir = panel_cfg.get('output_dir', 'vis_panel')
        
        # 生成完整报告
        report_lines = self._format_full_chinese_report(instances_by_class, measurements_by_instance, img_name)
        
        # 获取 overlay 尺寸
        h_overlay, w_overlay = overlay.shape[:2]
        
        # 创建临时渲染器用于计算文本尺寸
        vis_cfg = self.cfg.get('visualize', {})
        font_cfg = vis_cfg.get('font', {})
        stroke_cfg = font_cfg.get('stroke', {})
        
        # 计算需要的行数和分页
        if paginate_enable and len(report_lines) > lines_per_page:
            # 分页处理
            num_pages = (len(report_lines) + lines_per_page - 1) // lines_per_page
            for page_idx in range(num_pages):
                start_idx = page_idx * lines_per_page
                end_idx = min(start_idx + lines_per_page, len(report_lines))
                page_lines = report_lines[start_idx:end_idx]
                
                panel_img = self._create_panel_image(
                    overlay, page_lines, placement, width_px, height_px,
                    bg_color, padding, line_spacing, title_font_scale, text_font_scale,
                    auto_shrink, min_font_size, font_cfg, stroke_cfg
                )
                
                # 保存
                page_suffix = f"_p{page_idx + 1}" if num_pages > 1 else ""
                output_path = self.output_dir / output_dir / f'{img_name}_panel{page_suffix}.png'
                cv2.imwrite(str(output_path), panel_img)
        else:
            # 单页处理
            panel_img = self._create_panel_image(
                overlay, report_lines, placement, width_px, height_px,
                bg_color, padding, line_spacing, title_font_scale, text_font_scale,
                auto_shrink, min_font_size, font_cfg, stroke_cfg
            )
            
            # 保存
            output_path = self.output_dir / output_dir / f'{img_name}_panel.png'
            cv2.imwrite(str(output_path), panel_img)
    
    def _create_panel_image(self, overlay: np.ndarray, report_lines: List[str],
                           placement: str, width_px: int, height_px: int,
                           bg_color: List[int], padding: int, line_spacing: int,
                           title_font_scale: float, text_font_scale: float,
                           auto_shrink: bool, min_font_size: int,
                           font_cfg: dict, stroke_cfg: dict) -> np.ndarray:
        """创建带面板的图像"""
        h_overlay, w_overlay = overlay.shape[:2]
        color_space = self.cfg['classes'].get('color_space', 'BGR')
        
        # 创建临时渲染器用于计算文本尺寸
        temp_renderer = ChineseTextRenderer(
            font_path=font_cfg.get('font_path', ''),
            font_size=int(font_cfg.get('font_size', 26) * text_font_scale),
            font_color=font_cfg.get('font_color', [255, 255, 255]),
            line_spacing=line_spacing,
            stroke_enable=stroke_cfg.get('enable', True),
            stroke_width=stroke_cfg.get('width', 2),
            stroke_color=stroke_cfg.get('color', [0, 0, 0]),
            color_space=color_space
        )
        
        # 计算文本尺寸（可能需要自动缩放）
        text = '\n'.join(report_lines)
        text_w, text_h = temp_renderer.get_text_size(text)
        
        # 自动缩放字体（如果需要）
        if auto_shrink:
            if placement == 'left':
                max_text_h = h_overlay - 2 * padding
                max_text_w = width_px - 2 * padding
            else:  # bottom
                max_text_h = height_px - 2 * padding
                max_text_w = w_overlay - 2 * padding
            
            current_font_size = int(font_cfg.get('font_size', 26) * text_font_scale)
            scale_factor = 1.0
            
            # 如果文本太大，缩小字体
            while (text_h > max_text_h or text_w > max_text_w) and current_font_size > min_font_size:
                scale_factor *= 0.9
                current_font_size = max(int(current_font_size * 0.9), min_font_size)
                temp_renderer = ChineseTextRenderer(
                    font_path=font_cfg.get('font_path', ''),
                    font_size=current_font_size,
                    font_color=font_cfg.get('font_color', [255, 255, 255]),
                    line_spacing=int(line_spacing * scale_factor),
                    stroke_enable=stroke_cfg.get('enable', True),
                    stroke_width=stroke_cfg.get('width', 2),
                    stroke_color=stroke_cfg.get('color', [0, 0, 0]),
                    color_space=color_space
                )
                text_w, text_h = temp_renderer.get_text_size(text)
        
        # 创建画布
        if placement == 'left':
            # 左侧面板
            panel_h = h_overlay
            panel_w = width_px
            canvas_h = h_overlay
            canvas_w = w_overlay + panel_w
            panel_x = 0
            panel_y = 0
            overlay_x = panel_w
            overlay_y = 0
        else:  # bottom
            # 底部面板
            panel_h = height_px
            panel_w = w_overlay
            canvas_h = h_overlay + panel_h
            canvas_w = w_overlay
            panel_x = 0
            panel_y = h_overlay
            overlay_x = 0
            overlay_y = 0
        
        # 创建画布
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # 绘制面板背景
        bg_color_bgr = bg_color if color_space == 'BGR' else bg_color[::-1]
        canvas[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = bg_color_bgr
        
        # 复制 overlay
        canvas[overlay_y:overlay_y+h_overlay, overlay_x:overlay_x+w_overlay] = overlay
        
        # 在面板上绘制文本
        text_x = panel_x + padding
        text_y = panel_y + padding
        
        # 使用最终确定的渲染器绘制文本
        canvas = temp_renderer.draw_text(
            canvas,
            text,
            (text_x, text_y),
            bg_color=None,  # 面板已有背景，不需要文字背景
            bg_padding=0
        )
        
        return canvas
    
    def _save_masks(self, pred_out: np.ndarray, img_name: str):
        """保存 mask 文件"""
        save_cfg = self.cfg['io']['save']
        palette = self.cfg['classes']['palette']
        color_space = self.cfg['classes']['color_space']
        
        # ID mask
        if save_cfg['id_mask']:
            id_mask = pred_out.astype(np.uint8)
            cv2.imwrite(
                str(self.output_dir / 'mask' / f'{img_name}_id.png'),
                id_mask
            )
        
        # Color mask
        if save_cfg['color_mask']:
            color_mask = np.zeros((*pred_out.shape, 3), dtype=np.uint8)
            for class_id, class_name in [(1, 'crack'), (2, 'spalling'), (3, 'efflorescence')]:
                mask = (pred_out == class_id)
                color = palette[class_name]
                if color_space == 'RGB':
                    color = color[::-1]
                color_mask[mask] = color
            cv2.imwrite(
                str(self.output_dir / 'mask' / f'{img_name}_color.png'),
                color_mask
            )
    
    def _save_debug_raw7(self, img: np.ndarray, pred_raw: np.ndarray, img_name: str):
        """保存原始 7 类 debug overlay"""
        palette_7class = {
            0: [0, 0, 0],
            1: [255, 255, 255],
            2: [255, 0, 0],
            3: [255, 255, 0],
            4: [0, 255, 255],
            5: [0, 255, 0],
            6: [0, 0, 255],
        }
        
        overlay = img.copy()
        alpha = self.cfg['visualize']['alpha']
        
        for class_id, color in palette_7class.items():
            mask = (pred_raw == class_id)
            if mask.sum() > 0:
                overlay[mask] = (
                    overlay[mask] * (1 - alpha) + 
                    np.array(color[::-1]) * alpha
                ).astype(np.uint8)
        
        cv2.imwrite(
            str(self.output_dir / 'vis' / f'{img_name}_debug_raw7.png'),
            overlay
        )
    
    def process_single(self, img_path: Path) -> Tuple[List[Dict], Dict]:
        """处理单张图片（返回实例级结果列表和 RLE 记录）"""
        img_name = img_path.stem
        
        # 推理
        pred_raw, img_orig = self._inference_single(img_path)
        h, w = img_orig.shape[:2]
        
        # 类别重映射
        pred_out = self._apply_remap(pred_raw)
        
        # 提取实例并量化
        instances_by_class = {}
        measurements_by_instance = {}
        instance_results = []
        rle_masks = []  # 用于 RLE 编码的实例 mask 列表
        
        for class_id, class_name in [(1, 'crack'), (2, 'spalling'), (3, 'efflorescence')]:
            mask = (pred_out == class_id)
            if mask.sum() == 0:
                instances_by_class[class_name] = []
                continue
            
            # 提取实例（传递 debug 参数）
            debug_cfg = self.cfg.get('visualize', {}).get('debug', {})
            debug_output_dir = None
            debug_img_name = None
            if debug_cfg.get('save_bridge_debug', False):
                debug_output_dir = self.output_dir
                debug_img_name = img_name
            
            instances = self.measurer.extract_instances(
                mask, class_name, 
                debug_output_dir=debug_output_dir,
                debug_img_name=debug_img_name
            )
            instances_by_class[class_name] = instances
            
            # 量化每个实例
            for instance in instances:
                if class_name == 'crack':
                    m = self.measurer.measure_crack_instance(instance)
                else:
                    m = self.measurer.measure_spalling_efflorescence_instance(instance, class_name)
                
                key = (class_name, instance.instance_id)
                measurements_by_instance[key] = m
                
                # 构建结果记录（添加 bbox 信息）
                x, y, bbox_w, bbox_h = instance.bbox
                result = {
                    'image_name': img_name,
                    'class_name': class_name,
                    'class_name_cn': self.CLASS_NAMES_CN[class_name],
                    'instance_id': instance.instance_id,
                    'bbox_xmin': x,
                    'bbox_ymin': y,
                    'bbox_xmax': x + bbox_w,
                    'bbox_ymax': y + bbox_h,
                    'bbox_width': bbox_w,
                    'bbox_height': bbox_h,
                    **m
                }
                instance_results.append(result)
                
                # 收集实例 mask 用于 RLE 编码
                rle_masks.append({
                    'class_name': class_name,
                    'instance_id': instance.instance_id,
                    'mask': instance.mask
                })
        
        # 构建 RLE 记录
        rle_record = {
            'image_name': img_name,
            'width': w,
            'height': h,
            'masks': []
        }
        
        # 检查是否需要生成 RLE
        rle_cfg = self.cfg.get('io', {}).get('output', {}).get('rle', {})
        if rle_cfg.get('enable', False):
            rle_format = rle_cfg.get('format', 'coco')
            include_instance = rle_cfg.get('include_instance_masks', True)
            
            if include_instance:
                for mask_info in rle_masks:
                    binary_mask = mask_info['mask'].astype(np.uint8)
                    
                    try:
                        if rle_format == 'coco' and HAS_PYCOCOTOOLS:
                            rle = self._encode_rle_coco(binary_mask)
                        else:
                            # 使用 simple 格式
                            if rle_format == 'coco' and not HAS_PYCOCOTOOLS:
                                print(f"Warning: pycocotools not available, using simple RLE format for {img_name}")
                            order = rle_cfg.get('order', 'C')
                            rle = self._encode_rle_simple(binary_mask, order)
                        
                        rle_record['masks'].append({
                            'class_name': mask_info['class_name'],
                            'instance_id': mask_info['instance_id'],
                            'rle': rle
                        })
                    except Exception as e:
                        print(f"Warning: Failed to encode RLE for {img_name} {mask_info['class_name']} instance {mask_info['instance_id']}: {e}")
                        continue
        
        # 可视化
        save_cfg = self.cfg['io']['save']
        if save_cfg['overlay']:
            overlay = self._draw_instances_overlay(img_orig, instances_by_class, measurements_by_instance)
            
            # 添加信息框
            if self.cfg['visualize']['info_box']['enable']:
                report_lines = self._format_chinese_report(instances_by_class, measurements_by_instance)
                overlay = self._draw_info_box(overlay, report_lines, self.cfg['visualize'])
            
            cv2.imwrite(
                str(self.output_dir / 'vis' / f'{img_name}_overlay.png'),
                overlay
            )
            
            # 保存 panel 版可视化（额外输出，不影响现有 overlay）
            self._save_panel_visualizations(overlay, instances_by_class, measurements_by_instance, img_name)
        
        if save_cfg['debug_raw_7class_overlay']:
            self._save_debug_raw7(img_orig, pred_raw, img_name)
        
        # 保存 mask
        self._save_masks(pred_out, img_name)
        
        return instance_results, rle_record
    
    def run(self):
        """运行推理流水线"""
        input_path = self.cfg['io']['input']
        img_files = self._get_image_files(input_path)
        
        if not img_files:
            print(f"No image files found in {input_path}")
            return
        
        print(f"Found {len(img_files)} images")
        
        # 处理所有图片
        all_results = []
        all_rle_records = []
        show_progress = self.cfg['model']['show_progress']
        
        iterator = tqdm(img_files) if show_progress else img_files
        for img_path in iterator:
            try:
                results, rle_record = self.process_single(img_path)
                all_results.extend(results)
                all_rle_records.append(rle_record)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存汇总
        if all_results:
            df = pd.DataFrame(all_results)
            save_cfg = self.cfg['io']['save']
            output_cfg = self.cfg['measurement'].get('output', {})
            
            # CSV 表头中文化
            csv_header_lang = self.cfg.get('io', {}).get('csv', {}).get('header_language', 'zh')
            
            if save_cfg['csv'] or output_cfg.get('save_instance_csv', True):
                csv_path = self.output_dir / 'metrics_instances.csv'
                
                # 如果使用中文表头，重命名列
                if csv_header_lang == 'zh':
                    df_export = df.copy()
                    df_export.columns = [self.CSV_HEADER_MAP.get(col, col) for col in df_export.columns]
                else:
                    df_export = df
                
                df_export.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"Instance metrics saved to {csv_path} (header language: {csv_header_lang})")
            
            if save_cfg['json'] or output_cfg.get('save_instance_json', False):
                json_path = self.output_dir / 'metrics_instances.json'
                df.to_json(json_path, orient='records', indent=2, force_ascii=False)
                print(f"Instance metrics saved to {json_path}")
        
        # 保存 RLE 文件（即使没有结果也要保存，确保每张图片都有记录）
        rle_cfg = self.cfg.get('io', {}).get('output', {}).get('rle', {})
        if rle_cfg.get('enable', False):
            if all_rle_records:
                self._save_rle_records(all_rle_records, rle_cfg)
            else:
                print("Warning: No RLE records to save (no images processed)")
        
        print(f"Processing complete. Results saved to {self.output_dir}")
    
    def _save_rle_records(self, records: List[Dict], rle_cfg: dict):
        """保存 RLE 编码记录"""
        rle_file = rle_cfg.get('file', 'masks_rle.jsonl')
        rle_path = self.output_dir / rle_file
        
        # 判断是 JSONL 还是 JSON
        is_jsonl = rle_file.endswith('.jsonl')
        
        if is_jsonl:
            # JSONL 格式：每行一个 JSON 对象
            with open(rle_path, 'w', encoding='utf-8') as f:
                for record in records:
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
            print(f"RLE records saved to {rle_path} (JSONL format, {len(records)} records)")
        else:
            # JSON 格式：单个 JSON 数组
            with open(rle_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"RLE records saved to {rle_path} (JSON format, {len(records)} records)")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MMSegmentation 推理与缺陷量化脚本（实例级）')
    parser.add_argument('--cfg', type=str, required=True, help='YAML 配置文件路径')
    parser.add_argument('--input', type=str, default=None, help='覆盖输入路径（可选）')
    parser.add_argument('--output-dir', type=str, default=None, help='覆盖输出目录（可选）')
    parser.add_argument('--device', type=str, default=None, help='覆盖设备（可选，如 cuda:0）')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 构建覆盖字典
    overrides = {}
    if args.input:
        overrides['io'] = {'input': args.input}
    if args.output_dir:
        if 'io' not in overrides:
            overrides['io'] = {}
        overrides['io']['output_dir'] = args.output_dir
    if args.device:
        overrides['model'] = {'device': args.device}
    
    # 运行流水线
    pipeline = InferencePipeline(args.cfg, overrides)
    pipeline.run()


if __name__ == '__main__':
    main()
