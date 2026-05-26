# analyze_s2ds_crack_issue.py  (py3.7/3.8 compatible)
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import Counter
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
from PIL import Image

PALETTE_RGB = {
    (0, 0, 0): "background",
    (255, 255, 255): "crack",
    (255, 0, 0): "spalling",
    (255, 255, 0): "corrosion",
    (0, 255, 255): "efflorescence",
    (0, 255, 0): "vegetation",
    (0, 0, 255): "control_point",
}
CRACK_RGB = (255, 255, 255)


def load_mask_rgb(path: Path) -> Tuple[np.ndarray, str]:
    with Image.open(path) as im:
        mode = im.mode
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)  # drop alpha if any
    return rgb, mode


def unique_colors(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3)
    view = flat.view([("r", np.uint8), ("g", np.uint8), ("b", np.uint8)]).reshape(-1)
    uniq = np.unique(view)
    uniq_rgb = uniq.view(np.uint8).reshape(-1, 3)
    return uniq_rgb


def count_exact_color(rgb: np.ndarray, color: Tuple[int, int, int]) -> int:
    c = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    return int(np.all(rgb == c, axis=-1).sum())


def count_near_white(rgb: np.ndarray, tol: int) -> int:
    rgb_i = rgb.astype(np.int16)
    diff = np.max(np.abs(rgb_i - 255), axis=-1)
    near = (diff <= tol)
    exact = np.all(rgb == 255, axis=-1)
    return int((near & (~exact)).sum())


def find_split_mask_dir(root: Path, split: str) -> Optional[Path]:
    cand1 = root / (split + "_lab")
    cand2 = root / split
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


def analyze_split(mask_dir: Path, max_files: int, near_white_tol: int, topk_unknown: int) -> Dict[str, Any]:
    masks = sorted(mask_dir.glob("*_lab.png"))
    if not masks:
        return {"error": "no *_lab.png found in {}".format(mask_dir)}

    if max_files > 0:
        masks = masks[:max_files]

    mode_cnt = Counter()
    total_pixels = 0

    crack_img_count = 0
    crack_pixels_total = 0

    near_white_pixels_total = 0
    near_white_img_count = 0

    unknown_color_counter = Counter()
    palette_seen_counter = Counter()

    examples_no_crack: List[str] = []
    examples_near_white: List[str] = []
    examples_unknown: List[str] = []

    for p in masks:
        rgb, mode = load_mask_rgb(p)
        mode_cnt[mode] += 1
        H, W, _ = rgb.shape
        total_pixels += H * W

        uniq = unique_colors(rgb)
        uniq_tuples = [tuple(x.tolist()) for x in uniq]

        for c in uniq_tuples:
            if c in PALETTE_RGB:
                palette_seen_counter[c] += 1
            else:
                unknown_color_counter[c] += 1

        crack_pixels = count_exact_color(rgb, CRACK_RGB)
        crack_pixels_total += crack_pixels
        if crack_pixels > 0:
            crack_img_count += 1
        else:
            if len(examples_no_crack) < 10:
                examples_no_crack.append(str(p))

        nw = count_near_white(rgb, near_white_tol)
        near_white_pixels_total += nw
        if nw > 0:
            near_white_img_count += 1
            if len(examples_near_white) < 10:
                examples_near_white.append(str(p))

        if any((c not in PALETTE_RGB) for c in uniq_tuples):
            if len(examples_unknown) < 10:
                examples_unknown.append(str(p))

    top_unknown = unknown_color_counter.most_common(topk_unknown)
    top_unknown_fmt = [{"rgb": list(k), "count_in_images": v} for k, v in top_unknown]

    palette_seen_fmt = [{"rgb": list(k), "name": PALETTE_RGB[k], "count_in_images": v}
                        for k, v in palette_seen_counter.most_common()]

    return {
        "mask_dir": str(mask_dir),
        "scanned_files": len(masks),
        "pil_modes": dict(mode_cnt),
        "total_pixels_scanned": int(total_pixels),
        "crack": {
            "crack_rgb": list(CRACK_RGB),
            "images_with_crack": int(crack_img_count),
            "crack_pixels_total": int(crack_pixels_total),
            "crack_pixel_ratio": float(crack_pixels_total / max(1, total_pixels)),
            "examples_no_crack": examples_no_crack,
        },
        "near_white": {
            "tol": int(near_white_tol),
            "images_with_near_white_not_exact": int(near_white_img_count),
            "near_white_pixels_total": int(near_white_pixels_total),
            "near_white_pixel_ratio": float(near_white_pixels_total / max(1, total_pixels)),
            "examples_near_white": examples_near_white,
        },
        "palette_seen_in_images": palette_seen_fmt,
        "unknown_colors": {
            "unique_unknown_color_count": int(len(unknown_color_counter)),
            "top_unknown": top_unknown_fmt,
            "examples_with_unknown": examples_unknown,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--max_per_split", type=int, default=0)
    ap.add_argument("--near_white_tol", type=int, default=2)
    ap.add_argument("--topk_unknown", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root)
    splits = ["train", "val", "test"]

    print("=== S2DS Crack Issue Analyzer ===")
    print("Root:", root)
    print("max_per_split={}, near_white_tol={}".format(args.max_per_split, args.near_white_tol))
    print()

    for sp in splits:
        mask_dir = find_split_mask_dir(root, sp)
        if mask_dir is None:
            print("[{}] mask dir not found (expected {}_lab or {})".format(sp, sp, sp))
            continue

        rep = analyze_split(mask_dir, args.max_per_split, args.near_white_tol, args.topk_unknown)
        if "error" in rep:
            print("[{}] ERROR: {}".format(sp, rep["error"]))
            continue

        crack = rep["crack"]
        nw = rep["near_white"]
        unk = rep["unknown_colors"]

        print("--- [{}] ---".format(sp))
        print("mask_dir:", rep["mask_dir"])
        print("scanned_files:", rep["scanned_files"])
        print("pil_modes:", rep["pil_modes"])
        print("crack images: {}/{} , crack_pixel_ratio={:.6f}".format(
            crack["images_with_crack"], rep["scanned_files"], crack["crack_pixel_ratio"]))
        print("near-white(not exact) images: {}/{} , near_white_pixel_ratio={:.6f}".format(
            nw["images_with_near_white_not_exact"], rep["scanned_files"], nw["near_white_pixel_ratio"]))
        print("unknown unique colors:", unk["unique_unknown_color_count"])

        if crack["images_with_crack"] == 0:
            print("!! DIAG: 扫描范围内 GT 完全没有纯白(255,255,255) → 要么val根本没裂缝，要么白色被污染/不纯。")
        if nw["images_with_near_white_not_exact"] > 0 and crack["images_with_crack"] == 0:
            print("!! DIAG: 有近白但无纯白 → 很可能插值/resize把白色裂缝变成(254,254,254...)，导致颜色映射失败。")
        if unk["unique_unknown_color_count"] > 0:
            print("!! DIAG: mask 出现非标准调色板颜色 → 若严格匹配会变背景/ignore；需nearest resize或容错匹配。")

        if unk["unique_unknown_color_count"] > 0:
            print("top unknown colors (rgb -> count_in_images):")
            for item in unk["top_unknown"][:10]:
                print("  {} -> {}".format(tuple(item["rgb"]), item["count_in_images"]))

        print()

    print("Done.")


if __name__ == "__main__":
    main()
