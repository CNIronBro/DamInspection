#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect label mask types for semantic segmentation datasets.

Default expects:
  /root/Dataset/{DeepCrack,s2ds}/{train_lab,val_lab,test_lab}

It will infer label type per split:
  - binary_grayscale: 1-channel with values like {0,255} or {0,1}
  - indexed_multiclass: 1-channel with small set of integer ids
  - rgb_colormap: 3-channel with small set of unique colors
  - unknown: doesn't look like a standard segmentation label

Usage:
  python inspect_label_types.py \
    --root /root/Dataset \
    --datasets DeepCrack s2ds \
    --splits train val test \
    --sample-files 60 \
    --pixel-sample 200000 \
    --out /root/Dataset/label_report.json
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = []
    for fn in os.listdir(folder):
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(folder, fn))
    files.sort()
    return files


def is_rgb_grayscale(arr_rgb: np.ndarray) -> bool:
    # arr_rgb: HxWx3
    return np.array_equal(arr_rgb[..., 0], arr_rgb[..., 1]) and np.array_equal(arr_rgb[..., 1], arr_rgb[..., 2])


def sample_pixels(arr: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 else arr.reshape(-1)
    n = flat.shape[0]
    if n <= k:
        return flat
    idx = rng.choice(n, size=k, replace=False)
    return flat[idx]


def infer_label_type(modes: Counter,
                     channel_counts: Counter,
                     unique_value_stats: Dict[str, Any],
                     unique_color_stats: Dict[str, Any]) -> str:
    """
    Decide label type using aggregated stats from sampled files.
    """
    # Prefer the dominant channel count / mode
    dominant_channels = channel_counts.most_common(1)[0][0] if channel_counts else None
    dominant_mode = modes.most_common(1)[0][0] if modes else None

    if dominant_channels == 1:
        # Look at unique values across samples
        u_all = unique_value_stats.get("union_values", [])
        u_cnt = len(u_all)
        if u_cnt <= 2:
            vals = set(u_all)
            # common binary patterns
            if vals.issubset({0, 1}) or vals.issubset({0, 255}) or vals.issubset({0, 1, 255}):
                return "binary_grayscale"
        # multiclass index mask tends to have limited ids
        # (often <= 20; sometimes up to 255 for palette index)
        if 2 < u_cnt <= 256:
            # if values are integers in [0,255] it's likely indexed mask
            vmin, vmax = unique_value_stats.get("min", None), unique_value_stats.get("max", None)
            if vmin is not None and vmax is not None and 0 <= vmin <= 255 and 0 <= vmax <= 255:
                return "indexed_multiclass"
        return "unknown"

    if dominant_channels == 3:
        # Look at unique colors; segmentation colormap typically has small set
        color_union = unique_color_stats.get("union_colors", [])
        c_cnt = len(color_union)
        if c_cnt <= 256:
            return "rgb_colormap"
        return "unknown"

    # Palette mode in PIL often shows as "P" but array becomes 2D (1 channel)
    if dominant_mode == "P":
        return "indexed_multiclass"

    return "unknown"


def analyze_split(label_dir: str,
                  sample_files: int = 60,
                  pixel_sample: int = 200000,
                  seed: int = 42) -> Dict[str, Any]:
    files = list_images(label_dir)
    out: Dict[str, Any] = {
        "label_dir": label_dir,
        "num_files": len(files),
        "extensions": {},
        "modes": {},
        "channel_counts": {},
        "sizes": {},
        "unique_values": {},
        "unique_colors": {},
        "inferred_type": "unknown",
        "top_colors": [],
        "notes": []
    }

    if not files:
        out["notes"].append("label_dir not found or empty")
        return out

    # extension stats
    ext_counter = Counter(os.path.splitext(p)[1].lower() for p in files)
    out["extensions"] = dict(ext_counter)

    # choose sampled files
    rng = random.Random(seed)
    pick = files if len(files) <= sample_files else rng.sample(files, sample_files)

    modes = Counter()
    ch_counts = Counter()
    size_counter = Counter()

    union_values = set()
    union_colors = set()
    top_color_counter = Counter()

    np_rng = np.random.default_rng(seed)

    # limits to prevent giant JSON
    MAX_UNION_VALUES = 512
    MAX_UNION_COLORS = 1024

    for fp in pick:
        try:
            im = Image.open(fp)
            modes[im.mode] += 1
            size_counter[im.size] += 1

            arr = np.array(im)

            # Normalize to either 2D (H,W) or 3D (H,W,C)
            if arr.ndim == 2:
                ch_counts[1] += 1
                # sample pixels to estimate unique values
                pix = sample_pixels(arr, k=pixel_sample, rng=np_rng)
                vals = np.unique(pix)
                # update union (cap size)
                for v in vals.tolist():
                    if len(union_values) < MAX_UNION_VALUES:
                        union_values.add(int(v))
            elif arr.ndim == 3 and arr.shape[2] in (3, 4):
                # If RGBA, drop A channel for color analysis
                if arr.shape[2] == 4:
                    arr = arr[..., :3]
                if is_rgb_grayscale(arr):
                    ch_counts[1] += 1
                    # treat as grayscale
                    gray = arr[..., 0]
                    pix = sample_pixels(gray, k=pixel_sample, rng=np_rng)
                    vals = np.unique(pix)
                    for v in vals.tolist():
                        if len(union_values) < MAX_UNION_VALUES:
                            union_values.add(int(v))
                else:
                    ch_counts[3] += 1
                    pix = sample_pixels(arr, k=pixel_sample, rng=np_rng)  # Nx3
                    # unique colors on sampled pixels
                    # convert to tuples for hashing
                    for rgb in pix.tolist():
                        if len(union_colors) < MAX_UNION_COLORS:
                            union_colors.add(tuple(int(x) for x in rgb))
                        top_color_counter[tuple(int(x) for x in rgb)] += 1
            else:
                out["notes"].append(f"unsupported array shape {arr.shape} in {os.path.basename(fp)}")
        except Exception as e:
            out["notes"].append(f"failed to read {os.path.basename(fp)}: {repr(e)}")

    out["modes"] = dict(modes)
    out["channel_counts"] = dict(ch_counts)

    # sizes: only keep top 5
    out["sizes"] = {str(k): v for k, v in size_counter.most_common(5)}

    # summarize unique values
    if union_values:
        vals_sorted = sorted(union_values)
        out["unique_values"] = {
            "union_count": len(vals_sorted),
            "union_values": vals_sorted[:256],  # cap
            "min": int(vals_sorted[0]),
            "max": int(vals_sorted[-1]),
        }
        if len(vals_sorted) > 256:
            out["unique_values"]["note"] = f"union_values truncated to 256 (actual {len(vals_sorted)})"

    # summarize unique colors
    if union_colors:
        colors_sorted = sorted(union_colors)
        out["unique_colors"] = {
            "union_count": len(colors_sorted),
            "union_colors": [list(c) for c in colors_sorted[:256]],  # cap
        }
        if len(colors_sorted) > 256:
            out["unique_colors"]["note"] = f"union_colors truncated to 256 (actual {len(colors_sorted)})"

        # top colors by frequency (from sampled pixels)
        topk = top_color_counter.most_common(15)
        out["top_colors"] = [{"rgb": list(rgb), "count": int(cnt)} for rgb, cnt in topk]

    out["inferred_type"] = infer_label_type(modes, ch_counts, out.get("unique_values", {}), out.get("unique_colors", {}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/Dataset", help="Dataset root directory")
    ap.add_argument("--datasets", nargs="+", default=["DeepCrack", "s2ds"], help="Dataset folder names under root. Use '.' to scan root directly.")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to scan")
    ap.add_argument("--sample-files", type=int, default=60, help="How many label files to sample per split")
    ap.add_argument("--pixel-sample", type=int, default=200000, help="How many pixels to sample per file")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="", help="Optional output JSON path")
    args = ap.parse_args()

    report: Dict[str, Any] = {
        "root": args.root,
        "datasets": {},
        "config": {
            "splits": args.splits,
            "sample_files": args.sample_files,
            "pixel_sample": args.pixel_sample,
            "seed": args.seed,
        }
    }

    for ds in args.datasets:
        ds_path = args.root if ds == '.' else os.path.join(args.root, ds)
        ds_entry = {"path": ds_path, "splits": {}}
        for sp in args.splits:
            lab_dir = os.path.join(ds_path, f"{sp}_lab")
            ds_entry["splits"][sp] = analyze_split(
                lab_dir,
                sample_files=args.sample_files,
                pixel_sample=args.pixel_sample,
                seed=args.seed
            )
        report["datasets"][ds] = ds_entry

    # pretty print summary
    print("=" * 80)
    for ds, ds_entry in report["datasets"].items():
        print(f"[{ds}]  path={ds_entry['path']}")
        for sp, info in ds_entry["splits"].items():
            print(f"  - {sp}: files={info['num_files']}, inferred={info['inferred_type']}, "
                  f"modes={info.get('modes', {})}, channels={info.get('channel_counts', {})}")
            if info.get("unique_values"):
                uv = info["unique_values"]
                print(f"      unique_values: count={uv.get('union_count')} min={uv.get('min')} max={uv.get('max')}")
            if info.get("unique_colors"):
                uc = info["unique_colors"]
                print(f"      unique_colors: count≈{uc.get('union_count')} (sampled), top_colors={info.get('top_colors', [])[:5]}")
        print()

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()
