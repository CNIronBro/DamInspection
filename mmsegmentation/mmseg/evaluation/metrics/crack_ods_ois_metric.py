# crack_ods_ois_metric.py
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class CrackODS_OIS_Metric(BaseMetric):
    """ODS/OIS for crack-like binary segmentation, following the provided CVPR evaluate.py.

    OIS: for each image, max F1 over thresholds; then mean over images.
    ODS: for each threshold, mean F1 over images; then max over thresholds.

    Notes:
    - Returns values in [0, 1] to match typical crack papers (ODS/OIS in decimals).
    - Uses seg_logits -> probability for threshold sweeping.
    """

    def __init__(
        self,
        pos_label: int = 1,
        ignore_index: int = 255,
        thresh_step: float = 0.01,
        use_sigmoid_if_one_channel: bool = True,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.pos_label = int(pos_label)
        self.ignore_index = int(ignore_index)
        self.thresh_step = float(thresh_step)
        # thresholds like np.arange(0.0, 1.0, 0.01) in evaluate.py
        self.thresholds = torch.arange(0.0, 1.0, self.thresh_step, dtype=torch.float32)
        self.use_sigmoid_if_one_channel = bool(use_sigmoid_if_one_channel)

    @torch.no_grad()
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            # --- get GT label map (H,W) ---
            gt = sample["gt_sem_seg"]["data"]
            if gt.ndim > 2:
                gt = gt.squeeze()
            gt = gt.to(torch.int64)

            valid = gt.ne(self.ignore_index)  # like IoUMetric mask :contentReference[oaicite:10]{index=10}
            gt_pos = gt.eq(self.pos_label)

            # --- get probability map for positive class (H,W) ---
            prob = None

            assert 'seg_logits' in sample and sample['seg_logits'] is not None, \
                'seg_logits missing: please ensure model runs in mode="predict" and returns seg_logits.'

            if "seg_logits" in sample and sample["seg_logits"] is not None:
                logits = sample["seg_logits"]["data"]

                # squeeze possible batch dim / singleton dims
                # common shapes: (C,H,W) or (1,C,H,W)
                if logits.ndim == 4 and logits.shape[0] == 1:
                    logits = logits.squeeze(0)
                logits = logits.float()

                if logits.ndim == 2:
                    # already (H,W) logits
                    prob = torch.sigmoid(logits) if self.use_sigmoid_if_one_channel else logits
                elif logits.ndim == 3:
                    c = logits.shape[0]
                    if c == 1 and self.use_sigmoid_if_one_channel:
                        prob = torch.sigmoid(logits[0])
                    else:
                        # softmax over channel, take positive channel
                        prob = torch.softmax(logits, dim=0)[self.pos_label]
                else:
                    raise RuntimeError(f"Unsupported seg_logits shape: {tuple(logits.shape)}")
            else:
                # Fallback: if only hard predictions are available, sweeping thresholds is meaningless.
                # But we still provide a degenerate prob map.
                pred = sample["pred_sem_seg"]["data"].squeeze().to(torch.int64)
                prob = pred.eq(self.pos_label).float()

            # ensure same spatial size and on CPU for stable accumulation
            prob = prob.detach().cpu()
            gt_pos = gt_pos.detach().cpu()
            valid = valid.detach().cpu()

            # --- vectorized per-threshold F1 for this image ---
            # pred_bin: (T,H,W)
            thr = self.thresholds[:, None, None]
            pred_bin = prob[None, :, :].gt(thr)  # prob > thresh

            # apply valid mask
            v = valid[None, :, :]
            tp = (pred_bin & gt_pos[None, :, :] & v).sum(dim=(1, 2)).to(torch.float32)
            fp = (pred_bin & (~gt_pos[None, :, :]) & v).sum(dim=(1, 2)).to(torch.float32)
            fn = ((~pred_bin) & gt_pos[None, :, :] & v).sum(dim=(1, 2)).to(torch.float32)

            # p_acc = 1.0 if tp==0 and fp==0 else tp/(tp+fp)  (evaluate.py) :contentReference[oaicite:11]{index=11}
            precision = torch.where(
                (tp == 0) & (fp == 0),
                torch.ones_like(tp),
                tp / torch.clamp(tp + fp, min=1.0),
            )
            recall = torch.where(
                (tp + fn) > 0,
                tp / torch.clamp(tp + fn, min=1.0),
                torch.zeros_like(tp),
            )
            f1 = torch.where(
                (precision + recall) > 0,
                2 * precision * recall / (precision + recall),
                torch.zeros_like(tp),
            )

            # store per-image f1 curve (for ODS & OIS)
            self.results.append({"f1_curve": f1.numpy()})

    def compute_metrics(self, results: list) -> Dict[str, float]:
        if len(results) == 0:
            return OrderedDict(ODS=0.0, OIS=0.0)

        f1_curves = np.stack([r["f1_curve"] for r in results], axis=0)  # (N,T)

        # OIS: mean over images of (max over thresholds)
        ois = float(np.mean(np.max(f1_curves, axis=1)))  # :contentReference[oaicite:12]{index=12}

        # ODS: max over thresholds of (mean over images)
        ods_curve = np.mean(f1_curves, axis=0)          # :contentReference[oaicite:13]{index=13}
        ods = float(np.max(ods_curve))

        return OrderedDict(ODS=ods, OIS=ois)
