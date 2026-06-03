from __future__ import annotations

import gc
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, auc, roc_auc_score
from skimage.measure import label as cc_label


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_auroc_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(bool, copy=False)
    n_pos = int(y_true.sum())
    n_total = int(y_true.size)
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    try:
        order = np.argsort(y_score, kind="mergesort")
        sorted_scores = y_score[order]
        sorted_true = y_true[order]
        starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        ends = np.r_[starts[1:], n_total]
        pos_cumsum = np.r_[0.0, np.cumsum(sorted_true, dtype=np.float64)]
        pos_per_group = pos_cumsum[ends] - pos_cumsum[starts]
        avg_ranks = (starts.astype(np.float64) + 1.0 + ends.astype(np.float64)) * 0.5
        sum_pos_ranks = float(np.sum(pos_per_group * avg_ranks))
        return float((sum_pos_ranks - n_pos * (n_pos + 1.0) * 0.5) / (n_pos * n_neg))
    except MemoryError:
        return binary_auroc_hist(y_true, y_score)


def binary_auroc_hist(y_true: np.ndarray, y_score: np.ndarray, bins: int = 16384) -> float:
    y_true = y_true.astype(bool, copy=False)
    n_pos = int(y_true.sum())
    n_total = int(y_true.size)
    n_neg = n_total - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    finite = np.isfinite(y_score)
    if not bool(finite.all()):
        y_true = y_true[finite]
        y_score = y_score[finite]
    score_min = float(np.min(y_score))
    score_max = float(np.max(y_score))
    if score_max <= score_min:
        return 0.5
    edges = np.linspace(score_min, score_max, bins + 1, dtype=np.float32)
    pos_hist, _ = np.histogram(y_score[y_true], bins=edges)
    neg_hist, _ = np.histogram(y_score[~y_true], bins=edges)
    tp = np.cumsum(pos_hist[::-1], dtype=np.float64)
    fp = np.cumsum(neg_hist[::-1], dtype=np.float64)
    tpr = np.r_[0.0, tp / max(n_pos, 1)]
    fpr = np.r_[0.0, fp / max(n_neg, 1)]
    return float(auc(fpr, tpr))


def pixel_auroc_hist_stream(masks: List[np.ndarray], maps: List[np.ndarray], bins: int = 65536) -> float:
    pos_total = int(sum(int(m.astype(bool).sum()) for m in masks))
    total = int(sum(int(m.size) for m in masks))
    neg_total = total - pos_total
    if pos_total == 0 or neg_total == 0:
        return float("nan")
    score_min = min(float(np.min(score)) for score in maps)
    score_max = max(float(np.max(score)) for score in maps)
    if score_max <= score_min:
        return 0.5
    edges = np.linspace(score_min, score_max, bins + 1, dtype=np.float32)
    pos_hist = np.zeros(bins, dtype=np.int64)
    all_hist = np.zeros(bins, dtype=np.int64)
    for gt, score in zip(masks, maps):
        all_hist += np.histogram(score.reshape(-1), bins=edges)[0]
        gt_bool = gt.astype(bool, copy=False)
        if bool(gt_bool.any()):
            pos_hist += np.histogram(score[gt_bool], bins=edges)[0]
    neg_hist = all_hist - pos_hist
    tp = np.cumsum(pos_hist[::-1], dtype=np.float64)
    fp = np.cumsum(neg_hist[::-1], dtype=np.float64)
    tpr = np.r_[0.0, tp / max(pos_total, 1)]
    fpr = np.r_[0.0, fp / max(neg_total, 1)]
    return float(auc(fpr, tpr))


def hist_quantile_thresholds(
    maps: List[np.ndarray],
    num_thresholds: int,
    bins: int = 16384,
) -> np.ndarray:
    score_min = min(float(np.min(score)) for score in maps)
    score_max = max(float(np.max(score)) for score in maps)
    if score_max <= score_min:
        return np.array([], dtype=np.float32)
    edges = np.linspace(score_min, score_max, bins + 1, dtype=np.float32)
    hist = np.zeros(bins, dtype=np.int64)
    for score in maps:
        hist += np.histogram(score.reshape(-1), bins=edges)[0]
    total = int(hist.sum())
    if total == 0:
        return np.array([], dtype=np.float32)
    cdf = np.cumsum(hist, dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, num_thresholds)
    ranks = np.clip(quantiles * max(total - 1, 1) + 1, 1, total)
    idx = np.searchsorted(cdf, ranks, side="left")
    idx = np.clip(idx, 0, bins - 1)
    return np.unique(edges[idx])[::-1].astype(np.float32)


def flatten_float32(arrays: List[np.ndarray]) -> np.ndarray:
    total = sum(int(arr.size) for arr in arrays)
    out = np.empty(total, dtype=np.float32)
    offset = 0
    for arr in arrays:
        size = int(arr.size)
        out[offset : offset + size] = arr.reshape(-1).astype(np.float32, copy=False)
        offset += size
    return out


def flatten_bool(arrays: List[np.ndarray]) -> np.ndarray:
    total = sum(int(arr.size) for arr in arrays)
    out = np.empty(total, dtype=np.bool_)
    offset = 0
    for arr in arrays:
        size = int(arr.size)
        out[offset : offset + size] = arr.reshape(-1).astype(bool, copy=False)
        offset += size
    return out


def component_labels_flat(masks: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    total = sum(int(mask.size) for mask in masks)
    labels_flat = np.zeros(total, dtype=np.int32)
    areas = [0]
    normal_total = 0
    offset = 0
    comp_id = 0
    for gt in masks:
        gt_bool = gt.astype(bool, copy=False)
        normal_total += int((~gt_bool).sum())
        labeled = cc_label(gt_bool, connectivity=2)
        flat = labels_flat[offset : offset + gt_bool.size].reshape(gt_bool.shape)
        for idx in range(1, int(labeled.max()) + 1):
            comp = labeled == idx
            area = int(comp.sum())
            if area > 0:
                comp_id += 1
                flat[comp] = comp_id
                areas.append(area)
        offset += gt_bool.size
    return labels_flat, np.asarray(areas, dtype=np.float32), normal_total


def auc_from_hist(pos_hist: torch.Tensor, neg_hist: torch.Tensor) -> float:
    pos_total = torch.clamp(pos_hist.sum(), min=1.0)
    neg_total = torch.clamp(neg_hist.sum(), min=1.0)
    tp = torch.cumsum(torch.flip(pos_hist, dims=(0,)), dim=0)
    fp = torch.cumsum(torch.flip(neg_hist, dims=(0,)), dim=0)
    tpr = torch.cat([torch.zeros(1, device=tp.device), tp / pos_total])
    fpr = torch.cat([torch.zeros(1, device=fp.device), fp / neg_total])
    return float(torch.trapz(tpr, fpr).detach().cpu().item())


def thresholds_from_hist(
    all_hist: torch.Tensor,
    score_min: float,
    score_max: float,
    num_thresholds: int,
) -> np.ndarray:
    hist = all_hist.detach().cpu().numpy().astype(np.int64, copy=False)
    total = int(hist.sum())
    if total == 0:
        return np.array([], dtype=np.float32)
    edges = np.linspace(score_min, score_max, hist.shape[0] + 1, dtype=np.float32)
    cdf = np.cumsum(hist, dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, num_thresholds)
    ranks = np.clip(quantiles * max(total - 1, 1) + 1, 1, total)
    idx = np.searchsorted(cdf, ranks, side="left")
    idx = np.clip(idx, 0, hist.shape[0] - 1)
    return np.unique(edges[idx])[::-1].astype(np.float32)


def pixel_metrics_gpu(
    masks: List[np.ndarray],
    maps: List[np.ndarray],
    device: str = "cuda",
    max_fpr: float = 0.30,
    num_thresholds: int = 200,
    auroc_bins: int = 65536,
) -> Dict[str, float]:
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        return pixel_metrics_cpu(masks, maps, max_fpr=max_fpr, num_thresholds=num_thresholds)

    try:
        positive_total = int(sum(int(mask.astype(bool, copy=False).sum()) for mask in masks))
        normal_total = int(sum(int((~mask.astype(bool, copy=False)).sum()) for mask in masks))
        if positive_total == 0 or normal_total == 0:
            return {"pixel_auroc": float("nan"), "pro": float("nan")}

        score_min = min(float(np.min(score)) for score in maps)
        score_max = max(float(np.max(score)) for score in maps)
        if score_max <= score_min:
            return {"pixel_auroc": 0.5, "pro": 0.0}

        all_hist = torch.zeros(auroc_bins, device=dev, dtype=torch.float32)
        pos_hist = torch.zeros(auroc_bins, device=dev, dtype=torch.float32)
        for gt_np, score_np in zip(masks, maps):
            score = torch.from_numpy(np.ascontiguousarray(score_np.reshape(-1))).to(dev, non_blocking=True)
            gt = torch.from_numpy(np.ascontiguousarray(gt_np.reshape(-1).astype(bool, copy=False))).to(dev, non_blocking=True)
            all_hist += torch.histc(score, bins=auroc_bins, min=score_min, max=score_max)
            if bool(gt.any()):
                pos_hist += torch.histc(score[gt], bins=auroc_bins, min=score_min, max=score_max)
            del score, gt
        neg_hist = torch.clamp(all_hist - pos_hist, min=0)
        pixel_auroc = auc_from_hist(pos_hist, neg_hist)

        thresholds = thresholds_from_hist(all_hist, score_min, score_max, num_thresholds)
        if thresholds.size < 2:
            return {"pixel_auroc": pixel_auroc, "pro": float("nan")}

        threshold_t = torch.from_numpy(thresholds.astype(np.float32, copy=False)).to(dev)
        fp_counts = torch.zeros(threshold_t.numel(), device=dev, dtype=torch.float64)
        overlap_sums = torch.zeros(threshold_t.numel(), device=dev, dtype=torch.float64)
        comp_count_total = 0
        for gt_np, score_np in zip(masks, maps):
            gt_bool = gt_np.astype(bool, copy=False)
            labeled = cc_label(gt_bool, connectivity=2).astype(np.int32, copy=False)
            comp_count = int(labeled.max())
            if comp_count > 0:
                areas_np = np.bincount(labeled.reshape(-1), minlength=comp_count + 1).astype(np.float32)
                comp_count_total += comp_count
                comp_ids = torch.from_numpy(np.ascontiguousarray(labeled.reshape(-1))).to(dev, non_blocking=True)
                comp_area = torch.from_numpy(areas_np).to(dev, non_blocking=True)
                comp_positive = comp_ids > 0
            else:
                comp_ids = None
                comp_area = None
                comp_positive = None
            score = torch.from_numpy(np.ascontiguousarray(score_np.reshape(-1))).to(dev, non_blocking=True)
            normal = torch.from_numpy(np.ascontiguousarray((~gt_bool).reshape(-1))).to(dev, non_blocking=True)
            for idx, th in enumerate(threshold_t):
                pred = score >= th
                fp_counts[idx] += torch.count_nonzero(pred & normal).double()
                if comp_ids is not None and comp_area is not None and comp_positive is not None:
                    selected = comp_ids[pred & comp_positive].long()
                    if selected.numel() > 0:
                        hits = torch.bincount(selected, minlength=comp_count + 1).double()
                        overlap_sums[idx] += torch.sum(hits[1:] / comp_area[1:].double())
                del pred
            del score, normal, comp_ids, comp_area, comp_positive

        if comp_count_total == 0:
            return {"pixel_auroc": pixel_auroc, "pro": float("nan")}
        fprs_all = (fp_counts / float(normal_total)).detach().cpu().numpy()
        pros_all = (overlap_sums / float(comp_count_total)).detach().cpu().numpy()
        keep = fprs_all <= max_fpr
        if int(np.count_nonzero(keep)) < 2:
            pro_value = 0.0
        else:
            points = sorted(zip(fprs_all[keep].tolist(), pros_all[keep].tolist()), key=lambda x: x[0])
            xs = np.array([0.0] + [p[0] for p in points], dtype=np.float32)
            ys = np.array([0.0] + [p[1] for p in points], dtype=np.float32)
            if xs[-1] < max_fpr:
                xs = np.append(xs, max_fpr)
                ys = np.append(ys, ys[-1])
            pro_value = float(auc(xs, ys) / max_fpr)
        return {"pixel_auroc": pixel_auroc, "pro": pro_value}
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        return pixel_metrics_cpu(masks, maps, max_fpr=max_fpr, num_thresholds=num_thresholds)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pixel_metrics_cpu(
    masks: List[np.ndarray],
    maps: List[np.ndarray],
    max_fpr: float = 0.30,
    num_thresholds: int = 200,
) -> Dict[str, float]:
    pixel_auroc = pixel_auroc_hist_stream(masks, maps)
    gc.collect()
    return {
        "pixel_auroc": pixel_auroc,
        "pro": pro_auc(masks, maps, max_fpr=max_fpr, num_thresholds=num_thresholds),
    }


def pixel_metrics(
    masks: List[np.ndarray],
    maps: List[np.ndarray],
    device: str = "cuda",
    max_fpr: float = 0.30,
    num_thresholds: int = 200,
) -> Dict[str, float]:
    return pixel_metrics_gpu(masks, maps, device=device, max_fpr=max_fpr, num_thresholds=num_thresholds)


def pro_auc(
    masks: List[np.ndarray],
    maps: List[np.ndarray],
    max_fpr: float = 0.30,
    num_thresholds: int = 200,
) -> float:
    thresholds = hist_quantile_thresholds(maps, num_thresholds)
    if thresholds.size < 2:
        return float("nan")

    components_by_image = []
    normal_total = 0
    for gt in masks:
        gt_bool = gt.astype(bool)
        normal_total += int((~gt_bool).sum())
        labeled = cc_label(gt_bool, connectivity=2)
        image_components = []
        for idx in range(1, labeled.max() + 1):
            comp = labeled == idx
            area = int(comp.sum())
            if area > 0:
                image_components.append((comp, area))
        components_by_image.append((gt_bool, image_components))
    if not any(components for _, components in components_by_image) or normal_total == 0:
        return float("nan")

    fprs = []
    pros = []
    for th in thresholds:
        fp = 0
        overlap_sum = 0.0
        comp_count = 0
        for (gt_bool, _components), score in zip(components_by_image, maps):
            pred = score >= th
            fp += int(np.logical_and(pred, ~gt_bool).sum())
        for (_gt_bool, components), score in zip(components_by_image, maps):
            pred = score >= th
            for comp, area in components:
                overlap_sum += float(np.logical_and(pred, comp).sum()) / float(area)
                comp_count += 1
        fpr = fp / float(normal_total)
        if fpr <= max_fpr:
            fprs.append(fpr)
            pros.append(overlap_sum / max(comp_count, 1))

    if len(fprs) < 2:
        return 0.0
    points = sorted(zip(fprs, pros), key=lambda x: x[0])
    xs = np.array([0.0] + [p[0] for p in points], dtype=np.float32)
    ys = np.array([0.0] + [p[1] for p in points], dtype=np.float32)
    if xs[-1] < max_fpr:
        xs = np.append(xs, max_fpr)
        ys = np.append(ys, ys[-1])
    return float(auc(xs, ys) / max_fpr)
