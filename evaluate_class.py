"""
Class-aware text detection evaluation.

Expects the same prediction JSON layout as evaluate.py (image_name or file_name,
segmentation, score), plus category_id for the predicted class per detection.
Uses latin_eida ground truth and the class-aware matching / mAP from
notebook/detection_evaluation_latin.ipynb.

Optional: each entry may instead include "scores": [..] with length num_classes
(sigmoid probabilities), as produced by some exporters.
"""
import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from shapely.geometry import Polygon
from tqdm import tqdm

from datasets import build_dataset
from util.slconfig import SLConfig

LATIN_CLASSES_19 = [
    "word",
    "long",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "k",
    "m",
    "n",
    "o",
    "p",
    "q",
    "x",
    "L",
    "others",
]
LATIN_CLASSES_20 = LATIN_CLASSES_19 + ["symbol"]


def compute_iou(pred_poly, gt_poly):
    if torch.is_tensor(pred_poly):
        pred_poly = pred_poly.detach().cpu().numpy()
    if torch.is_tensor(gt_poly):
        gt_poly = gt_poly.detach().cpu().numpy()

    pred_poly = pred_poly.reshape(-1, 2)
    gt_poly = gt_poly.reshape(-1, 2)

    pred_shapely = Polygon(pred_poly)
    gt_shapely = Polygon(gt_poly)

    if not pred_shapely.is_valid:
        pred_shapely = pred_shapely.buffer(0)
    if not gt_shapely.is_valid:
        gt_shapely = gt_shapely.buffer(0)

    intersection = pred_shapely.intersection(gt_shapely).area
    union = pred_shapely.union(gt_shapely).area
    return intersection / union if union > 0 else 0


def match_predictions_independent_detections(
    pred_polygons: List[Any],
    pred_scores: np.ndarray,
    gt_polygons: List[Any],
    gt_labels: List[int],
    iou_threshold: float = 0.5,
) -> Dict[int, List[Dict]]:
    if torch.is_tensor(pred_scores):
        pred_scores = pred_scores.detach().cpu().numpy()
    if torch.is_tensor(pred_polygons):
        pred_polygons = pred_polygons.detach().cpu().numpy()

    num_classes = pred_scores.shape[-1]
    pred_scores = pred_scores.reshape(-1, num_classes)

    num_preds = pred_scores.shape[0]
    pred_polygons = pred_polygons.reshape(num_preds, -1)

    num_gts = len(gt_polygons)
    gt_labels_np = np.array(gt_labels, dtype=np.int32)

    iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float32)
    for i in range(num_preds):
        for j in range(num_gts):
            iou_matrix[i, j] = compute_iou(pred_polygons[i], gt_polygons[j])

    class_results = {c: [] for c in range(num_classes)}

    for class_id in range(num_classes):
        gt_indices_C = np.where(gt_labels_np == class_id)[0]
        gt_used_C = np.zeros(len(gt_indices_C), dtype=bool)

        detections_C = []
        for i in range(num_preds):
            score = pred_scores[i, class_id]
            if score > 1e-6:
                detections_C.append({"idx": i, "score": score})

        detections_C.sort(key=lambda x: x["score"], reverse=True)

        for det in detections_C:
            pred_idx = det["idx"]
            score = det["score"]

            matched_gt_idx = -1

            if len(gt_indices_C) > 0:
                iou_values = iou_matrix[pred_idx, gt_indices_C]
                sorted_gt_local_indices = np.argsort(iou_values)[::-1]

                for local_idx in sorted_gt_local_indices:
                    iou = iou_values[local_idx]
                    if iou < iou_threshold:
                        break
                    if not gt_used_C[local_idx]:
                        matched_gt_idx = local_idx
                        break

            if matched_gt_idx != -1:
                gt_used_C[matched_gt_idx] = True
                class_results[class_id].append(
                    {
                        "score": score,
                        "is_tp": 1,
                        "pred_idx": pred_idx,
                        "gt_idx": int(gt_indices_C[matched_gt_idx]),
                    }
                )
            else:
                class_results[class_id].append(
                    {
                        "score": score,
                        "is_tp": 0,
                        "pred_idx": pred_idx,
                        "gt_idx": -1,
                    }
                )

    return class_results


def calculate_interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(precision) == 0:
        return 0.0
    precision = np.asarray(precision, dtype=np.float64).copy()
    recall = np.asarray(recall, dtype=np.float64)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])
    r_to_trapz = np.concatenate(([0.0], recall))
    p_to_trapz = np.concatenate(([1.0], precision))
    return float(np.trapz(p_to_trapz, r_to_trapz))


def calculate_mAP(
    gt_instances_list,
    pred_instances_list,
    num_classes: int,
    iou_threshold: float = 0.5,
    image_set: str = "val",
    eval_threshold: float = None,
):
    total_gts_per_class = {i: 0 for i in range(num_classes)}
    class_eval_data = {i: [] for i in range(num_classes)}
    image_matches = []

    for gt, pred in zip(gt_instances_list, pred_instances_list):
        gt_labels = gt["labels"]
        if torch.is_tensor(gt_labels):
            gt_labels = gt_labels.cpu().numpy()
        for label in gt_labels:
            total_gts_per_class[int(label)] += 1

        gt_polys = gt["polygons"]
        pred_polys = pred["polygons"]
        pred_scores = pred["scores"]

        match_info = match_predictions_independent_detections(
            pred_polys,
            pred_scores,
            gt_polys,
            gt_labels,
            iou_threshold,
        )
        image_matches.append(match_info)

        for c in range(num_classes):
            for res in match_info[c]:
                class_eval_data[c].append(res)

    all_scores = []
    all_is_tp = []
    for c in range(num_classes):
        for res in class_eval_data[c]:
            all_scores.append(res["score"])
            all_is_tp.append(res["is_tp"])

    all_scores = np.array(all_scores, dtype=np.float64)
    all_is_tp = np.array(all_is_tp, dtype=np.float64)
    total_gt_global = sum(total_gts_per_class.values())

    if len(all_scores) == 0:
        ap_per_class = {c: 0.0 for c in range(num_classes)}
        f1_per_class = {c: 0.0 for c in range(num_classes)}
        return {
            "mAP": 0.0,
            "mF1": 0.0,
            "best_threshold": 0.0,
            "eval_threshold": eval_threshold if eval_threshold is not None else 0.0,
            "AP_per_class": ap_per_class,
            "F1_per_class": f1_per_class,
        }, image_matches

    indices = np.argsort(-all_scores)
    tp_cumsum = np.cumsum(all_is_tp[indices])
    fp_cumsum = np.cumsum(1 - all_is_tp[indices])

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / (total_gt_global + 1e-10)

    f1_curve = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = int(np.argmax(f1_curve))
    best_threshold = float(all_scores[indices][best_f1_idx])

    if eval_threshold is not None:
        th = eval_threshold
    elif image_set == "test":
        th = 0.3917
    else:
        th = best_threshold

    ap_per_class = {}
    f1_per_class = {}

    for c in range(num_classes):
        results = class_eval_data[c]
        total_gts = total_gts_per_class[c]

        if total_gts == 0:
            ap_per_class[c] = 0.0
            f1_per_class[c] = 0.0
            continue

        if len(results) == 0:
            ap_per_class[c] = 0.0
            f1_per_class[c] = 0.0
            continue

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        c_scores = np.array([r["score"] for r in results])
        c_tp = np.array([r["is_tp"] for r in results])

        c_tp_sum = np.cumsum(c_tp)
        c_fp_sum = np.cumsum(1 - c_tp)
        c_prec = c_tp_sum / (c_tp_sum + c_fp_sum + 1e-10)
        c_rec = c_tp_sum / total_gts
        ap_per_class[c] = calculate_interpolated_ap(c_rec, c_prec)

        top_dets = [r for r in results if r["score"] >= th]
        tp_count = sum(r["is_tp"] for r in top_dets)
        fp_count = len(top_dets) - tp_count

        p = tp_count / (tp_count + fp_count + 1e-10)
        r = tp_count / (total_gts + 1e-10)
        f1_per_class[c] = 2 * (p * r) / (p + r + 1e-10)

    return {
        "mAP": float(np.mean(list(ap_per_class.values()))),
        "mF1": float(np.mean(list(f1_per_class.values()))),
        "best_threshold": best_threshold,
        "eval_threshold": th,
        "AP_per_class": ap_per_class,
        "F1_per_class": f1_per_class,
    }, image_matches


def _segmentation_to_array(p: Dict) -> np.ndarray:
    seg = p["segmentation"]
    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
        seg = seg[0]
    return np.asarray(seg, dtype=np.float32).reshape(-1)


def _scores_from_prediction(p: Dict, num_classes: int) -> np.ndarray:
    """
    evaluate.py provides scalar "score". For class-aware matching, add
    "category_id" (predicted class, int). Alternatively pass "scores" of length num_classes.
    COCO-style "category_id" is supported; "label" is accepted as an alias.
    """
    if "scores" in p:
        v = np.array(p["scores"], dtype=np.float32).reshape(-1)
        if v.shape[0] != num_classes:
            raise ValueError(
                f'"scores" must have length {num_classes}, got {v.shape[0]}'
            )
        return v

    if "score" not in p:
        raise KeyError('Each prediction needs "score" (and "category_id"), or "scores".')

    cid = p.get("category_id", p.get("label"))
    if cid is None:
        raise KeyError(
            'Class-aware eval: add "category_id" (int) per detection, same JSON as '
            'evaluate.py otherwise; or provide a full "scores" vector.'
        )
    cid = int(cid)
    v = np.zeros(num_classes, dtype=np.float32)
    if 0 <= cid < num_classes:
        v[cid] = float(p["score"])
    return v


def main():
    parser = argparse.ArgumentParser(
        description="Class-aware eval: same JSON as evaluate.py (+ category_id), "
        "latin_eida GT, val/test mode."
    )
    parser.add_argument(
        "--results", type=str, required=True, help="Path to predictions JSON"
    )
    parser.add_argument("--config", type=str, default="config/latin_20class.py")
    parser.add_argument(
        "--mode", type=str, choices=["val", "test"], default="test"
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=2,
        help="[Det_Thresh unused_for_class_map RO_unused]: first value sets per-class "
        "F1 threshold in test mode (like a fixed operating point).",
    )
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument(
        "--print_per_class",
        action="store_true",
        help="Print AP and F1 for each class",
    )
    args = parser.parse_args()

    ds_args = SLConfig.fromfile(args.config)
    ds_args.fix_size = True
    ds_args.dataset_file = "latin_eida"
    num_classes = ds_args.num_classes

    if num_classes == 19:
        classes_name = LATIN_CLASSES_19
    else:
        classes_name = LATIN_CLASSES_20
    if len(classes_name) != num_classes:
        classes_name = [f"class_{i}" for i in range(num_classes)]

    dataset_val = build_dataset(
        image_set="test" if args.mode == "test" else "val",
        args=ds_args,
    )

    with open(args.results, "r") as f:
        raw_preds = json.load(f)

    preds_by_image = defaultdict(list)
    for p in raw_preds:
        key = p.get("image_name") or p.get("file_name")
        preds_by_image[key].append(p)

    gt_instances_list = []
    pred_instances_list = []

    eval_threshold = None
    if args.mode == "test" and args.thresholds is not None:
        eval_threshold = float(args.thresholds[0])

    print(
        f"Class-aware matching from {args.results} against {len(dataset_val)} GT images "
        f"({args.mode}, {num_classes} classes, IoU>={args.iou_thresh})..."
    )

    for i in tqdm(range(len(dataset_val))):
        _, labels = dataset_val[i]
        img_name = labels["name"]
        orig_size = labels["orig_size"]

        gt_polys = labels["boxes"].numpy()
        if gt_polys.max() <= 1.01:
            gt_polys = gt_polys * np.tile(
                [orig_size[0], orig_size[1]], gt_polys.shape[1] // 2
            )

        gt_labs = labels["labels"].numpy()

        plist = preds_by_image.get(img_name, [])
        pred_polys = []
        pred_score_rows = []

        for p in plist:
            pred_polys.append(_segmentation_to_array(p))
            pred_score_rows.append(_scores_from_prediction(p, num_classes))

        if len(pred_polys) == 0:
            poly_dim = gt_polys.shape[1] if len(gt_polys) else 64
            pred_arr = np.zeros((0, poly_dim), dtype=np.float32)
            score_arr = np.zeros((0, num_classes), dtype=np.float32)
        else:
            pred_arr = np.stack(pred_polys, axis=0)
            score_arr = np.stack(pred_score_rows, axis=0)

        gt_instances_list.append({"polygons": gt_polys, "labels": gt_labs})
        pred_instances_list.append({"polygons": pred_arr, "scores": score_arr})

    mAP_results, _ = calculate_mAP(
        gt_instances_list,
        pred_instances_list,
        num_classes=num_classes,
        iou_threshold=args.iou_thresh,
        image_set=args.mode,
        eval_threshold=eval_threshold,
    )

    th = mAP_results["eval_threshold"]
    print(f"\n--- Class-aware results ({args.mode.upper()}) ---")
    print(
        f"mAP@{args.iou_thresh:g}: {mAP_results['mAP']:.4f}, "
        f"mF1: {mAP_results['mF1']:.4f}"
    )
    print(f"Global best F1 threshold (score sweep): {mAP_results['best_threshold']:.4f}")
    print(f"Threshold used for per-class F1:        {th:.4f}")

    if args.print_per_class:
        print("\nPer-class AP / F1:")
        for c in range(num_classes):
            name = classes_name[c] if c < len(classes_name) else str(c)
            ap = mAP_results["AP_per_class"][c]
            f1 = mAP_results["F1_per_class"][c]
            print(f"  [{c:2d}] {name:8s}  AP: {ap:.4f}  F1: {f1:.4f}")


if __name__ == "__main__":
    main()
