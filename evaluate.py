import json
import math
import argparse
import numpy as np
import torch
from shapely.geometry import Polygon
from collections import defaultdict
from tqdm import tqdm

# Dataset imports
from datasets import build_dataset
from util.slconfig import SLConfig


def compute_iou(pred_poly, gt_poly):
    """Simple IoU computation between two polygons"""
    if torch.is_tensor(pred_poly):
        pred_poly = pred_poly.detach().cpu().numpy()
    if torch.is_tensor(gt_poly):
        gt_poly = gt_poly.detach().cpu().numpy()

    # Ensure they are [N, 2]
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


def match_predictions_with_gts(
    pred_polygons, pred_scores, gt_polygons, iou_threshold=0.5
):
    """Match predictions to ground truths using IoU"""
    num_preds = len(pred_polygons)
    num_gts = len(gt_polygons)

    iou_matrix = np.zeros((num_preds, num_gts))
    for i in range(num_preds):
        for j in range(num_gts):
            iou_matrix[i, j] = compute_iou(pred_polygons[i], gt_polygons[j])

    matched = []
    unmatched_preds = list(range(num_preds))
    unmatched_gts = list(range(num_gts))

    # Sort predictions by score descending
    pred_indices = np.argsort(-pred_scores)

    for pred_idx in pred_indices:
        if pred_idx in unmatched_preds:
            ious = iou_matrix[pred_idx]
            if len(ious) == 0:
                continue
            best_gt_idx = np.argmax(ious)
            best_iou = ious[best_gt_idx]

            if best_iou >= iou_threshold and best_gt_idx in unmatched_gts:
                matched.append((pred_idx, best_gt_idx))
                unmatched_preds.remove(pred_idx)
                unmatched_gts.remove(best_gt_idx)

    return matched, unmatched_preds, unmatched_gts


def is_reading_order_correct(pred_poly, gt_poly, key_indices):
    """Checks if corners match based on GT index mapping."""
    if torch.is_tensor(pred_poly):
        pred_poly = pred_poly.cpu().numpy()
    if torch.is_tensor(gt_poly):
        gt_poly = gt_poly.cpu().numpy()

    pred_coords = pred_poly.reshape(-1, 2)
    gt_coords = gt_poly.reshape(-1, 2)

    K = gt_coords.shape[0] // 2
    key_gt_indices = [0, K - 1, K, (2 * K) - 1]

    key_gt = {key_indices[0]: [key_gt_indices[0]],
              key_indices[1]: [key_gt_indices[1]],
              key_indices[2]: [key_gt_indices[2]],
              key_indices[3]: [key_gt_indices[3]]}
              
    local_score = 0.0
    for key_i in key_indices:
        pred_point = pred_coords[key_i]
        min_dist = float("inf")
        min_dist_idx = -1

        for j, gt_point in enumerate(gt_coords):
            if j not in key_gt_indices:
                continue
            dist = math.dist(pred_point, gt_point)
            if dist < min_dist:
                min_dist, min_dist_idx = dist, j

        if min_dist_idx in key_gt[key_i]:
            local_score += 1.0

    return 1 if (abs((local_score / 4.0) - 1.0) < 1e-9) else 0


def calculate_interpolated_ap(recall, precision):
    sort_indices = np.argsort(recall)
    recall = recall[sort_indices]
    precision = precision[sort_indices]
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])
    return np.trapz(precision, recall)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate with Prediction JSON and Dataset GTs"
    )
    parser.add_argument(
        "--results", type=str, required=True, help="Path to predictions JSON"
    )
    parser.add_argument("--config", type=str, default="config/finetuning.py")
    parser.add_argument("--mode", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--key_indices", type=int, nargs=4, default=[0, 24, 25, 49])
    parser.add_argument(
        "--thresholds", type=float, nargs=2, help="[Det_Thresh RO_Thresh]"
    )
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    args = parser.parse_args()

    # 1. Build the dataset class to get Ground Truths
    ds_args = SLConfig.fromfile(args.config)
    ds_args.fix_size = True
    ds_args.dataset_file = "eida"
    dataset_val = build_dataset(
        image_set="test" if args.mode == "test" else "val", args=ds_args
    )

    # 2. Load the Prediction JSON file
    with open(args.results, "r") as f:
        raw_preds = json.load(f)

    # Group predictions by image name for fast lookup
    preds_by_image = defaultdict(list)
    for p in raw_preds:
        # Use image_name or image_id depending on your JSON format
        key = p.get("image_name") or p.get("file_name")
        preds_by_image[key].append(p)

    all_scores, all_matches, all_ro_matches = [], [], []
    total_gt = 0

    print(
        f"Matching predictions from {args.results} against {len(dataset_val)} GT images..."
    )

    # 3. Iterate through Dataset to match GTs with the loaded Predictions
    for i in tqdm(range(len(dataset_val))):
        _, labels = dataset_val[i]
        img_name = labels["name"]
        orig_size = labels["orig_size"]  # [W, H]

        # Get and Scale GT Polygons
        gt_polys = labels["boxes"].numpy()
        if gt_polys.max() <= 1.01:
            # Scale normalized coordinates [0, 1] to [W, H]
            # Assuming boxes are [x1, y1, x2, y2 ... x32, y32]
            gt_polys = gt_polys * np.tile([orig_size[0], orig_size[1]], 32)

        total_gt += len(gt_polys)

        # Retrieve predictions for THIS image from the loaded JSON
        current_pred_polys = []
        current_pred_scores = []

        if img_name in preds_by_image:
            for p in preds_by_image[img_name]:
                seg = p["segmentation"]
                if isinstance(seg[0], list):
                    seg = seg[0]
                current_pred_polys.append(np.array(seg).reshape(-1, 2))
                current_pred_scores.append(p["score"])

        if len(current_pred_polys) == 0:
            continue

        # 4. Perform IoU-based matching
        matched, unmatched_preds, unmatched_gts = match_predictions_with_gts(
            current_pred_polys, np.array(current_pred_scores), gt_polys, args.iou_thresh
        )

        for p_idx, g_idx in matched:
            all_scores.append(current_pred_scores[p_idx])
            all_matches.append(1)
            all_ro_matches.append(
                is_reading_order_correct(
                    current_pred_polys[p_idx], gt_polys[g_idx], args.key_indices
                )
            )

        for p_idx in unmatched_preds:
            all_scores.append(current_pred_scores[p_idx])
            all_matches.append(0)
            all_ro_matches.append(0)

    # 5. Compute Metrics
    all_scores = np.array(all_scores)
    sorted_idx = np.argsort(-all_scores)
    sorted_scores = all_scores[sorted_idx]
    all_matches = np.array(all_matches)[sorted_idx]
    all_ro_matches = np.array(all_ro_matches)[sorted_idx]

    tp_cumsum = np.cumsum(all_matches)
    precision = tp_cumsum / np.arange(1, len(all_matches) + 1)
    recall = tp_cumsum / total_gt

    tp_cumsum_ro = np.cumsum(all_ro_matches)
    precision_ro = tp_cumsum_ro / np.arange(1, len(all_ro_matches) + 1)
    recall_ro = tp_cumsum_ro / total_gt

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    f1_scores_ro = 2 * (precision_ro * recall_ro) / (precision_ro + recall_ro + 1e-9)

    # Threshold selection logic
    if args.mode == "test" and args.thresholds:
        det_thresh, ro_thresh = args.thresholds
        idx = (
            np.where(sorted_scores >= det_thresh)[0][-1]
            if any(sorted_scores >= det_thresh)
            else 0
        )
        idx_ro = (
            np.where(sorted_scores >= ro_thresh)[0][-1]
            if any(sorted_scores >= ro_thresh)
            else 0
        )
    else:
        # Find optimal thresholds for val mode (or if no thresholds provided for test)
        idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
        idx_ro = np.argmax(f1_scores_ro) if len(f1_scores_ro) > 0 else 0

        det_thresh = sorted_scores[idx] if len(sorted_scores) > 0 else 0.0
        ro_thresh = sorted_scores[idx_ro] if len(sorted_scores) > 0 else 0.0

    print(f"\n--- Evaluation Results ({args.mode.upper()}) ---")
    if args.mode == "val":
        print(f"Optimal Threshold for F1:    {det_thresh:.4f}")
        print(f"Optimal Threshold for F1-RO: {ro_thresh:.4f}")
        print("-" * 30)

    print(
        f"F1:    {f1_scores[idx]:.4f} (Prec: {precision[idx]:.4f}, Rec: {recall[idx]:.4f}) @ Thresh: {det_thresh:.4f}"
    )
    print(
        f"F1-RO: {f1_scores_ro[idx_ro]:.4f} (Prec: {precision_ro[idx_ro]:.4f}, Rec: {recall_ro[idx_ro]:.4f}) @ Thresh: {ro_thresh:.4f}"
    )

    print(f"\nmAP@50:    {calculate_interpolated_ap(recall, precision):.4f}")
    print(f"mAP-RO@50: {calculate_interpolated_ap(recall_ro, precision_ro):.4f}")


if __name__ == "__main__":
    main()
