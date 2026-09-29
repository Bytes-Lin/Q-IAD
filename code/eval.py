import json
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


GT_JSON = "./code/config/COCO_format_v3_test.json"
PRED_JSON = "./code/predict/predict_3cad_lora.json"


def xywh_to_xyxy(box):
    """COCO bbox [x, y, w, h] -> [x1, y1, x2, y2]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def bbox_iou(box1, box2):
    """IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0.0

    return inter / union


def build_iou_matrix(gt_boxes, pred_boxes):
    """Rows = GT, columns = predictions."""
    matrix = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float64)

    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            matrix[i, j] = bbox_iou(gt_box, pred_box)

    return matrix


# Matching for mIoU
def matched_iou_sum(gt_boxes, pred_boxes):
    """
    One-to-one Hungarian matching maximizing total IoU.

    mIoU is later computed as:
        total matched IoU / total number of GT boxes

    Therefore, an unmatched GT contributes IoU = 0.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    iou_matrix = build_iou_matrix(gt_boxes, pred_boxes)

    # linear_sum_assignment minimizes cost, so use -IoU
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    return float(iou_matrix[gt_indices, pred_indices].sum())


# Matching for Precision / Recall
def maximum_matching_at_threshold(gt_boxes, pred_boxes, threshold):
    """
    Maximum one-to-one bipartite matching under a given IoU threshold.

    Returns:
        TP, FP, FN
    """
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    if num_gt == 0:
        return 0, num_pred, 0

    if num_pred == 0:
        return 0, 0, num_gt

    iou_matrix = build_iou_matrix(gt_boxes, pred_boxes)

    # adjacency[pred_idx] = GT indices that this prediction can match
    adjacency = []
    for pred_idx in range(num_pred):
        valid_gt = [
            gt_idx
            for gt_idx in range(num_gt)
            if iou_matrix[gt_idx, pred_idx] >= threshold
        ]

        # Try higher-IoU GT first.
        valid_gt.sort(
            key=lambda gt_idx: iou_matrix[gt_idx, pred_idx],
            reverse=True
        )
        adjacency.append(valid_gt)

    # Standard maximum bipartite matching
    gt_matched_by = [-1] * num_gt

    def dfs(pred_idx, visited_gt):
        for gt_idx in adjacency[pred_idx]:
            if visited_gt[gt_idx]:
                continue

            visited_gt[gt_idx] = True

            if (
                    gt_matched_by[gt_idx] == -1
                    or dfs(gt_matched_by[gt_idx], visited_gt)
            ):
                gt_matched_by[gt_idx] = pred_idx
                return True

        return False

    tp = 0

    # Predictions with stronger possible matches are processed first.
    pred_order = sorted(
        range(num_pred),
        key=lambda p: max(
            [iou_matrix[g, p] for g in adjacency[p]],
            default=0.0
        ),
        reverse=True
    )

    for pred_idx in pred_order:
        visited_gt = [False] * num_gt
        if dfs(pred_idx, visited_gt):
            tp += 1

    fp = num_pred - tp
    fn = num_gt - tp

    return tp, fp, fn


with open(GT_JSON, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

with open(PRED_JSON, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

# Sort by (image_id, category_id)
gt_groups = defaultdict(list)
pred_groups = defaultdict(list)

gt_image_ids = {img["id"] for img in gt_data["images"]}

# Ground truth
for ann in gt_data["annotations"]:
    if ann.get("iscrowd", 0) == 1:
        continue

    key = (ann["image_id"], ann["category_id"])
    gt_groups[key].append(xywh_to_xyxy(ann["bbox"]))

# Predictions
for pred in pred_data:
    if pred["image_id"] not in gt_image_ids:
        continue

    key = (pred["image_id"], pred["category_id"])
    pred_groups[key].append(xywh_to_xyxy(pred["bbox"]))

all_keys = set(gt_groups.keys()) | set(pred_groups.keys())

# mIoU
total_gt_boxes = 0
total_matched_iou = 0.0

for key in all_keys:
    gt_boxes = gt_groups.get(key, [])
    pred_boxes = pred_groups.get(key, [])

    total_gt_boxes += len(gt_boxes)

    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        total_matched_iou += matched_iou_sum(gt_boxes, pred_boxes)

miou = (
    total_matched_iou / total_gt_boxes
    if total_gt_boxes > 0
    else 0.0
)


# Precision and Recall
def evaluate_at_threshold(threshold):
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for key in all_keys:
        gt_boxes = gt_groups.get(key, [])
        pred_boxes = pred_groups.get(key, [])

        tp, fp, fn = maximum_matching_at_threshold(
            gt_boxes,
            pred_boxes,
            threshold
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    return precision, recall


p50, r50 = evaluate_at_threshold(0.50)
p75, r75 = evaluate_at_threshold(0.75)

print(f"mIoU : {miou * 100:.2f}")
print(f"P50  : {p50 * 100:.2f}")
print(f"P75  : {p75 * 100:.2f}")
print(f"R50  : {r50 * 100:.2f}")
print(f"R75  : {r75 * 100:.2f}")
