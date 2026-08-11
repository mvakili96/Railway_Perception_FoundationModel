import argparse
import csv
import json
import os
import re

import cv2
import numpy as np
from tqdm import tqdm


# =========================
# CONFIG
# =========================
GT_JSON_PATH = "dataset/test/rs19_egopath_1024.json"

# Use "single_model" for one-model confidence intervals, or
# "paired_comparison" to compare the two prediction directories below.
EVALUATION_MODE = "single_model"
METHOD_A_NAME = "Joint training"
PRED_MASK_DIR = "outputs/test/joint_reasoning/masks"
METHOD_B_NAME = "Segmentation only"
COMPARISON_PRED_MASK_DIR = "outputs/test/semantic_reasoning/masks"

AUDIT_CSV_PATH = "scripts/evaluation/metadata/route_logic_audit_30.csv"
EMPTY_MASK_THRESHOLD = 50
MULTI_TRACK_IOU_THRESHOLD = 0.90
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RANDOM_SEED = 2026


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ego-path masks with the tested CIoU, GIoU, N-acc, "
            "threshold, and paired-bootstrap calculations."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("single_model", "paired_comparison"),
        default=EVALUATION_MODE,
        help="Evaluate one prediction directory or compare two paired runs.",
    )
    parser.add_argument("--gt-json", default=GT_JSON_PATH)
    parser.add_argument("--predictions", default=PRED_MASK_DIR)
    parser.add_argument("--method-name", default=METHOD_A_NAME)
    parser.add_argument(
        "--comparison-predictions",
        default=COMPARISON_PRED_MASK_DIR,
    )
    parser.add_argument(
        "--comparison-method-name",
        default=METHOD_B_NAME,
    )
    parser.add_argument(
        "--audit-csv",
        default=AUDIT_CSV_PATH,
        help="CSV containing the 30 route-logic audit image indexes.",
    )
    return parser.parse_args()


# =========================
# GT PROCESSING
# =========================

def load_gt(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def load_multi_track_indexes(csv_path):
    """Load the image indexes belonging to the route-logic audit subset."""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "image_index" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain an 'image_index' column")

        return {int(row["image_index"]) for row in reader}


def extract_image_index(image_name):
    """Convert an RS19 image name such as rs06044.jpg to integer index 6044."""
    match = re.fullmatch(r"rs(\d+)\.jpg", image_name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot extract image index from {image_name}")
    return int(match.group(1))


def interpolate_rail(points, target_ys):
    pts = np.array(points, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # sort bottom → top (descending y)
    order = np.argsort(-ys)
    xs = xs[order]
    ys = ys[order]

    # flip for np.interp (ascending)
    ys_sorted = ys[::-1]
    xs_sorted = xs[::-1]

    interp_x = np.interp(target_ys, ys_sorted, xs_sorted)
    return interp_x


def rails_to_polygon(left_pts, right_pts):
    left = np.array(left_pts, dtype=np.float32)
    right = np.array(right_pts, dtype=np.float32)

    # overlap y-range only
    min_y = int(max(left[:,1].min(), right[:,1].min()))
    max_y = int(min(left[:,1].max(), right[:,1].max()))

    ys = np.arange(max_y, min_y - 1, -1)

    left_x = interpolate_rail(left_pts, ys)
    right_x = interpolate_rail(right_pts, ys)

    left_interp = np.stack([left_x, ys], axis=1)
    right_interp = np.stack([right_x, ys], axis=1)

    polygon = np.vstack([left_interp, right_interp[::-1]]).astype(np.int32)
    return polygon


def polygon_to_mask(polygon, shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def rails_annotation_to_mask(annotation, shape):
    left = annotation.get("left_rail", [])
    right = annotation.get("right_rail", [])

    if len(left) == 0 or len(right) == 0:
        return np.zeros(shape[:2], dtype=np.uint8)

    polygon = rails_to_polygon(left, right)
    return polygon_to_mask(polygon, shape)


# =========================
# PRED PROCESSING
# =========================

def load_pred_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_COLOR)
    if mask is None:
        raise FileNotFoundError(path)

    mask = mask[:, :, 0]  # any channel
    mask = (mask == 100).astype(np.uint8)

    # cv2.imshow("mask_pred", mask*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(GOOOz)

    return mask


# =========================
# METRIC
# =========================

def compute_intersection_union(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection, union


def compute_iou(pred, gt):
    intersection, union = compute_intersection_union(pred, gt)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def is_empty_prediction(mask, threshold=EMPTY_MASK_THRESHOLD):
    return int(mask.sum()) < threshold


def new_metric_accumulator():
    return {
        "ious": [],
        "intersections": [],
        "unions": [],
        "false_positive_pixels": [],
        "empty_targets": [],
        "empty_target_corrects": [],
        "total_intersection": 0,
        "total_union": 0,
        "empty_target_total": 0,
        "empty_target_correct": 0,
    }


def compute_sample_statistics(pred_mask, gt_mask):
    """Compute one sample's contributions to all reported metrics."""
    gt_is_empty = gt_mask.sum() == 0
    pred_is_empty = is_empty_prediction(pred_mask)

    if gt_is_empty:
        false_positive_pixels = int(pred_mask.sum())
        return {
            "iou": 1.0 if pred_is_empty else 0.0,
            "intersection": 0,
            "union": 0 if pred_is_empty else false_positive_pixels,
            "false_positive_pixels": false_positive_pixels,
            "empty_target": 1,
            "empty_target_correct": int(pred_is_empty),
        }

    intersection, union = compute_intersection_union(pred_mask, gt_mask)
    false_positive_pixels = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    return {
        "iou": intersection / union if union > 0 else 0.0,
        "intersection": int(intersection),
        "union": int(union),
        "false_positive_pixels": int(false_positive_pixels),
        "empty_target": 0,
        "empty_target_correct": 0,
    }


def add_sample_statistics(accumulator, sample_statistics):
    accumulator["ious"].append(sample_statistics["iou"])
    accumulator["intersections"].append(sample_statistics["intersection"])
    accumulator["unions"].append(sample_statistics["union"])
    accumulator["false_positive_pixels"].append(
        sample_statistics.get("false_positive_pixels", 0)
    )
    accumulator["empty_targets"].append(sample_statistics["empty_target"])
    accumulator["empty_target_corrects"].append(
        sample_statistics["empty_target_correct"]
    )
    accumulator["total_intersection"] += sample_statistics["intersection"]
    accumulator["total_union"] += sample_statistics["union"]
    accumulator["empty_target_total"] += sample_statistics["empty_target"]
    accumulator["empty_target_correct"] += sample_statistics["empty_target_correct"]


def percentile_confidence_interval(values, confidence_level):
    """Return the central percentile interval for a bootstrap distribution."""
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.percentile(values, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lower), float(upper)


def bootstrap_mean_distribution(values, resamples, rng):
    """Bootstrap a sample mean without constructing one large index matrix."""
    values = np.asarray(values, dtype=np.float64)
    sample_count = len(values)
    distribution = np.empty(resamples, dtype=np.float64)

    # Keep temporary bootstrap index arrays near or below one million entries.
    batch_size = max(1, min(resamples, 1_000_000 // sample_count))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indexes = rng.integers(
            0,
            sample_count,
            size=(stop - start, sample_count),
        )
        distribution[start:stop] = values[indexes].mean(axis=1)

    return distribution


def paired_bootstrap_confidence_intervals(
    accumulator,
    resamples=BOOTSTRAP_RESAMPLES,
    confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
    random_seed=BOOTSTRAP_RANDOM_SEED,
):
    """Bootstrap matched prediction/GT image pairs and return metric intervals.

    Each sampled unit is one image's matched prediction and ground truth. The
    mIoU and cIoU distributions use the same resampled image indexes so that
    every image-level prediction/GT pair remains intact. N-acc is bootstrapped
    over the subset of eligible images whose ground-truth masks are empty.
    """
    if resamples <= 0:
        raise ValueError("BOOTSTRAP_RESAMPLES must be greater than zero")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("BOOTSTRAP_CONFIDENCE_LEVEL must be between 0 and 1")

    sample_count = len(accumulator["ious"])
    empty_intervals = {
        "mIoU": None,
        "cIoU": None,
        "gIoU": None,
        "N-acc": None,
    }
    if sample_count == 0:
        return empty_intervals

    rng = np.random.default_rng(random_seed)
    ious = np.asarray(accumulator["ious"], dtype=np.float64)
    intersections = np.asarray(accumulator["intersections"], dtype=np.int64)
    unions = np.asarray(accumulator["unions"], dtype=np.int64)
    miou_distribution = np.empty(resamples, dtype=np.float64)
    ciou_distribution = np.empty(resamples, dtype=np.float64)

    # Process the resamples in bounded batches to avoid a large memory spike on
    # the single-track and overall subsets.
    batch_size = max(1, min(resamples, 1_000_000 // sample_count))
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indexes = rng.integers(
            0,
            sample_count,
            size=(stop - start, sample_count),
        )

        miou_distribution[start:stop] = ious[indexes].mean(axis=1)
        sampled_intersections = intersections[indexes].sum(axis=1)
        sampled_unions = unions[indexes].sum(axis=1)
        sampled_ciou = np.ones(stop - start, dtype=np.float64)
        nonzero_union = sampled_unions > 0
        sampled_ciou[nonzero_union] = (
            sampled_intersections[nonzero_union]
            / sampled_unions[nonzero_union]
        )
        sampled_ciou[~nonzero_union & (sampled_intersections != 0)] = 0.0
        ciou_distribution[start:stop] = sampled_ciou

    miou_interval = percentile_confidence_interval(
        miou_distribution,
        confidence_level,
    )
    ciou_interval = percentile_confidence_interval(
        ciou_distribution,
        confidence_level,
    )

    empty_targets = np.asarray(accumulator["empty_targets"], dtype=bool)
    empty_target_corrects = np.asarray(
        accumulator["empty_target_corrects"],
        dtype=np.float64,
    )[empty_targets]
    n_acc_interval = None
    if len(empty_target_corrects) > 0:
        n_acc_distribution = bootstrap_mean_distribution(
            empty_target_corrects,
            resamples,
            rng,
        )
        n_acc_interval = percentile_confidence_interval(
            n_acc_distribution,
            confidence_level,
        )

    return {
        "mIoU": miou_interval,
        "cIoU": ciou_interval,
        "gIoU": miou_interval,
        "N-acc": n_acc_interval,
    }


def paired_bootstrap_difference_confidence_intervals(
    accumulator_a,
    accumulator_b,
    resamples=BOOTSTRAP_RESAMPLES,
    confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
    random_seed=BOOTSTRAP_RANDOM_SEED,
):
    """Return paired-bootstrap CIs for Method A minus Method B.

    The two accumulators must contain the same images in the same order. A
    single bootstrap index matrix is applied to both methods, preserving the
    image-level pairing when calculating cIoU and gIoU differences.
    """
    if resamples <= 0:
        raise ValueError("BOOTSTRAP_RESAMPLES must be greater than zero")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("BOOTSTRAP_CONFIDENCE_LEVEL must be between 0 and 1")

    sample_count = len(accumulator_a["ious"])
    if sample_count != len(accumulator_b["ious"]):
        raise ValueError("Paired accumulators must contain the same number of images")
    if sample_count == 0:
        return {"cIoU": None, "gIoU": None}

    rng = np.random.default_rng(random_seed)
    ious_a = np.asarray(accumulator_a["ious"], dtype=np.float64)
    ious_b = np.asarray(accumulator_b["ious"], dtype=np.float64)
    intersections_a = np.asarray(
        accumulator_a["intersections"],
        dtype=np.int64,
    )
    intersections_b = np.asarray(
        accumulator_b["intersections"],
        dtype=np.int64,
    )
    unions_a = np.asarray(accumulator_a["unions"], dtype=np.int64)
    unions_b = np.asarray(accumulator_b["unions"], dtype=np.int64)

    giou_difference_distribution = np.empty(resamples, dtype=np.float64)
    ciou_difference_distribution = np.empty(resamples, dtype=np.float64)
    batch_size = max(1, min(resamples, 1_000_000 // sample_count))

    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indexes = rng.integers(
            0,
            sample_count,
            size=(stop - start, sample_count),
        )

        giou_a = ious_a[indexes].mean(axis=1)
        giou_b = ious_b[indexes].mean(axis=1)
        giou_difference_distribution[start:stop] = giou_a - giou_b

        sampled_intersections_a = intersections_a[indexes].sum(axis=1)
        sampled_intersections_b = intersections_b[indexes].sum(axis=1)
        sampled_unions_a = unions_a[indexes].sum(axis=1)
        sampled_unions_b = unions_b[indexes].sum(axis=1)

        ciou_a = np.ones(stop - start, dtype=np.float64)
        ciou_b = np.ones(stop - start, dtype=np.float64)
        nonzero_union_a = sampled_unions_a > 0
        nonzero_union_b = sampled_unions_b > 0
        ciou_a[nonzero_union_a] = (
            sampled_intersections_a[nonzero_union_a]
            / sampled_unions_a[nonzero_union_a]
        )
        ciou_b[nonzero_union_b] = (
            sampled_intersections_b[nonzero_union_b]
            / sampled_unions_b[nonzero_union_b]
        )
        ciou_a[~nonzero_union_a & (sampled_intersections_a != 0)] = 0.0
        ciou_b[~nonzero_union_b & (sampled_intersections_b != 0)] = 0.0
        ciou_difference_distribution[start:stop] = ciou_a - ciou_b

    return {
        "cIoU": percentile_confidence_interval(
            ciou_difference_distribution,
            confidence_level,
        ),
        "gIoU": percentile_confidence_interval(
            giou_difference_distribution,
            confidence_level,
        ),
    }


def summarize_metrics(accumulator):
    ious = accumulator["ious"]
    total_intersection = accumulator["total_intersection"]
    total_union = accumulator["total_union"]
    empty_target_total = accumulator["empty_target_total"]
    empty_target_correct = accumulator["empty_target_correct"]

    miou = np.mean(ious) if ious else 0.0
    ciou = (
        total_intersection / total_union
        if total_union > 0
        else (1.0 if total_intersection == 0 else 0.0)
    )
    n_acc = (
        empty_target_correct / empty_target_total
        if empty_target_total > 0
        else None
    )

    return {
        "predictions_evaluated": len(ious),
        "empty_targets_evaluated": empty_target_total,
        "mIoU": miou,
        "cIoU": ciou,
        "gIoU": miou,
        "N-acc": n_acc,
        "ious": ious,
    }


def summarize_iou_threshold(ious, threshold=MULTI_TRACK_IOU_THRESHOLD):
    """Count per-image IoUs strictly above, below, and equal to a threshold."""
    ious = np.asarray(ious, dtype=np.float64)
    above = int(np.count_nonzero(ious > threshold))
    below = int(np.count_nonzero(ious < threshold))
    equal = int(len(ious) - above - below)
    return {
        "threshold": threshold,
        "total": len(ious),
        "above": above,
        "below": below,
        "equal": equal,
    }


def summarize_high_iou_false_positive_bins(
    ious,
    false_positive_pixels,
    threshold=MULTI_TRACK_IOU_THRESHOLD,
):
    """Bin FP-pixel counts for images whose per-image IoU exceeds threshold."""
    ious = np.asarray(ious, dtype=np.float64)
    false_positive_pixels = np.asarray(false_positive_pixels, dtype=np.int64)
    if len(ious) != len(false_positive_pixels):
        raise ValueError("IoU and false-positive arrays must have equal lengths")

    eligible_fp = false_positive_pixels[ious > threshold]
    return {
        "iou_threshold": threshold,
        "eligible_samples": len(eligible_fp),
        "zero": int(np.count_nonzero(eligible_fp == 0)),
        "one_to_20": int(np.count_nonzero((eligible_fp >= 1) & (eligible_fp <= 20))),
        "twenty_one_to_40": int(
            np.count_nonzero((eligible_fp >= 21) & (eligible_fp <= 40))
        ),
        "exactly_41": int(np.count_nonzero(eligible_fp == 41)),
        "above_41": int(np.count_nonzero(eligible_fp > 41)),
    }


def print_iou_threshold_summary(title, summary):
    print(f"\n========== {title} ==========")
    threshold_percent = 100.0 * summary["threshold"]
    total = summary["total"]

    for relation, count_key in ((">", "above"), ("<", "below"), ("=", "equal")):
        count = summary[count_key]
        percentage = 100.0 * count / total if total else 0.0
        print(
            f"Per-image IoU {relation} {threshold_percent:.2f}%: "
            f"{count}/{total} ({percentage:.2f}%)"
        )


def print_high_iou_false_positive_summary(title, summary):
    print(f"\n========== {title} ==========")
    eligible_samples = summary["eligible_samples"]
    threshold_percent = 100.0 * summary["iou_threshold"]
    print(
        f"Eligible route-logic audit images with IoU > {threshold_percent:.2f}%: "
        f"{eligible_samples}"
    )

    bins = (
        ("0 FP pixels", "zero"),
        ("1-20 FP pixels", "one_to_20"),
        ("21-40 FP pixels", "twenty_one_to_40"),
        ("exactly 41 FP pixels", "exactly_41"),
        (">41 FP pixels", "above_41"),
    )
    for label, count_key in bins:
        count = summary[count_key]
        percentage = (
            100.0 * count / eligible_samples
            if eligible_samples
            else 0.0
        )
        print(f"{label}: {count}/{eligible_samples} ({percentage:.2f}%)")


def format_metric_with_interval(metric_name, metrics, confidence_level):
    value = metrics[metric_name]
    interval = metrics["confidence_intervals"][metric_name]
    if interval is None:
        return f"{metric_name}: {value:.4f} (paired-bootstrap CI unavailable)"

    confidence_percent = 100.0 * confidence_level
    return (
        f"{metric_name}: {value:.4f} "
        f"({confidence_percent:.1f}% paired-bootstrap CI "
        f"[{interval[0]:.4f}, {interval[1]:.4f}])"
    )


def print_metrics(title, metrics, confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL):
    print(f"\n========== {title} ==========")
    print(f"Predictions evaluated: {metrics['predictions_evaluated']}")
    print(format_metric_with_interval("mIoU", metrics, confidence_level))
    print(format_metric_with_interval("cIoU", metrics, confidence_level))
    print(format_metric_with_interval("gIoU", metrics, confidence_level))
    if metrics["N-acc"] is None:
        print("N-acc: N/A (no empty targets)")
    else:
        print(
            format_metric_with_interval("N-acc", metrics, confidence_level)
            + f"; empty targets: {metrics['empty_targets_evaluated']}"
        )


# =========================
# MAIN
# =========================

GROUP_TITLES = {
    "overall": "OVERALL",
    "single_track": "REMAINING TEST IMAGES",
    "multi_track": "ROUTE-LOGIC AUDIT",
}


def new_group_accumulators():
    return {
        "overall": new_metric_accumulator(),
        "single_track": new_metric_accumulator(),
        "multi_track": new_metric_accumulator(),
    }


def prediction_file_map(directory):
    """Map canonical image names to prediction-mask paths in one directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Prediction directory not found: {directory}")

    prediction_files = {}
    for pred_file in sorted(os.listdir(directory)):
        lower_name = pred_file.lower()
        if not lower_name.endswith(".jpg"):
            continue

        base_name = (
            lower_name[:-len("_mask.jpg")] + ".jpg"
            if lower_name.endswith("_mask.jpg")
            else lower_name
        )
        if base_name in prediction_files:
            raise ValueError(
                f"Multiple prediction masks in {directory} map to {base_name}"
            )
        prediction_files[base_name] = os.path.join(directory, pred_file)

    return prediction_files


def add_statistics_to_groups(accumulators, scene_group, sample_statistics):
    add_sample_statistics(accumulators["overall"], sample_statistics)
    add_sample_statistics(accumulators[scene_group], sample_statistics)


def scene_group_for_image(base_name, multi_track_indexes):
    image_index = extract_image_index(base_name)
    return "multi_track" if image_index in multi_track_indexes else "single_track"


def print_bootstrap_settings():
    print(
        "\nPaired image-level percentile bootstrap: "
        f"{BOOTSTRAP_RESAMPLES:,} resamples, "
        f"{100.0 * BOOTSTRAP_CONFIDENCE_LEVEL:.1f}% confidence level, "
        f"base seed {BOOTSTRAP_RANDOM_SEED}"
    )


def summarize_single_model_groups(accumulators):
    results = {}
    for group_number, (group, accumulator) in enumerate(accumulators.items()):
        metrics = summarize_metrics(accumulator)
        metrics["confidence_intervals"] = paired_bootstrap_confidence_intervals(
            accumulator,
            resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
            random_seed=BOOTSTRAP_RANDOM_SEED + group_number,
        )
        results[group] = metrics

    results["multi_track"]["iou_threshold_counts"] = summarize_iou_threshold(
        accumulators["multi_track"]["ious"],
        MULTI_TRACK_IOU_THRESHOLD,
    )
    results["multi_track"]["high_iou_false_positive_counts"] = (
        summarize_high_iou_false_positive_bins(
            accumulators["multi_track"]["ious"],
            accumulators["multi_track"]["false_positive_pixels"],
            MULTI_TRACK_IOU_THRESHOLD,
        )
    )
    return results


def evaluate_single_model(gt_data, multi_track_indexes):
    prediction_files = prediction_file_map(PRED_MASK_DIR)
    accumulators = new_group_accumulators()

    for base_name, pred_path in tqdm(
        sorted(prediction_files.items()),
        desc=METHOD_A_NAME,
    ):
        try:
            if base_name not in gt_data:
                print(f"[SKIP] No GT for {base_name}")
                continue

            ann = gt_data[base_name]
            pred_mask = load_pred_mask(pred_path)
            gt_mask = rails_annotation_to_mask(ann, pred_mask.shape)
            sample_statistics = compute_sample_statistics(pred_mask, gt_mask)
            scene_group = scene_group_for_image(
                base_name,
                multi_track_indexes,
            )
            add_statistics_to_groups(
                accumulators,
                scene_group,
                sample_statistics,
            )

        except Exception as e:
            print(f"[WARNING] {base_name}: {e}")

    results = summarize_single_model_groups(accumulators)
    print_bootstrap_settings()
    for group, title in GROUP_TITLES.items():
        print_metrics(
            f"{title} RESULTS",
            results[group],
            BOOTSTRAP_CONFIDENCE_LEVEL,
        )
    print_iou_threshold_summary(
        "ROUTE-LOGIC AUDIT PER-IMAGE IoU THRESHOLD",
        results["multi_track"]["iou_threshold_counts"],
    )
    print_high_iou_false_positive_summary(
        "HIGH-IoU ROUTE-LOGIC AUDIT FALSE-POSITIVE PIXELS",
        results["multi_track"]["high_iou_false_positive_counts"],
    )

    # Preserve the original top-level overall metric keys for existing callers,
    # while also exposing all three result groups.
    return {
        **results["overall"],
        "overall": results["overall"],
        "single_track": results["single_track"],
        "multi_track": results["multi_track"],
    }


def print_paired_comparison(title, comparison):
    print(f"\n========== {title} PAIRED COMPARISON ==========")
    print(f"Paired predictions evaluated: {comparison['paired_images']}")
    if comparison["paired_images"] == 0:
        print("cIoU: N/A (no paired predictions)")
        print("gIoU: N/A (no paired predictions)")
        return

    confidence_percent = 100.0 * BOOTSTRAP_CONFIDENCE_LEVEL

    for metric_name in ("cIoU", "gIoU"):
        value_a = comparison["method_a"][metric_name]
        value_b = comparison["method_b"][metric_name]
        difference = comparison["differences"][metric_name]
        interval = comparison["difference_confidence_intervals"][metric_name]
        if interval is None:
            interval_text = "paired-bootstrap CI unavailable"
        else:
            interval_text = (
                f"{confidence_percent:.1f}% paired-bootstrap CI "
                f"[{100.0 * interval[0]:+.2f}, {100.0 * interval[1]:+.2f}] pp"
            )

        print(
            f"{metric_name}: {METHOD_A_NAME}={value_a:.4f}, "
            f"{METHOD_B_NAME}={value_b:.4f}, "
            f"difference={100.0 * difference:+.2f} pp "
            f"({interval_text})"
        )


def evaluate_paired_comparison(gt_data, multi_track_indexes):
    prediction_files_a = prediction_file_map(PRED_MASK_DIR)
    prediction_files_b = prediction_file_map(COMPARISON_PRED_MASK_DIR)
    image_names_a = set(prediction_files_a)
    image_names_b = set(prediction_files_b)
    paired_image_names = sorted(image_names_a & image_names_b)

    if not paired_image_names:
        raise ValueError(
            "The two prediction directories have no matching image filenames"
        )

    print(f"{METHOD_A_NAME} prediction files: {len(prediction_files_a)}")
    print(f"{METHOD_B_NAME} prediction files: {len(prediction_files_b)}")
    print(f"Matched prediction filenames: {len(paired_image_names)}")
    print(f"Only in {METHOD_A_NAME}: {len(image_names_a - image_names_b)}")
    print(f"Only in {METHOD_B_NAME}: {len(image_names_b - image_names_a)}")

    accumulators_a = new_group_accumulators()
    accumulators_b = new_group_accumulators()

    for base_name in tqdm(paired_image_names, desc="Paired comparison"):
        try:
            if base_name not in gt_data:
                print(f"[SKIP] No GT for {base_name}")
                continue

            pred_mask_a = load_pred_mask(prediction_files_a[base_name])
            pred_mask_b = load_pred_mask(prediction_files_b[base_name])
            if pred_mask_a.shape != pred_mask_b.shape:
                raise ValueError(
                    "Prediction-mask shapes differ: "
                    f"{pred_mask_a.shape} versus {pred_mask_b.shape}"
                )

            gt_mask = rails_annotation_to_mask(
                gt_data[base_name],
                pred_mask_a.shape,
            )
            statistics_a = compute_sample_statistics(pred_mask_a, gt_mask)
            statistics_b = compute_sample_statistics(pred_mask_b, gt_mask)
            scene_group = scene_group_for_image(
                base_name,
                multi_track_indexes,
            )
            add_statistics_to_groups(
                accumulators_a,
                scene_group,
                statistics_a,
            )
            add_statistics_to_groups(
                accumulators_b,
                scene_group,
                statistics_b,
            )

        except Exception as e:
            # A failed image is omitted from both methods to preserve pairing.
            print(f"[WARNING] {base_name}: {e}")

    if not accumulators_a["overall"]["ious"]:
        raise ValueError("No valid paired predictions could be evaluated")

    results = {}
    for group_number, group in enumerate(GROUP_TITLES):
        metrics_a = summarize_metrics(accumulators_a[group])
        metrics_b = summarize_metrics(accumulators_b[group])
        difference_intervals = paired_bootstrap_difference_confidence_intervals(
            accumulators_a[group],
            accumulators_b[group],
            resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
            random_seed=BOOTSTRAP_RANDOM_SEED + group_number,
        )
        results[group] = {
            "paired_images": metrics_a["predictions_evaluated"],
            "method_a": metrics_a,
            "method_b": metrics_b,
            "differences": {
                "cIoU": metrics_a["cIoU"] - metrics_b["cIoU"],
                "gIoU": metrics_a["gIoU"] - metrics_b["gIoU"],
            },
            "difference_confidence_intervals": difference_intervals,
        }

    print_bootstrap_settings()
    print(
        f"Difference definition: {METHOD_A_NAME} minus {METHOD_B_NAME}; "
        f"positive values favour {METHOD_A_NAME}."
    )
    for group, title in GROUP_TITLES.items():
        print_paired_comparison(title, results[group])

    return {
        "mode": "paired_comparison",
        "method_a_name": METHOD_A_NAME,
        "method_b_name": METHOD_B_NAME,
        **results,
    }


def evaluate(args=None):
    global GT_JSON_PATH
    global EVALUATION_MODE
    global METHOD_A_NAME
    global PRED_MASK_DIR
    global METHOD_B_NAME
    global COMPARISON_PRED_MASK_DIR
    global AUDIT_CSV_PATH

    if args is None:
        args = parse_args()

    GT_JSON_PATH = args.gt_json
    EVALUATION_MODE = args.mode
    METHOD_A_NAME = args.method_name
    PRED_MASK_DIR = args.predictions
    METHOD_B_NAME = args.comparison_method_name
    COMPARISON_PRED_MASK_DIR = args.comparison_predictions
    AUDIT_CSV_PATH = args.audit_csv

    gt_data = load_gt(GT_JSON_PATH)
    multi_track_indexes = load_multi_track_indexes(AUDIT_CSV_PATH)
    mode = EVALUATION_MODE.strip().lower()

    if mode in {"single", "single_model"}:
        return evaluate_single_model(gt_data, multi_track_indexes)
    if mode in {"compare", "comparison", "paired_comparison"}:
        return evaluate_paired_comparison(gt_data, multi_track_indexes)

    raise ValueError(
        "EVALUATION_MODE must be 'single_model' or 'paired_comparison', "
        f"not {EVALUATION_MODE!r}"
    )


if __name__ == "__main__":
    evaluate()
