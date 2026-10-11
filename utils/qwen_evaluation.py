"""Native inference samples and full-denominator segmentation/answer scoring."""

from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.segmentation_sample import StructuredSegSample
from utils.rail_switch_validation import parse_switch_validation_answer


def inference_sample(image_path, prompt):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = ResizeLongestSide(1024, allow_upscale=False).apply_image(rgb)
    tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float()
    mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
    height, width = resized.shape[:2]
    tensor = F.pad((tensor - mean) / std, (0, 1024 - width, 0, 1024 - height))
    # Inference has no supplied answer/mask; empty placeholders are never loss targets.
    return StructuredSegSample(str(image_path), rgb, tensor, ["<image>\n" + prompt], [""],
        torch.empty(0, *rgb.shape[:2]), torch.full(rgb.shape[:2], 255), (height, width), [],
        torch.empty(0), inference=True)


def foreground_iou(prediction, target):
    """Ignore 255; match existing foreground per-mask and global IoU conventions."""
    prediction, target = torch.as_tensor(prediction).bool(), torch.as_tensor(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target sizes differ.")
    valid = target.ne(255)
    actual = target.eq(1) & valid
    prediction = prediction & valid
    intersection = int((prediction & actual).sum())
    union = int((prediction | actual).sum())
    return intersection, union, intersection / (union + 1e-5) if union else 1.0


def save_binary_mask(mask, path):
    # NumPy cannot represent BF16. Threshold in torch, then convert the bool mask.
    if not bool(torch.isfinite(mask).all()):
        raise FloatingPointError("Cannot save a nonfinite predicted mask.")
    pixels = (mask.detach().cpu() > 0).numpy().astype(np.uint8) * 255
    Image.fromarray(pixels).save(path)


def summarize_test_results(records):
    """One record per request: no filtering failures out of the denominator."""
    total = len(records)
    if not total:
        raise ValueError("Evaluation requires at least one requested image.")
    identities = [record["image"] for record in records]
    if len(set(identities)) != total:
        raise ValueError("Duplicate requested image identities.")
    summary = {"requested_images": total, "result_count": total,
        "generation_success_rate": sum("error" not in r for r in records) / total,
        "one_mask_rate": sum(r.get("mask_count") == 1 for r in records) / total,
        "missing_seg_count": sum(r.get("mask_count") == 0 for r in records),
        "error_count": sum("error" in r for r in records)}
    labeled = [record for record in records if record.get("has_mask_label")]
    if labeled:
        summary["segmentation_labeled_count"] = len(labeled)
        summary["giou"] = sum(r.get("iou", 0.0) for r in labeled) / len(labeled)
        intersection = sum(r.get("intersection", 0) for r in labeled)
        union = sum(r.get("union", 0) for r in labeled)
        summary["ciou"] = intersection / (union + 1e-10) if all("union" in r for r in labeled) else None
        summary["annotation_error_count"] = sum("union" not in r for r in labeled)
        summary["segmentation_scored_count"] = sum("iou" in r for r in labeled)
    for field, key in (("switch", "switch_type_accuracy"), ("direction", "route_direction_accuracy")):
        labeled = [r for r in records if r.get("expected", {}).get(field) is not None]
        if labeled:
            correct = sum("error" not in r and parse_switch_validation_answer(r.get("prediction", ""))[field]
                          == r["expected"][field] for r in labeled)
            summary[key] = correct / len(labeled)
            summary[key + "_labeled_count"] = len(labeled)
    return summary
