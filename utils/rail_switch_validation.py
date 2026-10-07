"""Labels and answer scoring for the held-out rail switch validation subset.

This module has no ML dependencies so labels can be installed and checked on
the HPC login node without loading a model or importing CUDA.
"""

import json
import re
from pathlib import Path


SWITCH_VALIDATION_PROMPT = (
    "Based on the blade positions in this switch, which route corresponds to "
    "the route the train takes? Please respond with segmentation mask and "
    "explain why."
)
LABEL_RELATIVE_PATH = Path("reason_seg/ReasonSeg/explanatory/val_switch_labels.json")
IMAGE_RELATIVE_PATH = Path("reason_seg/ReasonSeg/val")

_SWITCH_PATTERNS = (
    re.compile(r"\b(?:this|it)\s+is\s+(?:a|an)\s+(turnout|merge)\s+switch\b", re.I),
    re.compile(r"\b(?:this|the)\s+switch\s+is\s+(?:a\s+)?(turnout|merge)\b", re.I),
    re.compile(r"\bswitch\s+type\s*:\s*(turnout|merge)\b", re.I),
)
_ROUTE_PATTERN = re.compile(
    r"\b(?:ego[- ](?:path|route)|train)\s+"
    r"(?:follows|takes|uses|travels\s+along|will\s+(?:follow|take)|"
    r"is\s+(?:on|following)|corresponds\s+to)\s+"
    r"(?:(?:the|a)\s+)?(left|right)(?:[- ]hand)?\s+(?:path|route|branch)\b",
    re.I,
)


def load_switch_validation_labels(path):
    """Validate labels, including exact filenames, with no index conversion."""
    path = Path(path)
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Switch validation labels require schema_version=1")
    if data.get("prompt") != SWITCH_VALIDATION_PROMPT:
        raise ValueError("Switch validation labels must use the requested prompt")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Switch validation labels require a non-empty samples list")
    seen = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Every switch validation sample must be an object")
        index = sample.get("image_index")
        if type(index) is not int or not 8000 <= index <= 8499:
            raise ValueError("Switch validation image_index must be in 8000..8499")
        expected_image = "rs{:05d}.jpg".format(index)
        if sample.get("image") != expected_image:
            raise ValueError("Expected exact validation filename {}".format(expected_image))
        if expected_image in seen:
            raise ValueError("Duplicate switch validation image: {}".format(expected_image))
        seen.add(expected_image)
        if sample.get("switch") not in ("T", "M"):
            raise ValueError("Switch label must be T or M: {}".format(expected_image))
        if sample.get("direction") not in ("L", "R"):
            raise ValueError("Direction label must be L or R: {}".format(expected_image))
    return data


def resolve_switch_validation_manifest(dataset_dir, labels_path=None):
    """Require every selected held-out image before any training begins."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    path = (
        Path(labels_path).expanduser().resolve()
        if labels_path
        else dataset_dir / LABEL_RELATIVE_PATH
    )
    data = load_switch_validation_labels(path)
    image_dir = dataset_dir / IMAGE_RELATIVE_PATH
    manifest = [
        {**sample, "image_path": str(image_dir / sample["image"])}
        for sample in data["samples"]
    ]
    missing = [sample["image_path"] for sample in manifest if not Path(sample["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing {} switch validation images: {}".format(len(missing), ", ".join(missing))
        )
    manifest.sort(key=lambda sample: sample["image_index"])
    return manifest, str(path), str(image_dir)


def parse_switch_validation_answer(text):
    """Score independent explicit decisions, never infer route from blade state.

    Conflicting decisions are unparseable. A correct switch can still be scored
    when the route is absent, or vice versa; complete canonical prose is not
    required.
    """
    text = " ".join(text.split())
    switch_types = {
        match.group(1).lower()
        for pattern in _SWITCH_PATTERNS
        for match in pattern.finditer(text)
    }
    directions = {match.group(1).lower() for match in _ROUTE_PATTERN.finditer(text)}
    switch = {"turnout": "T", "merge": "M"}[next(iter(switch_types))] if len(switch_types) == 1 else None
    direction = {"left": "L", "right": "R"}[next(iter(directions))] if len(directions) == 1 else None
    return {"switch": switch, "direction": direction}


def summarize_switch_validation(manifest, results):
    """Use the full annotated subset as the denominator, including failures."""
    expected_by_image = {sample["image"]: sample for sample in manifest}
    if not expected_by_image or len(expected_by_image) != len(manifest):
        raise ValueError("Switch validation requires a non-empty, unique manifest")
    result_by_image = {}
    for result in results:
        image = result["image"]
        if image not in expected_by_image or image in result_by_image:
            raise ValueError("Unexpected or duplicate validation result: {}".format(image))
        result_by_image[image] = result

    total = len(manifest)
    summary = {
        "sample_count": total,
        "result_count": len(results),
        "error_count": 0,
        "switch_correct": 0,
        "route_direction_correct": 0,
        "joint_correct": 0,
        "switch_parse_count": 0,
        "route_direction_parse_count": 0,
        "one_mask_count": 0,
    }
    for image, expected in expected_by_image.items():
        result = result_by_image.get(image)
        if result is None or "error" in result:
            summary["error_count"] += 1
            continue
        parsed = parse_switch_validation_answer(result.get("prediction", ""))
        switch_correct = parsed["switch"] == expected["switch"]
        direction_correct = parsed["direction"] == expected["direction"]
        summary["switch_correct"] += int(switch_correct)
        summary["route_direction_correct"] += int(direction_correct)
        summary["joint_correct"] += int(switch_correct and direction_correct)
        summary["switch_parse_count"] += int(parsed["switch"] is not None)
        summary["route_direction_parse_count"] += int(parsed["direction"] is not None)
        summary["one_mask_count"] += int(result.get("mask_count") == 1)

    for key, count_key in (
        ("switch_type_accuracy", "switch_correct"),
        ("route_direction_accuracy", "route_direction_correct"),
        ("joint_accuracy", "joint_correct"),
        ("switch_parse_rate", "switch_parse_count"),
        ("route_direction_parse_rate", "route_direction_parse_count"),
        ("one_mask_rate", "one_mask_count"),
    ):
        summary[key] = summary[count_key] / total
    return summary


def switch_validation_wandb_metrics(summary):
    return {"val/switch_subset/" + name: value for name, value in summary.items()}
