"""Metric-based Qwen checkpoint selection; independent of ML/W&B packages."""

import math


DEFAULT_BEST_CHECKPOINT_METRIC = "val/giou"
BEST_CHECKPOINT_METRICS = (
    DEFAULT_BEST_CHECKPOINT_METRIC,
    "val/ciou",
    "val/switch_subset/joint_accuracy",
)


def select_best_checkpoint(metrics, metric, best_score, best_tag, tag, *, allow_missing=False):
    """Maximize exactly one metric; ties retain the earlier checkpoint.

    Only explicit no-evaluation runs may omit the selected metric. Never fall
    back to another metric or treat a missing value as a valid zero score.
    """
    if metric not in BEST_CHECKPOINT_METRICS:
        raise ValueError(f"Unsupported best-checkpoint metric: {metric}")
    if metric not in metrics:
        if allow_missing:
            return best_score, best_tag
        raise ValueError(f"Selected best-checkpoint metric was not computed: {metric}")
    try:
        score = float(metrics[metric])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid best-checkpoint metric value for {metric}") from exc
    if not math.isfinite(score):
        raise ValueError(f"Nonfinite best-checkpoint metric: {metric}={score}")
    return (score, tag) if score > best_score else (best_score, best_tag)


def restore_best_checkpoint(client_state, metric):
    """Keep historical GIoU best scores from being reused as joint accuracy."""
    if metric not in BEST_CHECKPOINT_METRICS:
        raise ValueError(f"Unsupported best-checkpoint metric: {metric}")
    saved_metric = client_state.get("best_checkpoint_metric", DEFAULT_BEST_CHECKPOINT_METRIC)
    if saved_metric != metric:
        raise ValueError(f"Cannot change best-checkpoint metric on resume: {saved_metric} -> {metric}")
    score = float(client_state.get("best_score", -1.0))
    if not math.isfinite(score):
        raise ValueError("Cannot resume a nonfinite best-checkpoint score.")
    return score, client_state.get("best_tag")


def best_checkpoint_record(metric, score, tag):
    """Write an explicit metric/score and preserve the legacy GIoU alias.

    No-evaluation runs have no best checkpoint or measured best score. Other
    criteria use their own alias, never mislabeling joint accuracy as GIoU.
    """
    if metric not in BEST_CHECKPOINT_METRICS:
        raise ValueError(f"Unsupported best-checkpoint metric: {metric}")
    value = float(score) if tag is not None else None
    if value is not None and not math.isfinite(value):
        raise ValueError("Cannot record a nonfinite best-checkpoint score.")
    return {"tag": tag, "metric": metric, "score": value, metric.rsplit("/", 1)[-1]: value}
