"""Offline Stage 2 processor/dataset gate. No training or model weights loaded."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import traceback
import unittest

from qwen_environment import load_manifest, package_versions, utc_now, version_errors, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-pixels", type=int, default=16384)
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.samples_per_source < 1 or not 0 < args.min_pixels <= args.max_pixels or args.max_seq_length < 1:
        parser.error("Sample count, pixel budgets, and sequence limit must be positive and ordered.")
    args.model_dir = args.model_dir.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    for protected in (args.model_dir, args.dataset_root):
        if args.report == protected or protected in args.report.parents:
            parser.error("Write the report outside the source checkpoint and dataset directories.")
    if args.report.suffix != ".json":
        parser.error("--report must be a JSON file.")
    if args.report.exists():
        parser.error("Report already exists; use a new name to preserve previous results.")
    # Processor persistence has its own managed directory, never a source snapshot.
    args.processor_output = args.report.with_name(args.report.stem + "_processor")
    if args.processor_output.exists():
        parser.error("Processor output already exists; choose a new report/job name to preserve it.")
    return args


def audit_sample(sample, collator):
    from utils.qwen_token_alignment import decode_exact

    batch = collator([sample])
    groups = []
    for row, positions in enumerate(batch["reasoning_group_positions"]):
        groups.append({name: decode_exact(collator.tokenizer,
            batch["input_ids"][row, indices].tolist()).strip() for name, indices in positions.items()})
    return {
        "image": sample.image_path, "raw_rgb_size": list(sample.original_size),
        "sam_resize": list(sample.resize), "sam_shape": list(sample.sam_image.shape),
        "mask_shape": list(sample.masks.shape), "mask_offsets": sample.mask_offsets,
        "weight_map_shape": list(sample.reason_seg_weight_maps.shape),
        "metadata": sample.metadata, "inference": sample.inference,
        "questions": sample.questions, "answers": sample.answers,
        "ego_side_label": sample.rail_ego_side_label,
        "image_grid_thw": batch["image_grid_thw"].tolist(),
        "visual_token_counts": batch["visual_token_counts"],
        "sequence_lengths": batch["attention_mask"].sum(-1).tolist(),
        "assistant_spans": batch["assistant_spans"], "reasoning_fields": groups,
        "seg_token_positions": batch["seg_token_positions"],
    }


def run_check(args, report):
    report.update(python=sys.version, versions=package_versions(), device="cpu",
                  model_weights_loaded=False, started_at=utc_now())
    errors = version_errors(report["versions"])
    if errors:
        raise ValueError(" ".join(errors))
    report["snapshot"] = load_manifest(args.model_dir)
    # Import after control validation so even missing-dependency failures are reported.
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from utils.dataset import ValDataset
    from utils.reason_seg_dataset import ReasonSegDataset
    from utils.sem_seg_dataset import SemSegDataset
    from utils.qwen_processing import QwenProcessingConfig, QwenSegCollator

    os.environ["QWEN_TEST_MODEL_DIR"] = str(args.model_dir)
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).parent / "tests"), pattern="test_qwen_*.py",
    )
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    report["tests"] = {"run": result.testsRun, "failures": len(result.failures),
                       "errors": len(result.errors), "skipped": list(result.skipped)}
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError("Stage 2 requires all tests to pass without any skips; see test output.")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    config = QwenProcessingConfig(args.min_pixels, args.max_pixels, args.max_seq_length)
    collator = QwenSegCollator(processor, config,
        use_rail_reasoning_weighted_ce=True, use_seg_token_weighted_ce=True)
    report.update(processing_config=asdict(config), seg_token_id=collator.seg_token_id,
                  dataset_audits=[], validation_dataset="ReasonSeg|val")
    root = args.dataset_root
    required = ("reason_seg/ReasonSegRail/train/*.jpg",
                "RailSem19-SemSeg-LISA/training/v2.0/labels/*.png",
                "reason_seg/ReasonSeg/val/*.jpg")
    for pattern in required:
        if not next(root.glob(pattern), None):
            raise ValueError(f"No samples found: {root / pattern}")
    if not (root / "reason_seg/ReasonSegRail/explanatory/train.json").is_file():
        raise ValueError("Rail explanatory/train.json is required to audit existing reasoning supervision.")
    common = dict(base_image_dir=str(root), tokenizer=processor.tokenizer,
                  vision_tower=None, image_size=1024, sample_mode="structured")
    sources = [
        ("rail_mask", ReasonSegDataset(**common, reason_seg_data="ReasonSegRail|train", explanatory=-1)),
        ("rail_mixed", ReasonSegDataset(**common, reason_seg_data="ReasonSegRail|train", explanatory=0.5,
                                      counterfactual_flip_prob=0.5, weight_map_weight=1.0)),
        ("rail_text", ReasonSegDataset(**common, reason_seg_data="ReasonSegRail|train", explanatory=1.0)),
        ("rail_flipped", ReasonSegDataset(**common, reason_seg_data="ReasonSegRail|train", explanatory=0.0,
                                        counterfactual_flip_prob=1.0, weight_map_weight=4.0)),
        ("semantic", SemSegDataset(**common, sem_seg_data="railsem")),
        ("validation", ValDataset(**common, val_dataset="ReasonSeg|val")),
    ]
    for source_index, (name, dataset) in enumerate(sources):
        for index in range(min(args.samples_per_source, len(dataset))):
            seed = args.seed + source_index * 1000 + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sample = dataset[index]
            record = audit_sample(sample, collator)
            record.update(source=name, seed=seed)
            report["dataset_audits"].append(record)
            print(json.dumps({"source": name, "image": sample.image_path,
                "tokens": record["sequence_lengths"], "masks": sample.masks.shape[0]}, ensure_ascii=False), flush=True)
    processor.save_pretrained(args.processor_output)
    config.save_pretrained(args.processor_output)
    # Reload persisted config/tokenizer so settings and token identity cannot drift.
    restored = QwenSegCollator(AutoProcessor.from_pretrained(args.processor_output, local_files_only=True),
                               QwenProcessingConfig.from_pretrained(args.processor_output))
    if restored.seg_token_id != collator.seg_token_id or restored.config != config:
        raise ValueError("Persisted processor configuration or [SEG] token identity changed.")
    report["processor_output"] = str(args.processor_output)


def main(argv=None):
    args = parse_args(argv)
    report = {"stage": 2, "status": "running"}
    try:
        run_check(args, report)
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    report["finished_at"] = utc_now()
    write_json(args.report, report)
    print(f"Stage 2 {report['status']}; report: {args.report}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
