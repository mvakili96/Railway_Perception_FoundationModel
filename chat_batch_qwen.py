#!/usr/bin/env python3
"""Run a self-contained Qwen/SAM export; save every answer and lossless PNG masks."""

import argparse
import json
from pathlib import Path
import torch

from model.QwenSAM import QwenSAM
from utils.data_processing import get_mask_from_json
from utils.qwen_checkpoint import write_json
from utils.qwen_evaluation import (foreground_iou, inference_sample, summarize_test_results,
                                   save_binary_mask, save_lisa_compatible_outputs)
from utils.qwen_processing import QwenSegCollator
from utils.qwen_training import move_batch
from utils.rail_switch_validation import SWITCH_VALIDATION_PROMPT


def load_requests(image_path=None, manifest_path=None):
    """Manifests can request absent files: keep those records, never glob them out."""
    if bool(image_path) == bool(manifest_path):
        raise ValueError("Provide exactly one of image_path or manifest.")
    if manifest_path:
        path = Path(manifest_path).resolve()
        value = json.loads(path.read_text())
        items = value["samples"] if isinstance(value, dict) else value
        if not isinstance(items, list):
            raise ValueError("Manifest must be a list or contain samples.")
        requests = []
        for item in items:
            if not isinstance(item, dict) or "image_path" not in item:
                raise ValueError("Every request must contain image_path (relative to manifest or absolute).")
            image = Path(item["image_path"])
            if not image.is_absolute():
                image = path.parent / image
            request = {**item, "image_path": str(image.resolve()), "image": item.get("image", image.name)}
            if item.get("mask_json"):
                mask = Path(item["mask_json"])
                request["mask_json"] = str(mask if mask.is_absolute() else path.parent / mask)
            requests.append(request)
    else:
        path = Path(image_path).resolve()
        paths = sorted(p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")) \
            if path.is_dir() else [path]
        requests = [{"image": p.name, "image_path": str(p)} for p in paths]
    if not requests or len({r["image"] for r in requests}) != len(requests):
        raise ValueError("Requests must be nonempty and have unique image identities.")
    return requests


def validate_output_directories(output_dir, model_path, requests, mask_save_path=None, vis_save_path=None):
    """Check all outputs before creating any directories or loading the model."""
    directories = [Path(path).resolve() for path in (output_dir, mask_save_path, vis_save_path) if path]
    if len(set(directories)) != len(directories):
        raise ValueError("Results, masks, and visualization directories must be distinct.")
    protected = [Path(model_path).resolve(), *(Path(r["image_path"]).resolve().parent for r in requests)]
    for directory in directories:
        if any(directory == p or directory.is_relative_to(p) or p.is_relative_to(directory) for p in protected):
            raise ValueError("Output directories must be separate from model and input directories.")
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError(f"Output directory must be empty: {directory}")
    if mask_save_path or vis_save_path:
        stems = [Path(r["image_path"]).stem.casefold() for r in requests]
        if len(set(stems)) != len(stems):
            raise ValueError("Image stems must be unique for image-named legacy JPEG outputs.")
    return directories


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_save_path", help="Optional LISA-compatible image-named 0/100 mask JPEGs")
    parser.add_argument("--vis_save_path", help="Optional LISA-compatible red-overlay JPEGs")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image_path")
    source.add_argument("--manifest")
    parser.add_argument("--prompt", default=SWITCH_VALIDATION_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--evaluate_masks", action="store_true", help="Require adjacent ReasonSeg annotation JSONs")
    args = parser.parse_args(argv)
    verification_path = Path(args.model_path) / "export_verification.json"
    if not verification_path.is_file() or not json.loads(verification_path.read_text()).get("passed"):
        parser.error("Run merge_qwen_sam.py and pass export verification before test inference.")
    requests = load_requests(args.image_path, args.manifest)
    output_dir = Path(args.output_dir).resolve()
    try:
        directories = validate_output_directories(output_dir, args.model_path, requests,
                                                  args.mask_save_path, args.vis_save_path)
    except ValueError as exc:
        parser.error(str(exc))
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    model = QwenSAM.from_pretrained(args.model_path).cuda().eval()
    collator = QwenSegCollator(model.processor, model.processing_config)
    records = []
    with (output_dir / "results.jsonl").open("w") as handle:
        for index, request in enumerate(requests):
            record = {"image": request["image"], "image_path": request["image_path"],
                "prediction": "", "mask_count": 0, "mask_files": [],
                "expected": request.get("expected", {key: request[key] for key in ("switch", "direction") if key in request})}
            annotation = request.get("mask_json")
            if args.evaluate_masks and not annotation:
                annotation = str(Path(request["image_path"]).with_suffix(".json"))
            record["has_mask_label"] = bool(annotation)
            batch, prediction = None, None
            try:
                sample = inference_sample(request["image_path"], request.get("prompt", args.prompt))
                target = None
                if annotation:
                    target = torch.from_numpy(get_mask_from_json(annotation, sample.rgb_image)[0])
                    # Error/missing/multiple-SEG outputs predict no foreground;
                    # retain target union in cIoU and count them as failed gIoU.
                    record["intersection"] = 0
                    record["union"] = int(target.eq(1).sum())
                batch = move_batch(collator([sample]), "cuda")
                with torch.no_grad():
                    prediction = model.generate_masks(batch, max_new_tokens=args.max_new_tokens, do_sample=False)
                masks = prediction["pred_masks"][0].cpu()
                record.update(prediction=prediction["answers"][0], mask_count=len(masks))
                print("text_output: ", record["prediction"], flush=True)
                for mask_index, mask in enumerate(masks):
                    name = f"{index:06d}_mask_{mask_index:02d}.png"
                    save_binary_mask(mask, output_dir / name)
                    record["mask_files"].append(name)
                if len(masks) and (args.mask_save_path or args.vis_save_path):
                    # Original LISA demo saves the first mask for an image.
                    # Keep all events in PNG/JSON outputs and flag multiple SEG.
                    filename = Path(request["image_path"]).stem + ".jpg"
                    mask_path = Path(args.mask_save_path) / filename if args.mask_save_path else None
                    vis_path = Path(args.vis_save_path) / filename if args.vis_save_path else None
                    save_lisa_compatible_outputs(masks[0], sample.rgb_image, mask_path, vis_path)
                    record.update(legacy_mask_index=0, legacy_mask_file=str(mask_path) if mask_path else None,
                                  legacy_vis_file=str(vis_path) if vis_path else None)
                record["mask_status"] = "ok" if len(masks) == 1 else "missing_seg" if not len(masks) else "multiple_seg"
                if target is not None and len(masks) == 1:
                    intersection, union, iou = foreground_iou(masks[0] > 0, target)
                    record.update(intersection=intersection, union=union, iou=iou)
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                batch, prediction = None, None
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            finally:
                batch, prediction = None, None
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps({"processed": index + 1, "total": len(requests), "image": record["image"],
                              "mask_count": record["mask_count"], "error": record.get("error")}), flush=True)
    summary = summarize_test_results(records)
    write_json(output_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
