#!/usr/bin/env python3
"""Dedicated BF16/ZeRO-2 Qwen3-VL + SAM training; no legacy LLaVA model."""

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from model.QwenSAM import QwenSAM
from model.qwen_sam_config import QwenSAMConfig
from utils.dataset import HybridDataset, ValDataset
from utils.qwen_checkpoint import (check_resume_settings, read_metadata, reconstruct_training_model,
    resolve_checkpoint_tag, save_run_metadata, write_json, check_resume_model_file, prune_completed_checkpoints)
from utils.qwen_checkpoint_selection import (BEST_CHECKPOINT_METRICS, DEFAULT_BEST_CHECKPOINT_METRIC,
    best_checkpoint_record, restore_best_checkpoint, select_best_checkpoint)
from utils.qwen_evaluation import foreground_iou, inference_sample
from utils.qwen_processing import QwenProcessingConfig, QwenSegCollator
from utils.qwen_training import TrainingPolicy, compute_losses, configure_training, deepspeed_config, move_batch
from utils.rail_switch_validation import (SWITCH_VALIDATION_PROMPT, resolve_switch_validation_manifest,
    summarize_switch_validation, switch_validation_wandb_metrics)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--model_dir")
    parser.add_argument("--sam_init_source", choices=("pretrained", "rail_checkpoint"))
    parser.add_argument("--sam_init_path")
    parser.add_argument("--resume", action="store_true", help="Restore this run, including optimizer and scheduler")
    parser.add_argument("--resume_tag")
    parser.add_argument("--seg_prompt_bridge_type", choices=("single", "four_query"), default="single")
    parser.add_argument("--train_sam_last_blocks", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="sem_seg||reason_seg_rail")
    parser.add_argument("--sample_rates", default="1,4")
    parser.add_argument("--sem_seg_data", default="railsem")
    parser.add_argument("--reason_seg_rail_data", default="ReasonSegRail|train")
    parser.add_argument("--val_dataset", default="ReasonSeg|val")
    parser.add_argument("--num_classes_per_sample", type=int, default=3)
    parser.add_argument("--explanatory", type=float, default=0.5)
    parser.add_argument("--rail_counterfactual_flip_prob", type=float, default=0.5)
    parser.add_argument("--reason_seg_weight_map_dir_name", default="weight_maps")
    parser.add_argument("--reason_seg_weight_map_weight", type=float, default=1.0)
    parser.add_argument("--use_rail_reasoning_weighted_ce", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_seg_token_weighted_ce", action="store_true")
    parser.add_argument("--zero_offload_optimizer", action="store_true")
    parser.add_argument("--epoch_switch_validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epoch_switch_validation_json")
    parser.add_argument("--epoch_switch_validation_max_new_tokens", type=int, default=256)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--stop_after_epochs", type=int, help="Short gate; retain the full saved LR schedule")
    parser.add_argument("--keep_last_checkpoints", type=int, default=0,
                        help="Opt-in retention; 0 keeps all, positive keeps last N plus best validation epoch")
    parser.add_argument("--best_checkpoint_metric", choices=BEST_CHECKPOINT_METRICS,
                        default=DEFAULT_BEST_CHECKPOINT_METRIC,
                        help="Validation metric to maximize for best-checkpoint retention; ties keep the earlier epoch")
    parser.add_argument("--learning_gate", action="store_true", help="Inspect full gradients and group updates on first update")
    parser.add_argument("--distributed_gate", action="store_true",
                        help="Short diagnostic only: rotate real semantic, rail-mask and rail-text samples across ranks")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="qwen-rail")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--local_rank", "--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    rate_help = {
        "lr": "Learning rate for SAM, the segmentation bridge/head and SEG rows",
        "lora_lr": "Learning rate for Qwen language LoRA only (separate from lr)",
        "vision_lr": "Learning rate for trainable Qwen vision-encoder blocks",
        "merger_lr": "Learning rate for the main and all DeepStack visual mergers (separate from vision_lr)",
    }
    for name, default in asdict(TrainingPolicy()).items():
        parser.add_argument("--" + name, type=type(default), default=default, help=rate_help.get(name))
    parser.add_argument("--min_pixels", type=int, default=16384)
    parser.add_argument("--max_pixels", type=int, default=1048576)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    args = parser.parse_args(argv)
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.grad_accumulation_steps) <= 0 or args.workers < 0:
        parser.error("Invalid training sizes.")
    if args.stop_after_epochs is not None and args.stop_after_epochs <= 0:
        parser.error("stop_after_epochs must be positive.")
    if args.keep_last_checkpoints < 0:
        parser.error("keep_last_checkpoints cannot be negative.")
    if args.best_checkpoint_metric == "val/switch_subset/joint_accuracy" and \
       (args.no_eval or not args.epoch_switch_validation):
        parser.error("Joint-accuracy checkpoint selection requires evaluation and epoch_switch_validation.")
    if args.distributed_gate and (args.batch_size != 1 or args.steps_per_epoch > 5 or not args.stop_after_epochs):
        parser.error("distributed_gate requires batch_size=1, steps_per_epoch<=5 and stop_after_epochs.")
    if args.resume and (args.sam_init_path or args.sam_init_source or args.model_dir):
        parser.error("Resume uses saved architecture/full checkpoint, not initialization paths.")
    if not args.resume and not all((args.model_dir, args.sam_init_path, args.sam_init_source)):
        parser.error("Fresh runs require model_dir, sam_init_source and sam_init_path.")
    if args.resume_tag and not args.resume:
        parser.error("resume_tag requires resume.")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def synchronized_call(function, device):
    """Propagate non-collective phase failures before the next rank collective."""
    value, error = None, None
    try:
        value = function()
    except Exception as exc:
        error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("Distributed phase failed: " + "; ".join(e for e in errors if e))
    return value


@torch.no_grad()
def validate(model, dataset, collator, device, rank, world):
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    model.eval()
    for index in range(rank, len(dataset), world):
        sample = replace(dataset[index], inference=False)
        batch = move_batch(collator([sample]), device)
        output = model(**batch)
        for prediction, target in zip(output["pred_masks"][0], sample.masks):
            intersection, union, iou = foreground_iou(prediction.cpu() > 0, target)
            totals += totals.new_tensor([intersection, union, iou, 1])
    return totals


@torch.no_grad()
def switch_validation(model, manifest, collator, device, rank, world, max_new_tokens):
    records = []
    for item in manifest[rank::world]:
        record = {"image": item["image"]}
        try:
            batch = move_batch(collator([inference_sample(item["image_path"], SWITCH_VALIDATION_PROMPT)]), device)
            output = model.generate_masks(batch, max_new_tokens=max_new_tokens, do_sample=False)
            record.update(prediction=output["answers"][0], mask_count=len(output["pred_masks"][0]))
        except Exception as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return records


def gradient_gate(engine, manifest):
    from deepspeed.utils import safe_get_full_grad
    result = {group["name"]: {"squared_norm": 0.0, "missing": []} for group in manifest}
    by_name = {name: group["name"] for group in manifest for name in group["parameter_names"]}
    bad = []
    for name, parameter in engine.module.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = safe_get_full_grad(parameter)  # every rank, in identical order
        group = result[by_name[name]]
        if gradient is None:
            group["missing"].append(name)
        elif not bool(torch.isfinite(gradient).all()):
            bad.append(name)
        else:
            group["squared_norm"] += float(gradient.float().square().sum())
    if bad or any(group["missing"] for group in result.values()):
        raise RuntimeError(f"Learning gate failed: nonfinite={bad}, missing={result}")
    return result


def master_weight_probe(model, manifest):
    from deepspeed.utils import safe_get_full_fp32_param
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            master = safe_get_full_fp32_param(parameter)  # collective, identical order on every rank
            if master is None or not bool(torch.isfinite(master).all()):
                raise RuntimeError(f"Invalid optimizer master weight: {name}")
            result[name] = master.reshape(-1)[:64].detach().cpu().clone()
    return result


def diagnostic_pool(dataset):
    """Select existing samples; never rewrite their text, masks or supervision."""
    pool = {}
    for _ in range(500):
        sample = dataset[0]
        key = "semantic" if not sample.is_rail_reasoning else "rail_mask" if len(sample.masks) else "rail_text"
        pool.setdefault(key, sample)
        if len(pool) == 3:
            return [pool[key] for key in ("semantic", "rail_mask", "rail_text")]
    raise ValueError("Mixed diagnostic requires real semantic, rail-mask and rail-text samples; none will be synthesized.")


def main(argv=None):
    args = parse_args(argv)
    import deepspeed
    local = int(os.environ.get("SLURM_LOCALID", args.local_rank))
    for key, slurm_key in (("RANK", "SLURM_PROCID"), ("WORLD_SIZE", "SLURM_NTASKS")):
        if key not in os.environ and slurm_key in os.environ:
            os.environ[key] = os.environ[slurm_key]
    os.environ["LOCAL_RANK"] = str(local)
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    deepspeed.init_distributed(dist_backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    seed_everything(args.seed)  # identical fresh bridge initialization on all ranks
    run_dir = Path(args.run_dir).resolve()
    policy = TrainingPolicy(**{key: getattr(args, key) for key in asdict(TrainingPolicy())})
    processing = QwenProcessingConfig(args.min_pixels, args.max_pixels, args.max_seq_length)
    ignored = {"run_dir", "resume", "resume_tag", "model_dir", "sam_init_path", "sam_init_source",
               "local_rank", "stop_after_epochs", "learning_gate", "use_wandb", "wandb_mode", "wandb_project"}
    settings = {key: value for key, value in vars(args).items() if key not in ignored}
    settings["world_size"] = world

    def initialize_model():
        if args.resume:
            check_resume_settings(read_metadata(run_dir), settings)
            if resolve_checkpoint_tag(run_dir, args.resume_tag) != resolve_checkpoint_tag(run_dir):
                raise ValueError("Resume the latest tag to avoid overwriting later checkpoints; historical tags remain exportable.")
            return reconstruct_training_model(run_dir)[:2]
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("Fresh run directory must be empty; use --resume explicitly.")
        model = QwenSAM.from_native_pretrained(args.model_dir, sam_init_source=args.sam_init_source,
            sam_init_path=args.sam_init_path, processing_config=processing,
            config=QwenSAMConfig(seg_prompt_bridge_type=args.seg_prompt_bridge_type,
                                 train_sam_last_blocks=args.train_sam_last_blocks))
        groups, _ = configure_training(model, policy)
        return model, groups

    model, groups = synchronized_call(initialize_model, device)
    manifest = [{"name": group["group_name"], "lr": group["lr"],
        "parameter_count": sum(p.numel() for p in group["params"]),
        "parameter_names": [name for name, p in model.named_parameters()
                            if any(p is candidate for candidate in group["params"])]} for group in groups]
    model.gradient_checkpointing_enable()
    if not args.resume:
        synchronized_call(lambda: save_run_metadata(run_dir, model, policy, settings, manifest)
                          if rank == 0 else None, device)
    if rank == 0:
        print(json.dumps([{k: v for k, v in group.items() if k != "parameter_names"} for group in manifest]), flush=True)
        print(f"Best-checkpoint metric: {args.best_checkpoint_metric} (maximize; ties keep the earlier epoch)", flush=True)
    config = deepspeed_config(args.batch_size, args.grad_accumulation_steps, world, policy,
        args.epochs * args.steps_per_epoch, args.zero_offload_optimizer, optimizer_groups=groups)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=groups, config=config)
    start_epoch = 0
    best_score, best_tag = -1.0, None
    if args.resume:
        tag = resolve_checkpoint_tag(run_dir, args.resume_tag)
        synchronized_call(lambda: check_resume_model_file(engine.module, run_dir, tag), device)
        loaded, client = engine.load_checkpoint(str(run_dir / "checkpoints"), tag=tag,
            load_module_strict=True, load_optimizer_states=True, load_lr_scheduler_states=True)
        if not loaded:
            raise ValueError("Checkpoint restore failed or metadata differs.")
        check_resume_settings({**read_metadata(run_dir), "settings": client.get("settings", {})}, settings)
        start_epoch = client["next_epoch"]
        best_score, best_tag = restore_best_checkpoint(client, args.best_checkpoint_metric)
        if engine.global_steps != start_epoch * args.steps_per_epoch:
            raise ValueError("Checkpoint is not a completed optimizer/epoch boundary.")
        engine.module.validate_auxiliary_parameters()

    def prepare_data():
        count = args.steps_per_epoch * args.grad_accumulation_steps * args.batch_size * world
        dataset = HybridDataset(args.dataset_dir, model.processor.tokenizer, None,
            samples_per_epoch=count, precision="bf16", image_size=1024, sample_mode="structured",
            num_classes_per_sample=args.num_classes_per_sample, dataset=args.dataset,
            sample_rate=[float(x) for x in args.sample_rates.split(",")], sem_seg_data=args.sem_seg_data,
            reason_seg_rail_data=args.reason_seg_rail_data, explanatory=args.explanatory,
            reason_seg_weight_map_dir_name=args.reason_seg_weight_map_dir_name,
            reason_seg_weight_map_weight=args.reason_seg_weight_map_weight,
            rail_counterfactual_flip_prob=args.rail_counterfactual_flip_prob)
        validation = None if args.no_eval else ValDataset(args.dataset_dir, model.processor.tokenizer,
            None, args.val_dataset, sample_mode="structured")
        if validation is not None and not len(validation):
            raise ValueError(f"Empty validation set: {args.val_dataset}")
        subset = resolve_switch_validation_manifest(args.dataset_dir, args.epoch_switch_validation_json)[0] \
            if args.epoch_switch_validation and not args.no_eval else None
        return dataset, validation, subset

    dataset, validation, subset = synchronized_call(prepare_data, device)
    collator = QwenSegCollator(model.processor, processing, args.use_rail_reasoning_weighted_ce,
                              args.use_seg_token_weighted_ce)
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=args.seed)
    pool = synchronized_call(lambda: diagnostic_pool(dataset), device) if args.distributed_gate else None
    wandb_run = None
    if args.use_wandb:
        def initialize_wandb():
            if rank == 0:
                import wandb
                return wandb.init(project=args.wandb_project, name=run_dir.name,
                    id=run_dir.name, resume="must" if args.resume and args.wandb_mode == "online" else "allow",
                    mode=args.wandb_mode, dir=str(run_dir), config={**settings, "policy": asdict(policy)})
        wandb_run = synchronized_call(initialize_wandb, device)
    end_epoch = min(args.epochs, start_epoch + args.stop_after_epochs) if args.stop_after_epochs else args.epochs
    for epoch in range(start_epoch, end_epoch):
        seed_everything(args.seed + epoch * world + rank)
        sampler.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch * world + rank)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
            collate_fn=collator, pin_memory=True, worker_init_fn=seed_worker, generator=generator)
        iterator = iter(loader)
        engine.train()
        torch.cuda.reset_peak_memory_stats(device)
        master_before = master_weight_probe(engine.module, manifest) if args.learning_gate and epoch == start_epoch else None
        for step in range(args.steps_per_epoch):
            report = {}
            for micro in range(args.grad_accumulation_steps):
                batch = synchronized_call(lambda: move_batch(collator([pool[(rank + step + micro) % 3]])
                    if pool else next(iterator), device), device)
                def forward_loss():
                    losses = compute_losses(engine(**batch), batch, policy)
                    if not bool(torch.isfinite(losses["loss"])):
                        raise FloatingPointError(f"Nonfinite loss: images={batch['image_paths']}")
                    return losses
                losses = synchronized_call(forward_loss, device)
                engine.backward(losses["loss"])
                if args.learning_gate and step == 0 and epoch == start_epoch and micro == args.grad_accumulation_steps - 1:
                    gradient_report = gradient_gate(engine, manifest)
                    if rank == 0:
                        write_json(run_dir / f"learning_gradients_epoch_{epoch}.json", gradient_report)
                engine.step()  # DeepSpeed advances optimizer only at accumulation boundaries
                for key, value in losses.items():
                    report[key] = report.get(key, 0.0) + float(value.detach())
                del losses, batch
            vector = torch.tensor(list(report.values()), device=device, dtype=torch.float64)
            dist.all_reduce(vector)
            report = dict(zip(report, vector.cpu().tolist()))
            for key in list(report):
                if not key.endswith(("_ce_sum", "_correct", "_token_count")):
                    report[key] /= world * args.grad_accumulation_steps
            for field in ("switch", "right_state"):
                count = report[field + "_token_count"]
                report[field + "_accuracy"] = report[field + "_correct"] / max(count, 1)
                report[field + "_ce"] = report[field + "_ce_sum"] / max(count, 1)
            memory = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)
            dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            report["peak_memory_gib"] = float(memory)
            if rank == 0:
                print(json.dumps({"epoch": epoch, "step": engine.global_steps, **report}), flush=True)
                if wandb_run:
                    wandb_run.log({"train/" + k: v for k, v in report.items()}, step=engine.global_steps)
        if engine.global_steps != (epoch + 1) * args.steps_per_epoch:
            raise RuntimeError("DeepSpeed optimizer-update count differs from intended schedule.")
        if master_before is not None:
            master_after = master_weight_probe(engine.module, manifest)
            updates = {group["name"]: sum(not torch.equal(master_before[name], master_after[name])
                for name in group["parameter_names"]) for group in manifest}
            if rank == 0:
                write_json(run_dir / f"learning_updates_epoch_{epoch}.json", updates)
            if any(count == 0 for count in updates.values()):
                raise RuntimeError(f"No sampled FP32 master-weight update in groups: {updates}. "
                                   "Use mixed gate data and enough warmup updates.")
        metrics = {}
        if validation is not None:
            totals = synchronized_call(lambda: validate(engine.module, validation, collator, device, rank, world), device)
            dist.all_reduce(totals)
            metrics.update({"val/giou": float(totals[2] / totals[3].clamp_min(1)),
                "val/ciou": float(totals[0] / (totals[1] + 1e-10)), "val/mask_count": int(totals[3])})
        if subset:
            local_records = synchronized_call(lambda: switch_validation(engine.module, subset, collator,
                device, rank, world, args.epoch_switch_validation_max_new_tokens), device)
            collected = [None] * world
            dist.all_gather_object(collected, local_records)
            records = sorted([item for part in collected for item in part], key=lambda x: x["image"])
            summary = summarize_switch_validation(subset, records)
            metrics.update(switch_validation_wandb_metrics(summary))
            synchronized_call(lambda: write_json(run_dir / f"switch_validation_epoch_{epoch}.json",
                {"summary": summary, "results": records}) if rank == 0 else None, device)
        if rank == 0:
            print(json.dumps({"epoch": epoch, "teacher_forced_segmentation": True, **metrics}), flush=True)
            if wandb_run:
                wandb_run.log({"epoch": epoch, **metrics}, step=engine.global_steps)
        tag = f"epoch_{epoch + 1:03d}"
        best_score, best_tag = select_best_checkpoint(metrics, args.best_checkpoint_metric,
            best_score, best_tag, tag, allow_missing=args.no_eval)
        # Every rank participates. Frozen native weights are retained for a
        # standalone restore/export, not omitted to reduce checkpoint size.
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        saved = engine.save_checkpoint(str(run_dir / "checkpoints"), tag=tag,
            client_state={"next_epoch": epoch + 1, "settings": settings,
                          "best_score": best_score, "best_tag": best_tag,
                          "best_checkpoint_metric": args.best_checkpoint_metric}, exclude_frozen_parameters=False)
        if not saved:
            raise RuntimeError("DeepSpeed checkpoint save failed.")
        dist.barrier()
        def finish_checkpoint():
            if rank == 0:
                write_json(run_dir / "best_checkpoint.json",
                           best_checkpoint_record(args.best_checkpoint_metric, best_score, best_tag))
                removed = prune_completed_checkpoints(run_dir, args.keep_last_checkpoints,
                                                       [best_tag] if best_tag else [])
                if removed:
                    print("Removed older checkpoints after successful save (not recoverable): " + ", ".join(removed), flush=True)
        synchronized_call(finish_checkpoint, device)
    if wandb_run:
        wandb_run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
