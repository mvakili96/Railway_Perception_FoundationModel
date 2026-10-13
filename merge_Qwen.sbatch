#!/bin/bash
# Full-state consolidation uses CPU RAM; model probes run sequentially on one L40.
#SBATCH -J Qwen-merge
#SBATCH -p gen-part
#SBATCH --qos=gpu1-32h
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --exclude=lse-hpcnode5,lse-hpcnode7,lse-hpcnode14
#SBATCH -t 04:00:00
#SBATCH -o qwen-merge-%j.out

set -euo pipefail
export QWEN_IMG="${QWEN_IMG:-/scratch/mohammadjavad/Qwen3VL-v1.sif}"
export QWEN_CODE="${QWEN_CODE:-/scratch/mohammadjavad/QWEN-LISA_Codebase}"
export QWEN_DATA="${QWEN_DATA:-/scratch/mohammadjavad/GSVA}"
export QWEN_CHECKPOINTS="${QWEN_CHECKPOINTS:-/scratch/mohammadjavad/LISA}"
export QWEN_MODEL="${QWEN_MODEL:-$QWEN_CHECKPOINTS/Qwen3-VL-8B-Instruct}"
# Edit these defaults for another experiment. Container paths below map through
# QWEN_CODE (/workspace/home), QWEN_DATA and QWEN_CHECKPOINTS respectively.
export QWEN_RUN_DIR="${QWEN_RUN_DIR:-/workspace/home/runs/qwen_159894}"
export QWEN_EXPORT_DIR="${QWEN_EXPORT_DIR:-/workspace/home/exports/qwen_159894}"
# Default: eight fixed held-out validation images. Optional single-image debug
# override: set QWEN_VERIFY_IMAGE or pass --verify_image to the submission.
export QWEN_VERIFY_MANIFEST="${QWEN_VERIFY_MANIFEST:-/workspace/home/configs/qwen_export_validation.json}"
export QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-512}"
# Empty uses checkpoints/latest, like merge_LISA.sbatch. Set an epoch tag here
# or pass --tag=epoch_XXX to sbatch to export a different retained checkpoint.
export QWEN_CHECKPOINT_TAG="${QWEN_CHECKPOINT_TAG:-}"
checkpoint_args=()
if [[ -n "$QWEN_CHECKPOINT_TAG" ]]; then
  checkpoint_args+=("--tag=$QWEN_CHECKPOINT_TAG")
fi
verification_args=("--verify_manifest=$QWEN_VERIFY_MANIFEST")
if [[ -n "${QWEN_VERIFY_IMAGE:-}" ]]; then
  verification_args=("--verify_image=$QWEN_VERIFY_IMAGE")
fi
for argument in "$@"; do
  case "$argument" in
    --verify_image|--verify_image=*|--verify_manifest|--verify_manifest=*) verification_args=(); break ;;
  esac
done
srun --kill-on-bad-exit=1 bash "$QWEN_CODE/qwen_hpc_run.sh" merge_qwen_sam.py \
  "--run_dir=$QWEN_RUN_DIR" "--output_dir=$QWEN_EXPORT_DIR" \
  ${verification_args[@]+"${verification_args[@]}"} "--max_new_tokens=$QWEN_MAX_NEW_TOKENS" \
  ${checkpoint_args[@]+"${checkpoint_args[@]}"} "$@"
