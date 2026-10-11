#!/bin/bash
# Dedicated Qwen run. Default to original SAM; do not reuse LISA run directories.
#SBATCH -J Qwen-SAM
#SBATCH -p gen-part
#SBATCH --qos=gpu8-4h
#SBATCH --account=research
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --exclude=lse-hpcnode5,lse-hpcnode14
#SBATCH -t 04:00:00
#SBATCH -o qwen-train-%j.out

set -euo pipefail
export QWEN_IMG="${QWEN_IMG:-/scratch/mohammadjavad/Qwen3VL-v1.sif}"
export QWEN_CODE="${QWEN_CODE:-/scratch/mohammadjavad/LISA_codebase}"
export QWEN_DATA="${QWEN_DATA:-/scratch/mohammadjavad/GSVA}"
export QWEN_CHECKPOINTS="${QWEN_CHECKPOINTS:-/scratch/mohammadjavad/LISA}"
export QWEN_MODEL="${QWEN_MODEL:-$QWEN_CHECKPOINTS/Qwen3-VL-8B-Instruct}"
export QWEN_SAM_INIT_SOURCE="${QWEN_SAM_INIT_SOURCE:-pretrained}"
export QWEN_SAM_INIT_PATH="${QWEN_SAM_INIT_PATH:-/workspace/checkpoints/sam_vit_h_4b8939.pth}"
export NCCL_SOCKET_IFNAME=ens3f0np0
export GLOO_SOCKET_IFNAME=ens3f0np0
test -f "$QWEN_IMG"
MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --nodelist="$MASTER_HOST" \
  bash -c 'ip -o -4 addr show dev ens3f0np0 | awk "{print \$4}" | cut -d/ -f1')
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
test -n "$MASTER_ADDR"
initialization=()
if [ "${QWEN_RESUME:-0}" = 1 ]; then
  : "${QWEN_RUN_DIR:?Set existing container run directory for resume}"
  initialization+=(--resume)
else
  : "${QWEN_SAM_INIT_SOURCE:?Set pretrained or rail_checkpoint}"
  : "${QWEN_SAM_INIT_PATH:?Set SAM path inside /workspace/checkpoints}"
  initialization+=(--model_dir=/workspace/qwen_model \
    "--sam_init_source=$QWEN_SAM_INIT_SOURCE" "--sam_init_path=$QWEN_SAM_INIT_PATH")
fi
run_dir="${QWEN_RUN_DIR:-/workspace/home/runs/qwen_${SLURM_JOB_ID}}"
srun --label --kill-on-bad-exit=1 bash "$QWEN_CODE/qwen_hpc_run.sh" train_qwen_sam.py \
  "--run_dir=$run_dir" --dataset_dir=/workspace/datasets/demo_dataset \
  --batch_size=1 --grad_accumulation_steps=2 --epochs=20 --steps_per_epoch=50 \
  --seg_prompt_bridge_type=single --use_wandb --wandb_mode=offline --keep_last_checkpoints=2 \
  --epoch_switch_validation_json=/workspace/datasets/demo_dataset/reason_seg/ReasonSeg/explanatory/rail_switch_validation.json \
  "${initialization[@]}" "$@"
