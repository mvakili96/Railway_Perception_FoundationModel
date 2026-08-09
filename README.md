# Railway Perception Foundation Model

### Reasoning-guided ego-path segmentation for railway switches

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-2ea44f.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
[![Built on LISA](https://img.shields.io/badge/Built%20on-LISA-2563eb)](https://github.com/JIA-Lab-research/LISA)
![Status: Research prototype](https://img.shields.io/badge/Status-Research%20prototype-f59e0b)

This is the official implementation of [Reasoning-guided Ego-path Segmentation for Autonomous Trains using Vision-language Models](https://doi.org/10.5194/isprs-archives-XLIX-B3-2026-89-2026), with newer model and training updates. It adapts LISA's LLaVA/CLIP and SAM paths to select the physically continuous ego-route at railway switches, producing a binary mask and an optional rationale from an image and text prompt.

> [!CAUTION]
> Research prototype only. Masks and rationales can be wrong or inconsistent and must not be used for safety-critical decisions.

## Results

The latest model was evaluated on 2,000 held-out images: 1,822 switch-independent and 178 switch-dependent. Scores are percentages against the ground-truth ego-route using the reasoning-oriented route prompt.

| Model | Switch-independent CIoU | Switch-independent GIoU | Switch-dependent CIoU | Switch-dependent GIoU |
|---|---:|---:|---:|---:|
| Original LISA | 30.07 | 32.39 | 36.59 | 39.73 |
| Rail-finetuned LISA — semantic only | 66.23 | 66.15 | 56.95 | 57.97 |
| **Rail-finetuned LISA — semantic + reasoning** | **89.00** | **88.34** | **90.49** | **90.33** |

CIoU pools pixels across the subset; GIoU averages image-level IoU. Compared with semantic-only training, joint training gains 22.77/22.19 CIoU/GIoU points on switch-independent scenes and 33.53/32.35 on switch-dependent scenes; all paired-bootstrap 95% confidence intervals are above zero.

### Paired-bootstrap comparison

Improvements over semantic-only finetuning from 10,000 paired image-level resamples:

| Scene type | Metric | Improvement | 95% confidence interval |
|---|---|---:|---:|
| Switch-independent | CIoU | +22.77 | [22.00, 23.53] |
| Switch-independent | GIoU | +22.19 | [21.50, 22.89] |
| Switch-dependent | CIoU | +33.53 | [31.10, 36.07] |
| Switch-dependent | GIoU | +32.35 | [30.10, 34.68] |

### Route-logic audit

Results on 30 switch-dependent images, balanced separately by switch type and route direction:

| Assessment | Correct |
|---|---:|
| Strict branch-aware mask | 24/30 (80.0%) |
| Switch type in rationale | 26/30 (86.7%) |
| Active-route direction in rationale | 21/30 (70.0%) |
| Both rationale fields | 18/30 (60.0%) |
| Correct mask **and** both rationale fields | 13/30 (43.3%) |

Evaluation notes:

- A generic track-bed prompt produces lower ego-route overlap, but this is not an all-track-bed accuracy measurement.
- Shared track regions can hide wrong-branch predictions in aggregate IoU; rationales are auxiliary outputs, not verified explanations.

### Qualitative examples

![Six railway scenes with predicted ego-route masks in red and generated route rationales.](docs/readme/qualitative-results.png)

Red overlays are predicted masks; adjacent text is model-generated.

## What Changed vs Original LISA
- Added rail reasoning dataset branch: `reason_seg_rail` (`ReasonSegRail|train`).
- Added rail semantic segmentation support: `railsem` in `utils/sem_seg_dataset.py`.
- Added training flow for merged HF checkpoints via `--hf_merged_model`.
- Added HPC pipeline scripts:
  - `fine_tune_LISA.sbatch`
  - `fine_tune_LISA_nodes.sbatch`
  - `merge_LISA.sbatch`

## Dataset Layout
```text
├── dataset
│   ├── reason_seg
│   │   └── ReasonSegRail
│   │       ├── train
│   │       ├── val
│   │       └── explanatory
│   ├── RailSem19-SemSeg-LISA
│   │   ├── config_v2.0.json
│   │   ├── training
```
Please note that we are still actively improving this method and expect to introduce newer versions of both the model and the dataset. Because the dataset may be expanded, refined, or restructured as the project evolves, we have not released the full data here yet and it will remain unavailable until further notice.

## HPC Slurm Workflow
This repo includes cluster scripts for Apptainer-based training and merging.

### Pre-trained weights

#### LLaVA
To train LISA-7B or 13B, you need to follow the [instruction](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) to merge the LLaVA delta weights. Typically, LISA authors use the final weights `LLaVA-Lightning-7B-v1-1` and `LLaVA-13B-v1-1` merged from `liuhaotian/LLaVA-Lightning-7B-delta-v1-1` and `liuhaotian/LLaVA-13b-delta-v1-1`, respectively. For Llama2, you can directly use the LLaVA full weights `liuhaotian/llava-llama-2-13b-chat-lightning-preview`.

#### SAM ViT-H weights
Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

### Container Setup
The Slurm scripts expect an Apptainer/Singularity image file (`.sif`) referenced by the `IMG` variable inside the scripts. The container used for this project is available from DockerHub:
[https://hub.docker.com/repositories/mvakili96](https://hub.docker.com/repositories/mvakili96)

Pull or build the corresponding container image, convert it to a `.sif` if needed for your cluster, and update the `IMG` path in the Slurm scripts before launching jobs.


### Fine-tune
Edit container/dataset/checkpoint paths in `fine_tune_LISA.sbatch`, then run:
```bash
sbatch fine_tune_LISA.sbatch
```

If you prefer multi-node, you can run:
```bash
sbatch fine_tune_LISA_2nodes.sbatch
```

### Merge
After the fine-tuning is done, in order to get the full model weight, merge the LoRA weights of pytorch_model.bin, and save the resulting model into your desired path in the Hugging Face format, edit paths in `merge_LISA.sbatch`, then run:
```bash
sbatch merge_LISA.sbatch
```

### Finetune Script Contrast: `fine_tune_LISA.sbatch` vs `fine_tune_LISA_2nodes.sbatch`
| Item | `fine_tune_LISA.sbatch` | `fine_tune_LISA_2nodes.sbatch` |
|---|---|---|
| Purpose | Single-node fine-tuning | Multi-node distributed fine-tuning |
| SLURM scale | `--gres=gpu:1` | Typically `--nodes=2` with multiple GPUs per node |
| Launch style | `deepspeed --num_gpus=$NUM_GPUS ...` | Typically `deepspeed --num_nodes ... --num_gpus ...` (or equivalent multi-node launcher) |
| Networking | Localhost-style env (`MASTER_ADDR=127.0.0.1`) | Cross-node rendezvous (`MASTER_ADDR` as first node hostname/IP) |
| Recommended use | Fast debug / small experiments | Paper-scale training (e.g., 2-node, multi-GPU setup) |


## Citation
If this repository is useful for your work, please cite both this paper and LISA.

```bibtex
@inproceedings{ghorbanalivakili2026railreason,
  title={Reasoning-guided Ego-path Segmentation for Autonomous Trains using Vision-language Models},
  author={Ghorbanalivakili, Mohammadjavad and Varghese, Ashley and Sohn, Gunho},
  booktitle={Acccepted and to be Published on ISPRS Archives of Photogrammetry and Remote Sensing},
  year={2026},
  note={Update final volume/pages/DOI}
}

@inproceedings{lai2024lisa,
  title={LISA: Reasoning Segmentation via Large Language Model},
  author={Lai, Xin and Tian, Zhuotao and Chen, Yukang and Li, Yanwei and Yuan, Yuhui and Liu, Shu and Jia, Jiaya},
  booktitle={CVPR},
  year={2024}
}
```

## Acknowledgement
This work is built upon:
- [LISA](https://github.com/JIA-Lab-research/LISA)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
