# Railway Perception Foundation Model

### Reasoning-guided ego-path segmentation for railway switches

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-2ea44f.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
[![Built on LISA](https://img.shields.io/badge/Built%20on-LISA-2563eb)](https://github.com/JIA-Lab-research/LISA)
![Status: Prototype](https://img.shields.io/badge/Status-Research%20prototype-f59e0b)

This is the official implementation of [Reasoning-guided Ego-path Segmentation for Autonomous Trains using Vision-language Models](https://doi.org/10.5194/isprs-archives-XLIX-B3-2026-89-2026), extended with selective CLIP/SAM tuning, weighted rail-reasoning tokens, counterfactual flipping, and an auxiliary ego-side loss. It predicts a binary ego-route mask and optional rationale from a rail image and text prompt.

## Results

Evaluated on 2,000 held-out test images: 1,822 switch-independent images and 178 switch-dependent images where route selection requires interpreting a visible downstream switch. Scores are percentages against the ground-truth ego-route.

These results update [our published paper](https://doi.org/10.5194/isprs-archives-XLIX-B3-2026-89-2026) using the latest checkpoints and evaluation. The exact values below do not appear in the paper.

Prompts used:

- **Reasoning-oriented:** `By examining rail continuity and switch geometry, segment the active ego-route the train is following in this image.`
- **Generic:** `Segment the track bed in this image.`

<div align="center">

| &nbsp;&nbsp;&nbsp;Model&nbsp;&nbsp;&nbsp; | Reasoning: switch-independent CIoU | Reasoning: switch-independent GIoU | Reasoning: switch-dependent CIoU | Reasoning: switch-dependent GIoU | Generic: switch-independent CIoU | Generic: switch-independent GIoU | Generic: switch-dependent CIoU | Generic: switch-dependent GIoU |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| &nbsp;&nbsp;&nbsp;[Original&nbsp;LISA&nbsp;(`xinlai/LISA-7B-v1`)](https://huggingface.co/xinlai/LISA-7B-v1)&nbsp;&nbsp;&nbsp; | 30.07 | 32.39 | 36.59 | 39.73 | 8.25 | 7.70 | 4.87 | 4.57 |
| &nbsp;&nbsp;&nbsp;[Rail-finetuned&nbsp;LISA&nbsp;—&nbsp;semantic&nbsp;only](https://huggingface.co/m-vakili75/railway-lisa-7b-semantic-clip)&nbsp;&nbsp;&nbsp; | 66.23 | 66.15 | 56.95 | 57.97 | 65.92 | 66.03 | 57.21 | 58.18 |
| &nbsp;&nbsp;&nbsp;[**Rail&#8209;finetuned&nbsp;LISA&nbsp;—&nbsp;semantic&nbsp;+&nbsp;reasoning**](https://huggingface.co/m-vakili75/railway-lisa-7b-semantic-reasoning-clip)&nbsp;&nbsp;&nbsp; | **89.00** | **88.34** | **90.49** | **90.33** | 65.58 | 65.94 | 57.87 | 58.89 |

</div>

Compared with semantic-only training, joint training gains 22.77/22.19 CIoU/GIoU points on switch-independent scenes and 33.53/32.35 on switch-dependent scenes. All paired-bootstrap 95% confidence intervals are above zero.

### Paired-bootstrap comparison

Improvements over semantic-only finetuning from 10,000 paired image-level percentile-bootstrap resamples (95% confidence level, seed 2026):

<div align="center">

| Scene type | Metric | Difference (percentage points) | Paired-bootstrap 95% confidence interval |
|:---:|:---:|:---:|:---:|
| Switch-independent | CIoU | +22.77 | [22.00, 23.53] |
| Switch-independent | GIoU | +22.19 | [21.50, 22.89] |
| Switch-dependent | CIoU | +33.53 | [31.10, 36.07] |
| Switch-dependent | GIoU | +32.35 | [30.10, 34.68] |

</div>

### Route-logic audit

The audit uses outputs generated with the reasoning-oriented prompt on 30 switch-dependent test images, balanced by switch type (15 turnout and 15 merge) and active route (15 left and 15 right).

A strict mask is correct only when the intended route has the highest IoU, intended-route IoU is above 0.90, the prediction reaches a route-exclusive region, and competing-route coverage in route-exclusive pixels is below 0.10.

**(a) Output-level correctness**

<div align="center">

| Assessment | Correct | Incorrect |
|:---:|:---:|:---:|
| Strict mask criterion | 24 (80.0%) | 6 (20.0%) |
| Switch type in rationale | 26 (86.7%) | 4 (13.3%) |
| Active-route direction in rationale | 21 (70.0%) | 9 (30.0%) |
| Switch type and direction jointly | 18 (60.0%) | 12 (40.0%) |

</div>

**(b) Paired mask–rationale outcomes**

<div align="center">

| Mask assessment | Type and direction correct | At least one incorrect | Total |
|:---:|:---:|:---:|:---:|
| Correct | 13 | 11 | 24 |
| Incorrect | 5 | 1 | 6 |
| **Total** | **18** | **12** | **30** |

</div>

### Qualitative examples

![Six sample railway scenes of the test set with predicted ego-route masks in red and generated route rationales.](docs/readme/qualitative-results.png)

<p align="center"><em>Six test-set examples generated using the reasoning-oriented prompt. Red overlays show the predicted ego-route masks, and the adjacent text shows the model-generated rationales.</em></p>

## Model and Training

![Railway-adapted LISA architecture showing the vision-language reasoning and SAM segmentation paths.](docs/readme/architecture.svg)

The included two-node job is the current reference configuration:

- **Trainable components:** LLM LoRA `q_proj`/`v_proj` adapters, token embeddings and language head, the `[SEG]` projection, multimodal projector, SAM prompt and mask decoders, the final 16 SAM blocks, the final 8 CLIP blocks used by LLaVA, and the ego-side head. Other CLIP/SAM blocks remain frozen.
- **Supervision and loss:** semantic masks teach rail appearance. Reasoning samples use fixed fields for switch type, both blade states, connected/disconnected paths, and the final left/right ego-path. Exact token alignment gives switch-type and right-blade-state targets 60× language-CE weight. The reference loss is `weighted CE + 2× mask BCE + Dice + 3× ego-side CE`. Optional boundary and pixel-map BCE weights are supported but disabled in the reference job.
- **Counterfactual augmentation:** with probability 0.5, a reasoning sample is flipped horizontally together with its mask, optional weight map, every left/right term in its rationale, and its ego-side label.
- **Monitoring:** Optional W&B record loss components, switch/right-blade token CE and accuracy, learning rates, and validation CIoU/GIoU. End-of-epoch probes generate rationales from a fixed image set using the live distributed model.

## Environment Setup

The supported environment is the public Linux/AMD64 container [`mvakili96/lisa:v2`](https://hub.docker.com/r/mvakili96/lisa), based on PyTorch 2.2.2, CUDA 12.1, and cuDNN 8. The current reference job uses bfloat16 on eight NVIDIA L40 GPUs across two nodes. An NVIDIA driver compatible with the container's CUDA runtime and Apptainer/Singularity with GPU support are required. Slurm is needed only for the supplied cluster workflows.

[`requirements.txt`](requirements.txt) is inherited from the original LISA repository and is not the installation path for this version.

### Clone and pull the container

```bash
git clone https://github.com/mvakili96/Railway_Perception_FoundationModel.git
cd Railway_Perception_FoundationModel
apptainer pull LISA.sif docker://mvakili96/lisa:v2
```

### Model prerequisites

<div align="center">

| Workflow | Required checkpoint |
|:---:|:---:|
| Original baseline and current finetuning (`--hf_merged_model`) | The authors' [`xinlai/LISA-7B-v1`](https://huggingface.co/xinlai/LISA-7B-v1) Hugging Face checkpoint. It is evaluated unchanged as [**Original LISA**](https://openaccess.thecvf.com/content/CVPR2024/html/Lai_LISA_Reasoning_Segmentation_via_Large_Language_Model_CVPR_2024_paper.html) in the first Results table and initializes both railway-finetuned models. |
| Initialization from backbones | A prepared LLaVA checkpoint following the [upstream LISA instructions](https://github.com/JIA-Lab-research/LISA#pre-trained-weights) and the [LLaVA model-preparation guidance](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md), plus the [SAM ViT-H checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) supplied through `--vision_pretrained`. |
| Inference | Both railway checkpoints are publicly available on the [`m-vakili75` Hugging Face account](https://huggingface.co/m-vakili75): [semantic only](https://huggingface.co/m-vakili75/railway-lisa-7b-semantic-clip) and [semantic + reasoning](https://huggingface.co/m-vakili75/railway-lisa-7b-semantic-reasoning-clip). Download either checkpoint as a complete folder with its `vision_tower/`. |

</div>

Download the exact LISA initialization/baseline snapshot used here:

```bash
hf download xinlai/LISA-7B-v1 \
  --revision 43c754eef75871fcc11c84d3930a402f7b0a754f \
  --local-dir /absolute/path/to/checkpoints/LISA-7B-v1
```

Set the launcher's `--version` argument to that local directory.

### Cluster-specific settings

Before using the Slurm scripts, replace their partition, QoS, account, log path, `IMG`, `PROC_DATA`, `PROC_CKPT`, `PROC_CODE`, and `PROC_OUTPUT` values. Also replace the hard-coded `ens3f0np0` NCCL/Gloo interface and review every bind mount, cache, dataset, checkpoint, and output path for the target cluster.

Slurm opens each `#SBATCH -o` path before the job starts. After changing the absolute log paths in [`merge_LISA.sbatch`](merge_LISA.sbatch) and [`demo_LISA.sbatch`](demo_LISA.sbatch) to your own home directory, create their parent directories before calling `sbatch`:

```bash
mkdir -p "$HOME/logs_LISA_merge"
mkdir -p "$HOME/logs_LISA_demo"
```

## Inference

[`demo_LISA.sbatch`](demo_LISA.sbatch) runs [`chat_batch.py`](chat_batch.py) on one GPU inside the project container. Before submission, replace its Slurm settings. `IMG`, `PROC_CKPT`, `PROC_CODE`, `PROC_DATA`, `PROC_OUTPUT`, `MODEL_NAME`, `TEST_IMAGE_SUBDIR`, `RUN_NAME`, and `LISA_PROMPT` can be overridden through the job environment.

Download the released joint checkpoint:

```bash
hf download m-vakili75/railway-lisa-7b-semantic-reasoning-clip \
  --local-dir /absolute/path/to/checkpoints/railway-lisa-7b-semantic-reasoning-clip
```

The merged model must include its exported `vision_tower/`. Point `--version` to the model directory and `--vision-tower` to that subdirectory. The current loader requires the local vision-tower path to contain `clip`.

Set `TEST_IMAGE_SUBDIR`, which the launcher forwards to `--image_path`, to either:

- one `.jpg`, `.jpeg`, or `.png` file for single-image inference, or
- a directory to process its top-level images sequentially with the same prompt.

`--mask_save_path` receives thresholded mask JPEGs, `--vis_save_path` receives red-overlay JPEGs, and the generated rationale is printed to the Slurm log. After editing the placeholders, run:

```bash
PROC_CKPT=/absolute/path/to/checkpoints \
MODEL_NAME=railway-lisa-7b-semantic-reasoning-clip \
sbatch demo_LISA.sbatch

# Or process one image under PROC_DATA.
PROC_CKPT=/absolute/path/to/checkpoints \
MODEL_NAME=railway-lisa-7b-semantic-reasoning-clip \
TEST_IMAGE_SUBDIR=test/images/rs06001.jpg \
sbatch demo_LISA.sbatch
```

The script's `bf16`, 1024-token context, and reasoning prompt are the current reference inference settings. Use the supplied `chat_batch.py` path because it applies the bundled tokenizer's generation-token IDs at runtime.

## Dataset Layout

The preparation commands below write directly to the locations used by the repository:

<div align="center">

| Split | Final location | Used by |
|:---:|:---:|:---:|
| Semantic training | `dataset/RailSem19-SemSeg-LISA/` with images under `training/images/` and labels under `training/v2.0/labels/` | [`init_railsem`](utils/sem_seg_dataset.py) with `--dataset_dir=dataset --sem_seg_data=railsem` |
| Reasoning training | `dataset/reason_seg/ReasonSegRail/` with `train/`, `explanatory/train.json`, and `weight_maps/` | [`ReasonSegDataset`](utils/reason_seg_dataset.py) with `--dataset_dir=dataset --reason_seg_rail_data='ReasonSegRail|train'` |
| Validation | `dataset/reason_seg/ReasonSegRail/val/` with one `.jpg` and same-stem `.json` per sample | [`ValDataset`](utils/dataset.py) with `--dataset_dir=dataset --val_dataset='ReasonSegRail|val'` |
| Test images | `dataset/test/images/` | [`demo_LISA.sbatch`](demo_LISA.sbatch) when `PROC_DATA="$PWD/dataset"` and `TEST_IMAGE_SUBDIR=test/images` |
| Test ground truth | `dataset/test/rs19_egopath_1024.json` | [`evaluate_ego_path.py`](scripts/evaluation/evaluate_ego_path.py) through `--gt-json` |

</div>

### Semantic-training set

Obtain the original RailSem19 intensity images and dense semantic label maps from the [official portal](https://www.wilddash.cc/download) under the [RailSem19 license](https://www.wilddash.cc/license/railsem19). Keep these downloads local under `dataset/external/railsem19/semantic/` and prepare the first 6,000 image/label pairs in filename order:

- Apply the same centred, bottom-aligned 1024×1024 crop to each image and its dense mask: `x0 = (width - 1024) // 2`, `y0 = height - 1024`, then crop `[y0:height, x0:x0+1024]`.
- Create an unsigned 8-bit label mask initialized to class `2` (**Background**). Set pixels whose original RailSem19 value is `15` (`trackbed`) to class `0` (**Track bed**) and pixels with value `12` (`rail-track`) to class `1` (**Rail**).
- Save every crop as `training/images/<stem>.jpg` and its class-ID mask as `training/v2.0/labels/<stem>.png`. A single-channel PNG is sufficient. Three equal channels are also accepted by the loader.
- Copy the provided [`config_v2.0.json`](scripts/data/templates/railsem19/config_v2.0.json) to the dataset root.

The final layout must be `dataset/RailSem19-SemSeg-LISA/training/images/<stem>.jpg` and `training/v2.0/labels/<stem>.png`, with paired stems. This is the structure loaded by [`init_railsem`](utils/sem_seg_dataset.py) when using `--dataset_dir=dataset --sem_seg_data=railsem`.

### Reasoning-training set

The 242 reasoning samples were manually selected from the first 4,000 RailSem19 images where the annotated ego-route crosses a switch whose configuration is visually clear. The set combines 126 manually cropped close-up views that retain the switch mechanism after CLIP preprocessing with 116 wider 1024×1024 views that preserve scene context. Each sample includes an ego-route polygon and five segmentation-prompt paraphrases. A companion manifest provides explanatory supervision describing switch topology, blade state, rail continuity, and the active route.

All samples are used for training. Place the requested package under `dataset/reason_seg/ReasonSegRail/` as shown below. The project-created reasoning annotations are available from the repository owner upon request. Any accompanying RailSem19 material remains governed by the [RailSem19 license](https://www.wilddash.cc/license/railsem19).

### Validation set

Place only the 500 validation images (`rs08000.jpg`–`rs08499.jpg`) in `dataset/external/railsem19/validation_images/`, alongside the TEP-Net annotation download already used for the test set. First create the crops and shifted rail coordinates:

```bash
mkdir -p dataset/metadata
python scripts/data/prepare_rs19_test_set.py \
  --input-json dataset/external/tepnet/egopath/rs19_egopath.json \
  --image-dir dataset/external/railsem19/validation_images \
  --output-dir dataset/reason_seg/ReasonSegRail/val \
  --output-json dataset/metadata/rs19_validation_egopath_1024.json
```

Then generate the per-image validation JSON beside each crop:

```bash
python scripts/data/generate_rs19_crop_jsons.py \
  --image-dir dataset/reason_seg/ReasonSegRail/val \
  --egopath-json dataset/metadata/rs19_validation_egopath_1024.json \
  --template-json scripts/data/templates/reason_seg_validation_template.json \
  --limit 500
```

The included template is one complete anonymized annotation sample, including its original polygon coordinates. It supplies the `text`, `is_sentence`, and shape structure expected by [`get_mask_from_json`](utils/data_processing.py). For every validation image, the generator replaces the prompts, image name, and points, forming the target polygon from the right rail followed by the reversed left rail. The current validation loader uses the first generated prompt. Select this split with `--val_dataset='ReasonSegRail|val'`.

### Test set

Obtain `rs19_val.zip` from the [official RailSem19 portal](https://www.wilddash.cc/download) and `rs19_egopath.json` through the [TEP-Net repository](https://github.com/irtrailenium/train-ego-path-detection#ego-path-annotations-and-trained-model-weights). Place the annotation file and held-out images 6,001–8,000 in the paths shown under [Expected directory structure](#expected-directory-structure), then run inside the supported container:

```bash
python scripts/data/prepare_rs19_test_set.py \
  --input-json dataset/external/tepnet/egopath/rs19_egopath.json \
  --image-dir dataset/external/railsem19/test_images \
  --output-dir dataset/test/images \
  --output-json dataset/test/rs19_egopath_1024.json
```

Use new or empty output paths. The script creates centred, bottom-aligned 1024×1024 crops, retains annotation points inside each crop, and shifts their coordinates to match SAM's input size.

### Run the test experiment

[`demo_LISA.sbatch`](demo_LISA.sbatch) runs inference on `dataset/test/images`. Set the checkpoint, prompt, and a unique output name.

Repeat for each checkpoint and the two prompts under [Results](#results).

Inside the container, evaluate one run or compare two prediction directories with the same tested calculations:

```bash
python scripts/evaluation/evaluate_ego_path.py \
  --mode paired_comparison \
  --gt-json dataset/test/rs19_egopath_1024.json \
  --predictions outputs/test/joint_reasoning/masks \
  --method-name "Joint training" \
  --comparison-predictions outputs/test/semantic_reasoning/masks \
  --comparison-method-name "Segmentation only" \
  --audit-csv scripts/evaluation/metadata/route_logic_audit_30.csv
```

[`route_logic_audit_30.csv`](scripts/evaluation/metadata/route_logic_audit_30.csv) contains the audit labels (`T/M`: turnout/merge; `R/L`: right/left). The current evaluator reads its `image_index` column to report the audit subset separately. All other valid predictions are reported as the remaining set. These groups do not reconstruct the full switch-dependent/switch-independent split in the Results tables.

The evaluator implements the segmentation metrics in the first Results table and the paired-bootstrap calculations in the second. Use `--mode single_model` for one prediction directory.

### Data composition and preprocessing

<div align="center">

| Data type (n) | RailSem19 source and selection | Input dimensions | Encoder-specific preprocessing |
|:---:|:---:|:---:|:---:|
| **Semantic training (6,000)** | First 6,000 images; all included. | Unscaled 1024×1024 crops; horizontally centered, bottom-aligned | **CLIP:** shortest side to 224; 224×224 center crop; normalize.<br>**SAM:** normalize; no resizing or padding. |
| **Reasoning: close-up (126)** | Qualifying switch scenes from images 1–4,000; annotated ego-path traverses a switch region whose configuration is visually discernible; per-image ROI retains the switch mechanism and relevant route geometry. | Width: 170–1266; height: 102–659 pixels | **CLIP:** shortest side to 224; 224×224 center crop; normalize.<br>**SAM:** if needed, resize the longest side down to 1024 while preserving aspect ratio; normalize; zero-pad to 1024×1024. Images are never upscaled. |
| **Reasoning: wider context (116)** | Qualifying switch scenes from images 1–4,000 under the same visible-switch criterion; surrounding context retained. | 1024×1024 | **CLIP:** shortest side to 224; 224×224 center crop; normalize.<br>**SAM:** normalize; no resizing or padding. |
| **Validation (500)** | RailSem19 images 8,001–8,500; held out from training. | Unscaled 1024×1024 crops; horizontally centered, bottom-aligned | **CLIP:** shortest side to 224; 224×224 center crop; normalize.<br>**SAM:** normalize; no resizing or padding. |
| **Test (2,000)** | RailSem19 images 6,001–8,000; held out from training and validation. | Unscaled 1024×1024 crops; horizontally centered, bottom-aligned | **CLIP:** shortest side to 224; 224×224 center crop; normalize.<br>**SAM:** normalize; no resizing or padding. |

</div>

### Reasoning-training distribution

Counts are before stochastic counterfactual flipping.

<div align="center">

| Switch topology | Active ego-route | Close-up | Wider context | Total |
|:---:|:---:|:---:|:---:|:---:|
| **Turnout** | Right | 42 | 33 | 75 |
| **Turnout** | Left | 33 | 37 | 70 |
| **Merge** | Right | 28 | 25 | 53 |
| **Merge** | Left | 23 | 21 | 44 |
| **Total** |  | **126** | **116** | **242** |

</div>

### Expected directory structure

```text
├── dataset
│   ├── external
│   │   ├── railsem19
│   │   │   ├── semantic
│   │   │   │   ├── images
│   │   │   │   └── labels
│   │   │   ├── test_images
│   │   │   └── validation_images
│   │   └── tepnet/egopath/rs19_egopath.json
│   ├── metadata
│   │   └── rs19_validation_egopath_1024.json
│   ├── test
│   │   ├── images
│   │   └── rs19_egopath_1024.json
│   ├── reason_seg/ReasonSegRail
│   │   ├── train
│   │   ├── val
│   │   ├── explanatory/train.json
│   │   └── weight_maps
│   ├── RailSem19-SemSeg-LISA
│   │   ├── config_v2.0.json
│   │   └── training
│   │       ├── images
│   │       └── v2.0/labels
```

## Distributed Training and Export

### Fine-tune

[`fine_tune_LISA_2nodes.sbatch`](fine_tune_LISA_2nodes.sbatch) is the current two-node launcher. Slurm starts four GPU tasks on each node, and every task runs one DeepSpeed rank inside the Apptainer container: eight ranks on eight NVIDIA L40 GPUs, using bfloat16 and ZeRO-2.

The arguments passed to [`train_ds.py`](train_ds.py) control the reusable model, data, and training behavior. Slurm resources, rank rendezvous, the `ens3f0np0` interface, container and bind sources, caches, logs, and W&B paths are cluster-specific. Replace them and ensure that the target cluster assigns one GPU to each task.

```bash
# Edit the launcher’s cluster settings and placeholder model/data paths first.
# Reference run: 2 nodes, 8 GPUs, bf16, effective batch 16, 20 × 50 updates.
sbatch fine_tune_LISA_2nodes.sbatch
```

Automatic resume is enabled: an existing `runs/<exp_name>/ckpt_model` is loaded. Use a new `--exp_name` when starting a clean run.

### Convert and export

Edit the placeholders, then run:

```bash
sbatch merge_LISA.sbatch
```

Keep the entire output directory together because the root weights do not duplicate the saved CLIP tower. When moving it, pass its local `vision_tower/` explicitly as documented under [Inference](#inference).

## Repository Validation
### Validation commands

Run these from the repository root inside the supported container:

```bash
python -m unittest discover -s tests -v
python -m compileall -q chat_batch.py train_ds.py \
  merge_lora_weights_and_save_hf_model.py model utils scripts tests
bash -n demo_LISA.sbatch fine_tune_LISA_2nodes.sbatch merge_LISA.sbatch
git diff --check
```

The 11 CPU unit tests check structured-rationale token alignment and left/right counterfactual transforms. They do not load a model, train, convert checkpoints, measure segmentation quality, or run inference.

### Troubleshooting

<div align="center">

| Symptom | Check and fix |
|:---:|:---:|
| Missing configuration, tokenizer, or weights | `--version` must point to the root of a complete merged export—not `ckpt_model/`, `fp32_model/`, or one weight file. Check the path inside the container; the root needs its configuration, tokenizer, model weight file(s), and sibling `vision_tower/`. |
| `Unknown vision tower` or CLIP load failure | Pass the exported tower through `--vision-tower`. Its directory needs CLIP configuration, weights, and processor configuration, and the current loader requires the full path string to contain `clip`. This explicit argument also overrides a stale absolute path saved during export. |
| CUDA out of memory | Inference already processes folder images sequentially; use BF16, free competing GPU jobs, or use a larger GPU. For training, lower `--batch_size` and raise `--grad_accumulation_steps` if the effective batch must stay fixed. The available 4/8-bit modes are not validated for the final checkpoint. |
| Distributed startup hangs or times out | Replace `ens3f0np0` consistently in `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and the master-address lookup with an IPv4 interface reachable from every node. Verify the printed address, port, world size, ranks, and one-GPU-per-task mapping. |
| Rationale appears but no mask is saved | A mask is decoded only when the model generates `[SEG]`. Inspect `text_output`, use the documented prompt and `llava_v1` template, and verify that the merged model, tokenizer, and vision tower belong to the same export. Do not add `[SEG]` to the user prompt manually. |

</div>

## Citation

If this repository is useful for your work, please cite the [published paper](https://isprs-archives.copernicus.org/articles/XLIX-B3-2026/89/2026/) and LISA.

```bibtex
@article{ghorbanalivakili2026reasoning,
  author  = {Ghorbanalivakili, Mohammadjavad and Varghese, Ashley and Sohn, Gunho},
  title   = {Reasoning-guided Ego-path Segmentation for Autonomous Trains using Vision-language Models},
  journal = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume  = {XLIX-B3-2026},
  pages   = {89--96},
  year    = {2026},
  doi     = {10.5194/isprs-archives-XLIX-B3-2026-89-2026},
  url     = {https://isprs-archives.copernicus.org/articles/XLIX-B3-2026/89/2026/}
}

@inproceedings{Lai_2024_CVPR,
  author    = {Lai, Xin and Tian, Zhuotao and Chen, Yukang and Li, Yanwei and Yuan, Yuhui and Liu, Shu and Jia, Jiaya},
  title     = {LISA: Reasoning Segmentation via Large Language Model},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2024},
  pages     = {9579--9589}
}
```

## Credits

This project adapts [LISA](https://github.com/JIA-Lab-research/LISA), built with [LLaVA](https://github.com/haotian-liu/LLaVA) and [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything). [RailSem19](https://openaccess.thecvf.com/content_CVPRW_2019/html/Autonomous_Driving/Zendel_RailSem19_A_Dataset_for_Semantic_Rail_Scene_Understanding_CVPRW_2019_paper.html) provides the source railway imagery and semantic labels. Switch rationales and reasoning masks were annotated for this project. Evaluation ego-path ground truth derives from the RailSem19 extension introduced with [TEP-Net](https://arxiv.org/abs/2403.13094).

## License

Unless otherwise noted, repository code is licensed under the [Apache License 2.0](LICENSE). Third-party code and model weights retain their upstream terms. LLaVA-derived checkpoints may also inherit the base language model's license.

RailSem19 is not covered by this repository's license. Raw source data is not bundled. The qualitative result figure contains six derived visualizations based on RailSem19 frames, which remain subject to the [RailSem19 license agreement](https://www.wilddash.cc/license/railsem19) and are not a substitute for the dataset. That agreement applies separate terms to imagery, dense metadata, and sparse metadata. The TEP-Net `rs19_egopath.json` annotation file is distributed upstream under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Downloaded annotations and the derived crop-coordinate JSON remain local and are not relicensed by this repository. The included audit CSV contains project-created labels only—not image pixels or TEP-Net rail coordinates.
