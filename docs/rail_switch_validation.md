# Switch validation after each epoch

The selected set contains 40 held-out images: 20 turnouts, 20 merges, 24 left ego-routes, and 16 right ego-routes. The labels below are transcribed from the handwritten table and await verification by the dataset owner. Index 8302 and the five-digit filename convention were confirmed explicitly.

Each index refers directly to a filename: 8006 means `rs08006.jpg`, with no subtraction or conversion. `T` means turnout, `M` means merge, `L` means left, and `R` means right.

## Transcription for verification

The columns reproduce the three groups in the photograph.

| Index | Switch | Direction | Index | Switch | Direction | Index | Switch | Direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8006 | T | R | 8299 | T | L | 8470 | M | L |
| 8016 | M | L | 8302 | M | R | 8482 | T | L |
| 8051 | M | R | 8306 | M | L | 8483 | M | L |
| 8071 | T | L | 8319 | T | R | 8496 | M | R |
| 8083 | T | R | 8341 | T | R | | | |
| 8089 | M | L | 8343 | M | R | | | |
| 8096 | M | L | 8352 | M | L | | | |
| 8120 | M | L | 8357 | T | L | | | |
| 8124 | T | L | 8361 | T | L | | | |
| 8125 | M | R | 8372 | T | R | | | |
| 8189 | M | L | 8378 | M | R | | | |
| 8192 | T | L | 8389 | T | L | | | |
| 8200 | T | R | 8397 | M | L | | | |
| 8201 | M | L | 8402 | T | L | | | |
| 8205 | M | R | 8409 | M | R | | | |
| 8228 | T | L | 8437 | T | L | | | |
| 8272 | T | R | 8447 | T | L | | | |
| 8293 | T | R | 8465 | M | L | | | |

The machine-readable source is [rail_switch_validation.json](../scripts/data/templates/rail_switch_validation.json). Correct that source if verification reveals a transcription error.

## Persist the labels on HPC before training

Upload [rail_switch_validation.json](../scripts/data/templates/rail_switch_validation.json) to this HPC location, naming the uploaded file `val_switch_labels.json`:

```text
/scratch/mohammadjavad/GSVA/demo_dataset/reason_seg/ReasonSeg/explanatory/val_switch_labels.json
```

The training launcher reads that JSON through `--epoch_switch_validation_json`. There is no installation step in the training job. Verify the uploaded labels against the photograph before training. At startup, the evaluator checks that every selected image exists under `reason_seg/ReasonSeg/val/`.

The standalone installer remains an optional convenience for copying and printing the labels on the login node:

```bash
python3 /scratch/mohammadjavad/LISA_codebase/scripts/data/install_rail_switch_validation.py \
  --dataset-dir=/scratch/mohammadjavad/GSVA/demo_dataset
```

It preserves an identical existing manifest and stops if the persisted labels differ from the repository source. After reviewing an intentional correction, add `--overwrite` to that optional command.

## Evaluation behavior and metrics

The launcher enables `--epoch_switch_validation` and leaves `--val_dataset` unset so regular mask validation uses its existing default, `ReasonSeg|val`. Both validation passes read images from `reason_seg/ReasonSeg/val/`. Reasoning training remains in `ReasonSegRail|train`. The old training-image reasoning probe remains available through its original arguments but is no longer enabled in this launcher.

After each epoch, the live trained model runs the selected images without augmentation using greedy generation and this exact prompt:

> Based on the blade positions in this switch, which route corresponds to the route the train takes? Please respond with segmentation mask and explain why.

SAM decodes the generated `[SEG]` prompts using the configured bridge. This uses that epoch's current parameters even when the epoch did not produce a new best checkpoint. It does not reload an older checkpoint.

Switch type and route direction are parsed independently from explicit generated decisions, such as `This is a turnout switch` and `the ego-path follows the right-hand path`. Blade positions alone are never used to infer the route prediction. Missing or conflicting decisions count as incorrect. All annotated images stay in the denominator; inference errors are persisted and then fail the pass.

These W&B metrics are logged in the same epoch record as `val/giou` and `val/ciou`:

- `val/switch_subset/switch_type_accuracy`
- `val/switch_subset/route_direction_accuracy`
- `val/switch_subset/joint_accuracy`
- `val/switch_subset/switch_parse_rate`
- `val/switch_subset/route_direction_parse_rate`
- `val/switch_subset/one_mask_rate`
- Counts of samples, results, errors, correct decisions, parsed decisions, and single-mask outputs.

Accuracies are fractions from 0 to 1. Generated text, parsed predictions, labels, and errors are saved per epoch at `runs/<exp_name>/val_switch/epoch_0001/results.json`. The adjacent `masks/` directory contains lossless PNGs for every generated mask, with background 0 and foreground 100. The same generated-answer validation also runs when `--eval_only` and `--epoch_switch_validation` are both enabled.

## Local checks

```bash
python -m unittest tests.test_rail_switch_validation
python -m py_compile train_ds.py model/LISA.py utils/rail_switch_validation.py scripts/data/install_rail_switch_validation.py
bash -n fine_tune_LISA_2nodes.sbatch
```

The tests cover labels, installation, answer parsing, full-subset scoring, and mocked epoch-runner behavior including eight-rank sharding and W&B logging. Actual CUDA model generation requires verification on HPC.
