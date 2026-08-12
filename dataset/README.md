# Local dataset workspace

Only this file is versioned. Railway images, downloaded annotations, derived crops, and other generated data remain local and are ignored by Git.

```text
dataset/
├── external/
│   ├── railsem19/semantic/                  # original images + dense labels
│   ├── railsem19/test_images/              # held-out images 6001–8000
│   ├── railsem19/validation_images/        # validation images rs08000–rs08499
│   └── tepnet/egopath/rs19_egopath.json   # official TEP-Net annotations
├── metadata/
│   └── rs19_validation_egopath_1024.json  # shifted validation rails
├── reason_seg/ReasonSegRail/val/           # validation crops + per-image JSON
├── RailSem19-SemSeg-LISA/
│   ├── config_v2.0.json                    # copied three-class config
│   └── training/
│       ├── images/                         # generated 1024x1024 JPG crops
│       └── v2.0/labels/                    # generated class-ID PNG masks
└── test/
    ├── images/                             # generated 1024×1024 crops
    └── rs19_egopath_1024.json              # shifted rail coordinates
```

Obtain RailSem19 from its [official download portal](https://www.wilddash.cc/download) and the ego-path annotations through the [TEP-Net data instructions](https://github.com/irtrailenium/train-ego-path-detection#ego-path-annotations-and-trained-model-weights). Each source retains its own license and access conditions.

See the main README for the validation and held-out test preparation commands.

The final validation pairs must be under `reason_seg/ReasonSegRail/val/`. Final test images go under `test/images/`, with their combined shifted-coordinate JSON at `test/rs19_egopath_1024.json`. These are the default locations consumed by the validation loader, demo launcher, and evaluator; `external/` is only a local staging area.

The semantic loader expects `RailSem19-SemSeg-LISA/config_v2.0.json`, images under `training/images/`, and same-stem masks under `training/v2.0/labels/`. The main README shows how to generate this three-class dataset without committing the RailSem19 source files.
