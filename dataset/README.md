# Local dataset workspace

Only this file is versioned. Railway images, downloaded annotations, derived crops, and other generated data remain local and are ignored by Git.

```text
dataset/
├── external/
│   ├── railsem19/test_images/              # held-out images 6001–8000
│   └── tepnet/egopath/rs19_egopath.json   # official TEP-Net annotations
└── test/
    ├── images/                             # generated 1024×1024 crops
    └── rs19_egopath_1024.json              # shifted rail coordinates
```

Obtain RailSem19 from its [official download portal](https://www.wilddash.cc/download) and the ego-path annotations through the [TEP-Net data instructions](https://github.com/irtrailenium/train-ego-path-detection#ego-path-annotations-and-trained-model-weights). Each source retains its own license and access conditions.
