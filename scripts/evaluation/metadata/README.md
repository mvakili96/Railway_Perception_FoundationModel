# Evaluation metadata

- `route_logic_audit_30.csv` contains human labels only for the balanced 30-image route-logic audit. `T` and `M` mean turnout and merge; `R` and `L` mean right and left.

The mask evaluator currently uses only `image_index` to select this subset. Scoring the type/direction labels and strict branch-aware correctness is planned separately.

This file contains project-created evaluation metadata only; it does not include RailSem19 images or TEP-Net rail coordinates.
