# README Visual Assets

These assets present the repository's current method and latest internal evaluation as post-publication extensions of the original conference paper.

| Asset | Intended README use | Caption / provenance |
|---|---|---|
| `architecture.svg` | Model overview | Repository-created, code-aligned diagram based on `model/LISA.py`, `train_ds.py`, and the current two-node reference configuration. |
| `qualitative-results.png` | Qualitative results | Rebuilt evaluation plate derived from six RailSem19 frames. The prediction overlays are unchanged; rationale text follows the canonical template in `utils/rail_reasoning.py`. |
| `qualitative-results.svg` | Editable figure source | Deterministic layout source for `qualitative-results.png`, using six RailSem19-derived prediction overlays without pixel edits. |
| `qualitative-rationales.json` | Figure data | Machine-readable route attributes and full responses used in the rebuilt qualitative figure. |
| `qualitative/example-000.jpg`–`example-005.jpg` | Responsive result gallery | Pixel-preserving JPEG extraction of six RailSem19-derived prediction overlays used in the qualitative evaluation plate. |

Suggested accessible captions:

- `architecture.svg`: “The rail image enters CLIP/LLaVA and SAM paths; the projected segmentation token prompts SAM to decode the ego-route mask.”
- `qualitative-results.png`: “Six railway scenes with predicted ego-route masks in red and generated route rationales.”

The qualitative images are experimental outputs supplied by the repository owner and remain subject to the RailSem19 terms. Do not describe the generated rationale as verified, causal, or safety-certified.
