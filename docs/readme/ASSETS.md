# README Visual Assets

These assets present the repository's current method and latest internal evaluation as post-publication extensions of the original conference paper.

| Asset | Intended README use | Caption / provenance |
|---|---|---|
| `architecture.svg` | Model overview | Repository-created, code-aligned diagram based on `model/LISA.py`, `train_ds.py`, and the current two-node reference configuration. |
| `qualitative-results.png` | Qualitative results | Directly rendered evaluation plate. Red overlays are model-predicted valid ego-route masks; adjacent text is model-generated rationale. |
| `qualitative/example-000.jpg`–`example-005.jpg` | Responsive result gallery | Pixel-preserving JPEG extraction of the six prediction overlays used in the qualitative evaluation plate. |

Suggested accessible captions:

- `architecture.svg`: “The rail image enters CLIP/LLaVA and SAM paths; the projected segmentation token prompts SAM to decode the ego-route mask.”
- `qualitative-results.png`: “Six railway scenes with predicted ego-route masks in red and generated route rationales.”

The qualitative images are experimental outputs supplied by the repository owner. Do not describe the generated rationale as verified, causal, or safety-certified.
