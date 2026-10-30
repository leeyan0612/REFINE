# REFINE

Compositional retinal image retrieval and evidence localization with **CaSMO** supervision and **Q-SAT** training.

REFINE adapts frozen visual features with a query-conditioned head. The head combines local appearance, global image context and a differentiable image-only latent geometry encoding. A sentence-conditioned decoder supplies both localization maps and image retrieval scores.

## Figure

[Figure Legend](figures/Figure_Legend.pdf) illustrates CaSMO supervision, the REFINE query head and Q-SAT optimization.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Python 3.10+ and PyTorch 2.5+ are required. Use a CUDA build of PyTorch for training.

## Pretrained heads

The `weights/` directory provides three RET-CLIP REFINE query heads. The manifest lists their dimensions and checksums. Use the author RET-CLIP visual encoder and EyeCLIP sentence encoder to construct the input features.

Load a released head:

```python
from refine import load_model
model = load_model("weights/retclip_qsat_seed1901.safetensors", device="cuda")
```

## Inference

```bash
python -m refine.infer \
  --weights weights/retclip_qsat_seed1901.safetensors \
  --features example_features.npz \
  --output outputs/query_maps.npz --device cuda
```

The feature NPZ contains:

| Field | Shape | Meaning |
|---|---|---|
| `local` | `[B, 784, local_dim]` | Frozen, spatially registered 28×28 visual features |
| `global_features` | `[B, global_dim]` | Frozen full-image context |
| `text_vectors` | `[Q, 512]` | Frozen EyeCLIP sentence embeddings |
| `support` | `[B, 256, 256]` | Image-only non-padding support |

Use the model dimensions recorded in `weights/manifest.json`. Feature extraction uses a centered square image, one full view and sixteen non-overlapping 4×4 tile views. Tile features are mapped by physical area overlap to the common grid. Keep the authors' encoder preprocessing and text encoder version. Arbitrarily resized feature arrays are not interchangeable with these inputs.

Inference produces sigmoid query heatmaps, image retrieval scores, and the latent geometry predictions. It does not consume reference masks, landmark coordinates or group membership. The two predicted reference points are latent variables; this release does not identify them as validated anatomical landmarks.

## CaSMO annotations

`annotations/` provides the released reference targets, query definitions and source roles. See [the annotation documentation](annotations/README.md) and the manifest for exact coverage, format, source attribution and licenses.

- **C** selects hard exudates, hemorrhages, microaneurysms or soft exudates.
- **S** applies foveal-distance and disc-relative direction conditions.
- **M** selects members of qualifying geometric lesion groups.
- Unknown reference coverage is retained and contributes no negative supervision.

The annotations are geometric references constructed from the recorded sources. Derived spatial and morphology targets are distinguished from the underlying official lesion annotations.

## Training

Q-SAT uses a separate META subset to adjust C/S/M axis weights and M category weights. The neural head uses AdamW; the meta-update uses a one-step SGD preview following Auto-Lambda.

```bash
python -m refine.train --bundle prepared_bundle.json \
  --output outputs/seed1901 --seed 1901 --steps 3000 --device cuda
```

Alternatively, read the released annotations directly and supply the matching author-encoder feature banks:

```bash
python -m refine.train --annotations annotations/manifest.json \
  --feature-root /path/to/frozen_features --text-bank /path/to/eyeclip_text.npz \
  --output outputs/seed1901 --seed 1901 --steps 3000 --device cuda
```

Per-image feature files are named `<public_id>.npz` and contain both independently encoded orientations: `local [2,784,768]` and `global_features [2,512]` for the released RET-CLIP interface. The text NPZ contains `train [64,3,512]`, ordered C4, S32, S-child8, M20. The annotation reader selects only the permitted C4/S28/M16 objectives and uses seeded image, flip and wording sampling.

The training entry accepts prepared tensor batches so that frozen encoder extraction can be reused. [Data format and batch construction](docs/data_format.md) describes the required tensors. Supply disjoint CORE and META content groups; the reference configuration uses CORE/META/CAL sizes 360/91/162. CAL does not update the model or controller. A static-weight control is available with `--static`.

## Code layout

- `refine/model.py`: query head, latent geometry, decoder and retrieval readout.
- `refine/losses.py`: known-area C/S/M and member-coverage objectives.
- `refine/satisfaction.py`: spatial condition and whole-group META feedback.
- `refine/training.py`: Q-SAT controller and update rule.
- `refine/infer.py`, `refine/train.py`: public entrypoints.
- `refine/data.py`: released annotations, content roles and member incidence.
- `refine/metrics.py`: image AP with grouped exact ties and unknown exclusion.

## Licenses and attribution

Project source code is released under the [MIT License](LICENSE). Annotation assets and trained heads have their own terms in their respective directories. Original dataset licenses remain applicable. Third-party backbone weights and source images are obtained separately.

Primary method and data references:

- IDRiD: https://doi.org/10.3390/data3030025
- DDR: https://github.com/nkicsl/DDR-dataset
- Auto-Lambda: https://github.com/lorenmt/auto-lambda
- RET-CLIP author code and checkpoints: https://github.com/sStonemason/RET-CLIP
- EyeCLIP author sentence encoder and checkpoints: https://github.com/Michi-3000/EyeCLIP

Use the author releases corresponding to the feature specifications above; RET-CLIP supplies the visual features and EyeCLIP supplies the 512-dimensional sentence vectors.

```bibtex
@software{refine2026,
  author = {{REFINE contributors}},
  title = {REFINE: Compositional Retinal Retrieval with CaSMO and Q-SAT},
  year = {2026},
  url = {https://github.com/leeyan0612/REFINE}
}
```
