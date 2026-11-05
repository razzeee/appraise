# icon-taste

Predict Flathub icon-quality review outcomes from icon images.

An experimental local tool that learns from public Flathub reviews and scores
icons for footprint, contrast, detail, baked-in shadows, and contemporary style.
It includes resumable dataset collection, label auditing, handcrafted and
CLIP-based classifiers, and duplicate-aware evaluation.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. From a fresh clone,
collect the dataset and train a model:

```sh
uv sync --extra embeddings
uv run --extra embeddings icon-review collect --limit 0
uv run --extra embeddings icon-review audit
uv run --extra embeddings icon-review train --features clip --output artifacts/full-clip
```

Once trained, score an icon:

```sh
uv run --extra embeddings icon-review predict /path/to/icon.png --model artifacts/full-clip/model.joblib
```

Predictions run locally and return per-criterion, uncalibrated scores, not
automatic approval decisions. The CLIP encoder downloads its pretrained weights
on first use. For a lighter model, use `--features handcrafted` when training.

## Training status

The full-catalog baseline and CLIP experiments completed training and local
prediction verification. Each used 2,273 training apps and 756 held-out apps.
Dataset and model files are **not included in this repository**; they live in
the gitignored `data/` and `artifacts/` directories on the training machine.
The commands above create your own local artifacts.

See [PIPELINE.md](PIPELINE.md) for collection, training, and development commands,
and [EXPERIMENTS.md](EXPERIMENTS.md) for results and limitations.
