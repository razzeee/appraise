# icon-taste

Predict Flathub icon-quality review outcomes from icon images.

An experimental local tool that learns from public Flathub reviews and scores
icons for footprint, contrast, detail, baked-in shadows, and contemporary style.
It also reviews app names, summaries, and branding colors using automatic rules
and calibrated local classifiers.
It includes resumable dataset collection, label auditing, handcrafted and
CLIP-based classifiers, and duplicate-aware evaluation.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. From a fresh clone,
download the trained models from the `v0.2.0` release (requires GitHub CLI access
to this repository):

```sh
uv sync --extra embeddings --locked
gh release download v0.2.0 --repo razzeee/icon-taste \
  --pattern icon-taste-models-v0.2.0.tar.gz --pattern SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS
tar -xzf icon-taste-models-v0.2.0.tar.gz
uv run --extra embeddings icon-review review-url https://flathub.org/apps/org.gnome.Calculator
```

The bundle contains calibrated CLIP, baseline, and listing models, evaluation
results, split manifests, and dependency provenance. The pretrained CLIP encoder
weights download separately on first use. Use this release's code and locked
dependencies with its model files.

To collect the dataset and train your own models instead:

```sh
uv sync --extra embeddings
uv run --extra embeddings icon-review collect --limit 0
uv run --extra embeddings icon-review audit
uv run --extra embeddings icon-review train --features clip --output artifacts/calibrated-clip
uv run --extra embeddings icon-review train-listing --output artifacts/listing
```

Once trained, score an icon:

```sh
uv run --extra embeddings icon-review predict /path/to/icon.png --model artifacts/calibrated-clip/model.joblib
```

Predictions run locally and return per-criterion scores and calibration status,
not automatic approval decisions. New training runs fit sigmoid calibration on
separate held-out groups; the original `model-v1` release is uncalibrated.
The CLIP encoder downloads its pretrained weights
on first use. For a lighter model, use `--features handcrafted` when training.

## Training status

The full-catalog baseline and CLIP experiments completed training and local
prediction verification. Each used 2,273 training apps and 756 held-out apps.
Dataset and model files are **not included in this repository**; they live in
the gitignored `data/` and `artifacts/` directories on the training machine.
The release bundle supplies trained artifacts; the training commands above
create your own.

Subsequent calibrated experiments used 1,693 training apps, 580 calibration apps,
and the same 756 test apps. Their local artifacts are in
`artifacts/calibrated-baseline/` and `artifacts/calibrated-clip/`. To use the
calibrated CLIP model on the training machine:

```sh
uv run --extra embeddings icon-review predict /path/to/icon.png --model artifacts/calibrated-clip/model.joblib
```

See [PIPELINE.md](PIPELINE.md) for collection, training, and development commands,
and [EXPERIMENTS.md](EXPERIMENTS.md) for results and limitations.

## Name, summary, and branding checks

Review a live Flathub page using both trained models:

```sh
uv run --extra embeddings icon-review review-url https://flathub.org/apps/org.gnome.Calculator
```

This fetches current metadata and the icon, then runs all 15 criteria. It expects
`artifacts/listing/model.joblib` and `artifacts/calibrated-clip/model.joblib` locally;
override them with `--model` and `--icon-model` if needed.

Add `--format moderation-json` to print individual moderation POST bodies as a
JSON array. `--pass-threshold` controls score conversion (default `0.5`).

With a collected dataset:

```sh
uv run --extra embeddings icon-review audit-listing
uv run --extra embeddings icon-review train-listing --output artifacts/listing
uv run --extra embeddings icon-review review --app-id org.gnome.Calculator \
  --model artifacts/listing/model.joblib
```

For your own metadata, edit `examples/listing.json` and run:

```sh
uv run --extra embeddings icon-review review examples/listing.json \
  --icon /path/to/icon.png --model artifacts/listing/model.joblib
```

See [LISTING.md](LISTING.md) for all criteria, combined icon review, input format,
rules-only mode, training details, and limitations.
