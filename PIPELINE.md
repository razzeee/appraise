# Flathub icon review

Predict individual Flathub icon-quality review outcomes from icon images. This is
an experimental offline pipeline: collect public labels, audit the dataset, train
classifiers, and evaluate on held-out apps.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 (uv can install it).

```sh
uv sync
uv run icon-review collect --limit 300
uv run icon-review audit
uv run icon-review train
uv run icon-review predict /path/to/icon.png
```

Collection uses a seeded random sample of the catalog, sequential requests, a
minimum 250 ms request interval, bounded retries, and per-app checkpoints. Repeat
the same command to resume. Errors are recorded in `collection-errors.json` and
produce a nonzero exit code. Entries without an icon are recorded separately in
`collection-skipped.json`. `--limit 0` selects the entire catalog; increasing
the limit extends the same sample. The cached catalog fixes the sampling frame.
Use a new `--data` directory to take a fresh snapshot rather than mixing dates.
Do not run concurrent collectors into the same directory.

Collect specific apps with repeated `--app-id org.gnome.Calculator` options.
All commands accept `--help`.

## Pretrained image embeddings

The default handcrafted baseline uses alpha/footprint, color histograms, edge
strength, and low-resolution pixels. It establishes an inexpensive reference.
The CLIP path uses a frozen pretrained encoder with separate light/dark image
views, followed by the same per-criterion logistic-regression classifiers:

```sh
uv sync --extra embeddings
uv run --extra embeddings icon-review train --features clip --output artifacts/clip
uv run --extra embeddings icon-review predict /path/to/icon.png --model artifacts/clip/model.joblib
```

This downloads CPU PyTorch and the public `openai/clip-vit-base-patch32` weights on
first use. The resolved model revision is saved in the artifact and reused for
inference. Feature vectors are cached by image hash and feature specification.
Neither backend sends icons to an inference service.

## Labels and provenance

The five targets are `app-icon-footprint`, `app-icon-contrast`, `app-icon-detail`,
`app-icon-no-baked-in-shadows`, and `app-icon-in-line-with-contemporary-styles`.

- `true`: explicit pass; `false`: explicit fail; `null`/missing: unreviewed.
- Each criterion is trained only on its labeled examples. Unreviewed is never
  treated as failure. Known-stale labels are excluded.
- Overall label is false if any criterion fails, true if all five pass, otherwise
  unknown. Models predict individual criteria, not a calibrated joint probability.
- We save the original AppStream/review responses, collection time, canonical
  icon URL, exact downloaded bytes, SHA-256, decoded-pixel hash, and dimensions.
- The public API provides category metadata-change timestamps but **not review
  timestamps or historical reviewed icon bytes**. Current label/current icon
  pairings therefore have *unknown freshness*, and experiments use them with that
  limitation recorded. A metadata change alone does not prove a label is stale.
- The parser can identify stale labels when review `updated_at` fields are
  present. An authenticated/export workflow is not implemented yet.
- Category timestamps track changes to the AppStream `icon` URL, not necessarily
  visual changes to the image. `needed_to_pass_since` is a guideline effective
  date, not the date an app was reviewed.

Source semantics checked against the [public API](https://flathub.org/api/v2/docs),
[quality moderation route](https://github.com/flathub/website/blob/main/backend/app/routes/quality_moderation.py),
and [metadata tracking](https://github.com/flathub/website/blob/main/backend/app/quality_metadata.py)
on 2026-11-05.

## Evaluation

We make one deterministic 75/25 train/test split, grouped across app IDs, exact
decoded icons, and near-duplicate difference hashes. Each connected group stays
in one partition. Near-duplicate detection is heuristic: it can over-group simple
icons and miss related brands. Review the saved groups before drawing conclusions.

Logistic regression uses fixed hyperparameters and class balancing. Criteria
with fewer than five examples of either class in training are skipped. The model
is fitted only on training data, including feature scaling; the saved artifact is
the evaluated model, not a silently refitted all-data model.

`evaluation.json` reports failure precision/recall, false-alarm rate, confusion
counts, and failure average precision, alongside an always-pass reference.
Undefined metrics are `null`. Small or single-class holdouts cannot establish
useful accuracy. Prediction returns **uncalibrated scores**, not approval
decisions. Before deployment, use a larger independent test set and separate
validation data for calibration/threshold selection.

## Files

```text
src/flathub_icon_review/  collector, audit, feature extraction, training, CLI
tests/                   label, grouping, collection, and training checks
data/sample/             local dataset (gitignored)
  catalog.json           fixed catalog sampling frame
  selection.json         IDs selected by the latest collection invocation
  records/               per-app raw responses and normalized labels
  icons/                 original bytes, content-addressed
  features/              cached feature vectors
  audit.json             dataset counts
artifacts/baseline/       local model, split manifest, evaluation (gitignored)
```

Dataset/icon files retain their upstream ownership; they are not included in the
source repository. Only load trusted `model.joblib` files (pickle-based format).

## Development

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Next milestones: inspect label freshness with moderator exports; expand the
sample until each criterion has enough failures; compare CLIP with the cheap
baseline; manually audit duplicate groups and model errors; calibrate feedback
thresholds on separate validation data.
