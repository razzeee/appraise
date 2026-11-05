# Experiments — 2026-11-05

## Full-catalog expansion (completed)

The expansion finished at 22:07 UTC in `/home/razze/.t3/projects/icon-taste`.
Collection covered the cached 3,388-entry catalog: 3,357 icon records, 30 entries
without icons, and one unavailable quality-review endpoint (`org.winehq.Wine`,
HTTP 404). There are 3,029 apps with at least one reviewed icon criterion.

Both models were retrained using identical grouped splits: 2,273 training apps
and 756 held-out apps, with 2,888 image groups overall. Both saved models passed
local inference checks. Missing labels remain masked per criterion.

- `artifacts/full-baseline/model.joblib`: expanded handcrafted-feature model.
- `artifacts/full-clip/model.joblib`: expanded frozen-CLIP-feature model.
- Each model directory includes `evaluation.json` and `split.json`.
- `artifacts/expansion-run.json` records collection, audit, and completion status.

The audit found five criterion/group conflicts where identical decoded icons
have different review labels (IRPF detail, Freeciv footprint, KWallet shadows,
Okular contrast, and Tryton detail). These labels were retained in this run;
duplicate grouping keeps identical images in one partition. Review freshness
remains unknown for all public records.

The expanded holdout differs from the initial sample holdout below, so their
metrics are not a controlled measurement of improvement from more data. Both
expanded models remain exploratory and uncalibrated. Inspect label conflicts
and use separate validation data before tuning thresholds or selecting models.

## Initial 300-app experiment

## Dataset

Command: `uv run icon-review collect --limit 300` (seed 42).

- Catalog sampling frame: 3,388 app IDs.
- 298 icons collected; 270 apps have at least one reviewed icon criterion.
- Two selected entries have no AppStream icon:
  `org.flathub.flatpak-external-data-checker` and `radio.k0swe.Kel_Agent`.
- 298 distinct decoded images. No exact-duplicate label conflicts.
- All public review-label freshness values are unknown.

| Criterion | Pass | Fail | Unreviewed |
| --- | ---: | ---: | ---: |
| Footprint | 113 | 146 | 39 |
| Contrast | 141 | 76 | 81 |
| Detail | 145 | 77 | 76 |
| No baked-in shadows | 153 | 35 | 110 |
| Contemporary style | 124 | 120 | 54 |

## Handcrafted baseline

Command: `uv run icon-review train` (seed 42).

202 training apps and 68 held-out apps, with missing labels masked per criterion.
The duplicate-grouping heuristic found 270 separate groups among labeled apps.
All five classifiers had enough training labels. A local prediction smoke test
successfully loaded the saved model and produced all five scores.

| Criterion | Test failures / total | Failure precision | Failure recall | False-alarm rate |
| --- | ---: | ---: | ---: | ---: |
| Footprint | 42 / 66 | 77.8% | 66.7% | 33.3% |
| Contrast | 22 / 56 | 75.0% | 68.2% | 14.7% |
| Detail | 25 / 57 | 47.6% | 40.0% | 34.4% |
| No baked-in shadows | 10 / 48 | 45.5% | 50.0% | 15.8% |
| Contemporary style | 31 / 61 | 64.5% | 64.5% | 36.7% |

These are exploratory estimates from a small holdout, using uncalibrated scores
and a fixed 0.5 threshold. They are not deployment-quality accuracy estimates.
The baseline demonstrates a learnable signal, especially for footprint/contrast,
but its false alarms preclude automatic review decisions.

Local artifacts (gitignored):

- `data/sample/audit.json`: complete label counts and freshness audit.
- `artifacts/baseline/evaluation.json`: full metrics and always-pass reference.
- `artifacts/baseline/split.json`: exact app/image/label/split provenance.
- `artifacts/baseline/model.joblib`: evaluated model fitted on training data only.

## Frozen CLIP experiment

Command: `uv run --extra embeddings icon-review train --features clip --output artifacts/clip`.

Completed using the same 202/68 app split and fixed classifier hyperparameters.
Encoder: `openai/clip-vit-base-patch32`, revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Each icon produces concatenated
light- and dark-background embeddings. The encoder itself is not fine-tuned.

| Criterion | Failure precision | Failure recall | False-alarm rate |
| --- | ---: | ---: | ---: |
| Footprint | 71.1% | 64.3% | 45.8% |
| Contrast | 60.0% | 40.9% | 17.6% |
| Detail | 54.5% | 48.0% | 31.3% |
| No baked-in shadows | 42.9% | 30.0% | 10.5% |
| Contemporary style | 64.5% | 64.5% | 36.7% |

CLIP improves detail prediction in this small experiment, but does not
consistently outperform the handcrafted baseline. For shadows it flags fewer
approved icons but also catches fewer failures. This supports testing a hybrid
of explicit visual measurements and pretrained features, rather than assuming
pretraining alone solves the task. These differences have not been established
as statistically reliable.

The evaluated CLIP model, metrics, and split manifest are under `artifacts/clip/`.
Both pipelines now work end-to-end. Next: expand the dataset, inspect mistakes
and review freshness, then compare hybrid features using separate validation
data. Preserve a new independent test set for final claims; do not repeatedly
tune against this exploratory holdout.
