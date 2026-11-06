# Experiments — 2026-11-05

## Full-catalog expansion (completed)

The expansion finished at 22:07 UTC in the project checkout.
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

Command: `uv run appraise collect --limit 300` (seed 42).

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

Command: `uv run appraise train` (seed 42).

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

Command: `uv run --extra embeddings appraise train --features clip --output artifacts/clip`.

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

## Separate calibration experiment

Both backends were retrained with 1,693 training apps and 580 calibration apps,
preserving the existing 756-app test partition. The two split manifests were
verified identical, with no image groups shared across partitions. Unweighted
sigmoid calibration was fitted separately for all five criteria. Saved-model
prediction was verified for both backends, including calibration status.

Local outputs: `artifacts/calibrated-baseline/` and `artifacts/calibrated-clip/`.
Original models and the uncalibrated `model-v1` release are preserved.

Test Brier scores (lower is better), raw versus calibrated scores from the
same newly trained classifiers:

| Criterion | Baseline raw | Baseline calibrated | CLIP raw | CLIP calibrated |
| --- | ---: | ---: | ---: | ---: |
| Contrast | 0.1578 | 0.1515 | 0.1959 | 0.1779 |
| Detail | 0.2536 | 0.2137 | 0.2363 | 0.1899 |
| Footprint | 0.1469 | 0.1480 | 0.2256 | 0.1997 |
| Contemporary style | 0.1952 | 0.1972 | 0.2271 | 0.1915 |
| No baked-in shadows | 0.1823 | 0.1479 | 0.1842 | 0.1444 |

Log loss decreased for every criterion in both backends. Brier score improved
for all CLIP criteria, but worsened slightly for baseline footprint and style.
This does not establish perfect calibration or production reliability. The
test set has been inspected in earlier experiments, label freshness remains
unknown, and comparisons to older models also reflect the smaller training set.
Thresholds and classifier hyperparameters were not tuned against these results.

## Listing review: name, summary, and branding

The first listing run completed over the 3,357 cached AppStream/review snapshots.
Its 3,089 connected image/text groups produced 1,873 training apps, 614 calibration
apps, and 870 test apps. All seven learned targets were trained and calibrated.
The three automatic rules exactly matched recorded labels on the test partition.
The listing partition is separate from the earlier icon partition.

Held-out results at the fixed 0.5 threshold:

| Criterion | Labeled train/calibration/test | Calibrated Brier | Failure precision | Failure recall |
| --- | --- | ---: | ---: | ---: |
| Just the name | 1502 / 483 / 705 | 0.0640 | 64.9% | 32.9% |
| Name formatting | 1535 / 487 / 713 | 0.0664 | 76.2% | 38.1% |
| Nontechnical summary | 1526 / 495 / 707 | 0.1595 | 64.2% | 39.9% |
| Summary formatting | 1426 / 472 / 675 | 0.1031 | 81.4% | 58.9% |
| No repeated name | 1380 / 460 / 650 | 0.0704 | 67.5% | 40.3% |
| No leading article | 1451 / 488 / 677 | 0.0533 | 96.1% | 74.2% |
| Good brand colors | 529 / 196 / 268 | 0.1926 | 90.0% | 20.0% |

These results expose limited recall, especially for branding; a high pass score
is not an approval guarantee. Character n-grams are a lightweight text baseline,
not a semantic language model. No thresholds or hyperparameters were tuned after
examining these test results.

Audit found two conflicting identical-input groups: the name “2048” across two
apps and branding across five NetXMS variants. All review freshness is unknown.
Of 1,004 labeled branding reviews, 993 have supported colors and usable icons;
other branding rows lack both resolved themes or contain unsupported hex colors.
The collector's icon requirement also limits coverage of otherwise usable text.

Artifacts are in `artifacts/listing/`; the detailed audit is in
`data/sample/listing-audit.json`. Saved-model inference was verified through the
CLI for both custom metadata and a collected app. Combined review of
`org.gnome.Calculator` produced all ten listing criteria plus all five calibrated
icon criteria. See [LISTING.md](LISTING.md) for reproduction commands and input
semantics. These artifacts are included in the `v0.2.0` release bundle.
