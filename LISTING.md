# Name, summary, and branding review

The listing pipeline extends the icon classifier with ten additional Flathub
criteria: three deterministic backend checks and seven learned review targets.
Everything runs locally. Text classifiers are lightweight character n-gram
baselines, not language models or claims of human-level semantic understanding.

## Train from a fresh checkout

```sh
uv sync
uv run icon-review collect --limit 0
uv run icon-review audit-listing
uv run icon-review train-listing --output artifacts/listing
```

If collection reports an error, repeat the collection command to resume.
If you already have `data/sample/records/`, start with `audit-listing`. The raw
AppStream and review responses in those snapshots already contain these fields;
no recollection or destructive data migration is needed. Existing icon commands,
artifacts, and label semantics are preserved. Use `--extra embeddings` with uv
if you also use CLIP; plain `uv sync` can remove that optional dependency set.

Training writes `model.joblib`, `evaluation.json`, and `split.json` under the
output directory. Audit writes `data/sample/listing-audit.json`. Models and
datasets remain gitignored; copy the model to another computer along with a
checkout of the code that supports it. The `v0.2.0` release bundle includes the
listing and calibrated icon models; see [README.md](README.md) for download
commands. The older `model-v1` release contains only the original icon classifier.

## Review a collected app

### Review a live Flathub page instead

```sh
uv run --extra embeddings icon-review review-url \
  https://flathub.org/apps/org.gnome.Calculator
```

This fetches fresh default-language AppStream metadata and its canonical icon
from Flathub's public API, then runs the listing and icon models. Localized page
links (such as `/en/apps/...`) and older `/apps/details/...` links also work.
Page locale does not change model input: these classifiers use English/default
metadata. Query strings and fragments are ignored.

The defaults are `artifacts/listing/model.joblib` and
`artifacts/calibrated-clip/model.joblib`. Both must already exist locally; use
`--model` and `--icon-model` to choose other trained artifacts. For example, an
existing baseline icon model works without installing the optional CLIP extras.
The command does not train models or download model artifacts automatically.

Results include the fetched name, summary, branding, app ID, source URLs, and
fetch timestamp. Icon bytes are temporary and deleted after scoring. Training
snapshots are not changed, and public review labels are not fetched or used as
prediction inputs. If Flathub has no icon, text checks still run and icon checks
are marked unavailable. HTTP failures are reported rather than falling back to
old cached data.

### Review the saved snapshot

```sh
uv run icon-review review --app-id org.gnome.Calculator \
  --model artifacts/listing/model.joblib
```

This reads the **saved snapshot**, including its downloaded icon. It does not
fetch the app's latest metadata. Collect a new snapshot in a different `--data`
directory when you need fresh content.

To include the five existing icon criteria in the same JSON output:

```sh
uv run --extra embeddings icon-review review --app-id org.gnome.Calculator \
  --model artifacts/listing/model.joblib \
  --icon-model artifacts/calibrated-clip/model.joblib
```

## Review a new app or proposed changes

Copy and edit [`examples/listing.json`](examples/listing.json), then run:

```sh
uv run icon-review review examples/listing.json \
  --icon /path/to/icon.png --model artifacts/listing/model.joblib
```

The input is an AppStream-style JSON object with English/default-language `name`
and `summary` strings, and an optional `branding` array. Branding entries use
`type: "primary"`, `scheme_preference: "light"` or `"dark"`, and a hexadecimal
color `value` (`#RRGGBB` or `#RGB`). A theme-independent primary entry is a
fallback for both themes; explicit theme colors take precedence. Duplicate
theme entries and malformed colors make learned branding scoring unavailable.
Icon URLs in JSON are not downloaded; pass a local file using `--icon`.

Omit `--model` to run rules, measurements, and hints without a trained model:

```sh
uv run icon-review review examples/listing.json
```

Without an icon, name and summary scoring still work, but branding-quality
scoring is unavailable. Missing text, missing colors, transparent icons, skipped
classifiers, and insufficient calibration labels are explicit in the output.
`--icon-model` requires an icon (either `--icon` or a collected app's icon).

## Results and criteria

### Print moderation API request bodies

```sh
uv run --extra embeddings icon-review review-url \
  https://flathub.org/apps/org.gnome.Calculator \
  --format moderation-json
```

This prints a JSON array of individual request bodies, for example:

```json
[
  {"guideline_id": "app-icon-contrast", "passed": true},
  {"guideline_id": "app-name-just-the-name", "passed": false}
]
```

Each object is a body for `POST
https://flathub.org/api/v2/quality-moderation/{app_id}`. The endpoint accepts one
object per request, **not the whole array**. This mode only prints payloads; it
does not authenticate or submit requests. It emits all available supported
writable targets, not a diff against current moderation values.

`--pass-threshold 0.5` is the default: scores at or above the threshold become
`true`, and lower scores become `false`. Set a different threshold explicitly
if needed. This is a score conversion, not a validated deployment threshold;
uncalibrated model scores are also converted. Automatic backend-owned rules and
unavailable predictions are omitted. With no available model scores the output
is `[]`. There is no null/uncertain write in this API.

The same flags work with `review` for local JSON or saved `--app-id` snapshots.

### Full review output

`criteria` maps the original Flathub guideline IDs to results:

- **Rules:** `method: "rule"` and `passed`, with lengths and limits where relevant.
- **Learned targets:** `method: "model"`, `pass_score`, and `calibrated`.
- **Unavailable:** an `unavailable` explanation, never an invented zero score.

`hints` explains observable issues such as trailing punctuation, repeating the
name, leading articles, empty text, or identical theme colors. Hints are
heuristics, separate from learned judgments and their brand-specific exceptions.
`measurements` exposes text counts and color/contrast features used by the model.
Contrast statistics describe icon pixels over each brand background; the 1.5
measurement cutoff is a feature, not a WCAG requirement or an approval rule.

| Category | Automatic rules | Learned targets |
| --- | --- | --- |
| Name | At most 20 characters | Just the name; no weird formatting |
| Summary | At most 35 characters | Not technical; no weird formatting; don't repeat the name; don't start with an article |
| Branding | Has a primary color entry | Good primary brand colors, with both themes and the icon as inputs |

The automatic rules follow Flathub's backend, including its edge cases. The
written guidelines recommend a name of at most 15 characters and say shorter
than 20, whereas the backend accepts 20. The backend presence rule accepts any
primary color entry, even though the guidelines recommend distinct light/dark
colors. Empty strings also pass the backend length check; they generate a hint
and cannot be scored by the learned text models. Invalid color values can pass
the backend presence check but cannot be used for learned branding scoring.

Sources checked for this implementation:

- [Written quality guidelines](https://github.com/flathub-infra/documentation/blob/main/docs/02-for-app-authors/03-metainfo-guidelines/01-quality-guidelines.md)
- [Automatic backend checks](https://github.com/flathub/website/blob/main/backend/app/worker/update_quality_moderation.py)
- [Metadata-change categories](https://github.com/flathub/website/blob/main/backend/app/quality_metadata.py)

## Learning, splitting, and limitations

Name and summary models combine training-only TF-IDF character n-grams with
numeric text measurements. Summary features include the app name for repetition
checks. Branding models measure light/dark color properties, their difference,
and their contrast and color distance to visible icon pixels. All learned
targets use class-balanced logistic regression; each criterion requires five
training examples per class. Labels, comments, app IDs, review metadata, and
descriptions are never predictive features.

Splitting joins image duplicate groups, identical normalized names, and identical
normalized summaries into connected groups. A 75/25 development/test split is
followed by a 75/25 training/calibration split of development groups. Group sizes
mean app proportions vary. This is a new listing partition over the full cached
cohort, not the original icon test partition. Exact-text grouping can miss
related titles, and image grouping remains heuristic. The snapshot's collection
currently includes only apps with downloadable icons.

Feature vocabularies and scaling fit on training rows only. Separate unweighted
sigmoid calibrators fit on calibration rows with at least five examples of each
class. Otherwise, predictions explicitly identify their uncalibrated scores.
Evaluation includes raw and calibrated Brier scores, log loss, failure metrics,
an always-pass reference, class counts, and rule agreement with recorded labels.
The threshold remains 0.5; it is not tuned against test results. Calibration is
not proof of reliable probabilities, especially with few failures.

Unreviewed labels are excluded, not counted as failures. Known-stale labels are
excluded using category timestamps, with name changes also affecting summary
freshness and icon changes also affecting branding freshness. The public API
does not normally expose review timestamps or historical reviewed content, so
freshness remains unknown for the current snapshot. Audit includes conflicting
labels for identical inputs and counts of missing inputs. Models approximate
public review outcomes, including noisy labels and subjective exceptions; they
are review assistance rather than automatic approval decisions.
