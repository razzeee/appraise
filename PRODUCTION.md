# Production use

## Supported use: assisted quality review

Appraise runs locally, does not upload app inputs to an inference service, and
does not change Flathub moderation values. A moderator must inspect the app and
approve any proposed payload. Model scores, including calibrated scores, are
not approval guarantees. The default 0.5 export threshold is not a validated
deployment policy.

## Installation and verification

1. Use a tagged release and its model bundle from the same GitHub repository.
2. Verify `SHA256SUMS` before extraction and use `uv sync --locked --extra
   embeddings` for the matching environment. Checksums detect corruption; obtain
   them from a trusted release. Never load model files from an untrusted source:
   joblib models use pickle and can execute code when loaded.
3. Run `uv run --extra embeddings appraise --version` and a known app review.
4. Preserve the full review JSON, app ID, fetched metadata, and model bundle
   manifest for audit. Moderation-only output intentionally omits this context.
5. Inspect proposed changes before making any authenticated API requests.

For reproducible offline operation, download the pinned CLIP weights once, then
use local metadata with `HF_HUB_OFFLINE=1`. Live URL review necessarily requires
network access. Never put authentication cookies or tokens in review metadata.

## Resource and failure handling

- Live URL review accepts HTTPS Flathub app links and downloads only from
  `flathub.org` and `dl.flathub.org`, checking redirect destinations as well.
- Downloads are capped at 16 MiB of decoded response bytes. Redirect chains are
  capped at six responses. Transient requests have bounded retries and backoff;
  Retry-After delays are capped at 30 seconds. Each request has a 30-second
  inactivity timeout. This is not a whole-command wall-clock deadline.
- Decoded icons are limited to 16 megapixels. Unsupported/oversized inputs
  produce errors or explicit unavailable criteria, never invented scores.
- Temporary live-review icons are cleaned up. Download failures do not fall
  back to old data. JSON output is written only after successful inference.
- Cached features and model files are published using atomic replacement.
  Unreadable feature caches are regenerated. Training with no viable
  classifiers preserves existing output.
- Train into a **new output directory**, validate all three output files, and
  switch deployments only after validation. Model, split, and evaluation files
  are individually atomic, not a transactional directory. Do not train into an
  actively served artifact directory or run concurrent training into one output.
- Run the CLI with least privilege, under an external job timeout and memory
  limit when processing untrusted inputs. The image decoder and pickle runtime
  are not security sandboxes. Keep dependencies updated and rerun validation.

## Release acceptance checklist

Every release must be self-contained: attach its wheel, source distribution,
complete model bundle, and `SHA256SUMS` covering every attached package and bundle.
The model bundle must contain all three supported model pipelines (calibrated
CLIP, calibrated baseline, and listing), each with `model.joblib`,
`evaluation.json`, and `split.json`, plus a provenance manifest with per-file
checksums. Include unchanged artifacts too; never require users to fetch files
from an earlier release. Verify the uploaded asset inventory before announcing
the release. Pretrained third-party CLIP encoder weights remain a separately
documented first-use dependency.

```sh
uv sync --locked --extra embeddings
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
uv build
```

CI runs lint, formatting, tests, and package builds on Python 3.12 and 3.13.
Before publishing model assets, also verify their checksums, load the release
bundle in a fresh wheel installation, and run combined review and JSON export.
Verify CLIP separately with the pinned encoder weights; the base CI job does
not install or download that optional encoder. Keep earlier releases for rollback.

## Not yet supported: unattended moderation

Production automation that writes review decisions is blocked on independent
validation, not just software tests. The public snapshot does not establish
that labels describe the exact current content. Existing test results are
exploratory and some failure recall is low.

Before enabling automated writes, a quality moderator must supply a new,
independently reviewed dataset tied to exact icon bytes and metadata, plus an
acceptable per-criterion false-pass/false-fail policy. Use separate data to
select thresholds and abstention bands, and untouched data to verify the policy.
Track drift and audit samples after deployment. Existing evidence is insufficient
to choose safe automatic-write thresholds; Appraise deliberately has no API
write command.
