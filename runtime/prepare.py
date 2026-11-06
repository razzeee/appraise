"""Verify release models and prefetch the encoder before exporting the runtime."""

import hashlib
import json
import tarfile
from pathlib import Path

from flathub_icon_review.artifacts import load_model
from flathub_icon_review.features import Extractor

root = Path("/opt/appraise")
release = json.loads(Path("/tmp/release.json").read_text())
identity = release["identity"]
bundle = Path("/tmp/models.tar.gz")
if hashlib.sha256(bundle.read_bytes()).hexdigest() != identity["bundle"]:
    raise ValueError("Model bundle checksum does not match the runtime release")
with tarfile.open(bundle) as archive:
    archive.extractall(root, filter="data")

icon = root / "artifacts/calibrated-clip/model.joblib"
listing = root / "artifacts/listing/model.joblib"
baseline = root / "artifacts/calibrated-baseline/model.joblib"
artifact = load_model(icon, "icon")
load_model(listing, "listing")
load_model(baseline, "icon")
revision = identity["encoder"]
if artifact["features"].get("revision") != revision:
    raise ValueError("Encoder revision must match the model bundle")
Extractor("clip", revision=revision)
(root / "deployment.json").write_text(
    json.dumps(
        {
            "identity": identity,
            "source_commit": release["commit"],
            "files": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (icon, listing)
            },
        },
        indent=2,
    )
)
