"""Snapshot collection and label semantics. Unknown is never a negative label."""

import hashlib
import io
import json
import random
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
from PIL import Image

API = "https://flathub.org/api/v2"
CRITERIA = (
    "app-icon-footprint",
    "app-icon-contrast",
    "app-icon-detail",
    "app-icon-no-baked-in-shadows",
    "app-icon-in-line-with-contemporary-styles",
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def labels_from_review(review: dict, criteria=CRITERIA) -> dict:
    labels = dict.fromkeys(criteria)
    for guideline in review["guidelines"]:
        key = guideline["guideline_id"]
        if key in labels:
            value = guideline["passed"]
            if value is not None and type(value) is not bool:
                raise ValueError(f"Unexpected label for {key}: {value!r}")
            labels[key] = value
    return labels


def overall_label(labels: dict):
    if any(value is False for value in labels.values()):
        return False
    if all(labels.get(key) is True for key in CRITERIA):
        return True
    return None


def freshness_from_review(review: dict, criteria=CRITERIA) -> dict:
    """Without review timestamps, freshness is unknown, not implicitly current."""
    freshness = dict.fromkeys(criteria, "unknown")
    for guideline in review["guidelines"]:
        key = guideline["guideline_id"]
        category = guideline.get("guideline", {}).get("category", "app-icon")
        changed = (review.get("metadata_changed_at") or {}).get(category)
        updated = guideline.get("updated_at")
        if key in freshness and changed and updated:

            def parse(value):
                result = datetime.fromisoformat(value)
                return result.replace(tzinfo=UTC) if result.tzinfo is None else result

            freshness[key] = "stale" if parse(updated) < parse(changed) else "current"
    return freshness


def usable_label(record: dict, criterion: str):
    if record["freshness"][criterion] == "stale":
        return None
    return record["labels"][criterion]


def load_records(root: Path) -> list[dict]:
    return [
        json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))
    ]


class Client:
    def __init__(self, interval: float):
        self.interval = interval
        self.last_request = 0.0
        self.http = httpx.Client(
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "appraise/0.2 (dataset experiment)"},
        )

    def get(self, url):
        for attempt in range(4):
            time.sleep(max(0, self.interval - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.http.get(url)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in (429, 500, 502, 503, 504):
                    raise
                if attempt == 3:
                    raise
                retry_after = error.response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 2**attempt
            except httpx.TransportError:
                if attempt == 3:
                    raise
                delay = 2**attempt
            time.sleep(delay)
        raise RuntimeError("Request retries exhausted")


def collect(root: Path, limit: int, seed: int, interval: float, app_ids: list[str]):
    root.mkdir(parents=True, exist_ok=True)
    client = Client(interval)
    failures = []
    skipped = []
    completed = 0
    try:
        catalog_path = root / "catalog.json"
        if app_ids:
            ids = list(dict.fromkeys(app_ids))
        else:
            if not catalog_path.exists():
                write_json(catalog_path, client.get(f"{API}/appstream").json())
            ids = sorted(json.loads(catalog_path.read_text()))
            random.Random(seed).shuffle(ids)
        if limit:
            ids = ids[:limit]
        write_json(root / "selection.json", {"seed": seed, "app_ids": ids})
        for index, app_id in enumerate(ids, 1):
            if not re.fullmatch(r"[A-Za-z_][\w.\-]+", app_id):
                raise ValueError(f"Invalid app ID: {app_id!r}")
            path = root / "records" / f"{app_id}.json"
            if path.exists():
                continue
            try:
                review = client.get(f"{API}/quality-moderation/{app_id}").json()
                app = client.get(f"{API}/appstream/{app_id}").json()
                # Use the same canonical icon displayed by the website.
                url = app.get("icon")
                if not url:
                    skipped.append({"app_id": app_id, "reason": "No AppStream icon"})
                    continue
                image_bytes = client.get(url).content
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    rgba = image.convert("RGBA")
                    width, height = rgba.size
                    pixels_hash = hashlib.sha256(
                        f"{width}x{height}:".encode() + rgba.tobytes()
                    ).hexdigest()
                digest = hashlib.sha256(image_bytes).hexdigest()
                image_path = root / "icons" / digest
                image_path.parent.mkdir(exist_ok=True)
                image_path.write_bytes(image_bytes)
                labels = labels_from_review(review)
                record = {
                    "schema_version": 1,
                    "app_id": app_id,
                    "collected_at": datetime.now(UTC).isoformat(),
                    "icon_url": url,
                    "icon_sha256": digest,
                    "pixel_sha256": pixels_hash,
                    "icon_path": str(image_path.relative_to(root)),
                    "width": width,
                    "height": height,
                    "labels": labels,
                    "overall_label": overall_label(labels),
                    "freshness": freshness_from_review(review),
                    "review": review,
                    "appstream": app,
                }
                write_json(path, record)
                completed += 1
            except (httpx.HTTPError, ValueError, KeyError, OSError) as error:
                failures.append({"app_id": app_id, "error": str(error)})
                print(f"[{index}/{len(ids)}] {app_id}: {error}", flush=True)
            if index % 25 == 0 or index == len(ids):
                print(
                    f"[{index}/{len(ids)}] saved {completed}, errors {len(failures)}",
                    flush=True,
                )
    finally:
        client.http.close()
        write_json(root / "collection-errors.json", failures)
        write_json(root / "collection-skipped.json", skipped)
    return {
        "saved": completed,
        "skipped": len(skipped),
        "errors": len(failures),
        "total": len(load_records(root)),
    }


def audit(root: Path) -> dict:
    records = load_records(root)
    counts = {}
    for criterion in CRITERIA:
        counts[criterion] = {
            "pass": sum(r["labels"][criterion] is True for r in records),
            "fail": sum(r["labels"][criterion] is False for r in records),
            "unreviewed": sum(r["labels"][criterion] is None for r in records),
            "freshness": dict(Counter(r["freshness"][criterion] for r in records)),
        }
    hashes = Counter(r["pixel_sha256"] for r in records)
    conflicts = []
    for digest, count in hashes.items():
        if count < 2:
            continue
        duplicates = [r for r in records if r["pixel_sha256"] == digest]
        for criterion in CRITERIA:
            values = {usable_label(r, criterion) for r in duplicates} - {None}
            if len(values) > 1:
                conflicts.append(
                    {
                        "pixel_sha256": digest,
                        "criterion": criterion,
                        "apps": [r["app_id"] for r in duplicates],
                    }
                )
    result = {
        "apps": len(records),
        "unique_pixel_images": len(hashes),
        "duplicate_image_groups": sum(n > 1 for n in hashes.values()),
        "conflicting_duplicate_labels": conflicts,
        "reviewed_apps": sum(
            any(v is not None for v in r["labels"].values()) for r in records
        ),
        "criteria": counts,
    }
    write_json(root / "audit.json", result)
    return result
