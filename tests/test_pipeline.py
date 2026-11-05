import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from flathub_icon_review.dataset import (
    CRITERIA,
    Client,
    audit,
    collect,
    freshness_from_review,
    labels_from_review,
    load_records,
    overall_label,
    usable_label,
    write_json,
)
from flathub_icon_review.features import handcrafted, views
from flathub_icon_review.model import failure_metrics, image_groups, predict, train


def test_partial_reviews_do_not_become_negative_labels():
    labels = labels_from_review(
        {
            "guidelines": [
                {"guideline_id": CRITERIA[0], "passed": True},
                {"guideline_id": CRITERIA[1], "passed": None},
            ]
        }
    )
    assert labels[CRITERIA[0]] is True
    assert all(labels[key] is None for key in CRITERIA[1:])
    assert overall_label(labels) is None
    labels[CRITERIA[1]] = False
    assert overall_label(labels) is False
    assert overall_label(dict.fromkeys(CRITERIA, True)) is True
    with pytest.raises(ValueError):
        labels_from_review({"guidelines": [{"guideline_id": CRITERIA[0], "passed": 0}]})


def test_stale_labels_excluded_without_inventing_public_review_dates():
    review = {
        "metadata_changed_at": {"app-icon": "2026-10-01T00:00:00"},
        "guidelines": [
            {"guideline_id": CRITERIA[0], "updated_at": "2026-09-01T00:00:00Z"},
            {"guideline_id": CRITERIA[1], "updated_at": "2026-10-02T00:00:00Z"},
            {"guideline_id": CRITERIA[2]},
        ],
    }
    freshness = freshness_from_review(review)
    assert [freshness[key] for key in CRITERIA[:3]] == ["stale", "current", "unknown"]
    record = {"freshness": freshness, "labels": dict.fromkeys(CRITERIA, True)}
    assert usable_label(record, CRITERIA[0]) is None
    assert usable_label(record, CRITERIA[2]) is True


def test_collection_preserves_bytes_labels_and_resumes(tmp_path, monkeypatch):
    buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (255, 0, 0, 128)).save(buf, format="PNG")
    payload = buf.getvalue()
    calls = []
    has_icon = True

    def handle(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/quality-moderation/org.example.App"):
            return httpx.Response(
                200,
                json={
                    "guidelines": [
                        {"guideline_id": CRITERIA[0], "passed": False},
                        {"guideline_id": CRITERIA[1], "passed": None},
                    ]
                },
            )
        if request.url.path.endswith("/appstream/org.example.App"):
            app = {"icon": "https://example.org/icon.png"} if has_icon else {}
            return httpx.Response(200, json=app)
        if request.url.path == "/icon.png":
            return httpx.Response(200, content=payload)
        raise AssertionError(request.url)

    real_init = Client.__init__

    def mocked_init(self, interval):
        real_init(self, interval)
        self.http.close()
        self.http = httpx.Client(transport=httpx.MockTransport(handle))

    monkeypatch.setattr(Client, "__init__", mocked_init)
    assert collect(tmp_path, 0, 42, 0, ["org.example.App"])["saved"] == 1
    (record,) = load_records(tmp_path)
    assert (tmp_path / record["icon_path"]).read_bytes() == payload
    assert record["labels"][CRITERIA[0]] is False
    assert record["labels"][CRITERIA[1]] is None
    assert record["freshness"][CRITERIA[0]] == "unknown"
    assert audit(tmp_path)["criteria"][CRITERIA[0]]["fail"] == 1
    assert collect(tmp_path, 0, 42, 0, ["org.example.App"])["saved"] == 0
    assert len(calls) == 3
    has_icon = False
    no_icon_root = tmp_path / "no-icon"
    result = collect(no_icon_root, 0, 42, 0, ["org.example.App"])
    assert result == {"saved": 0, "skipped": 1, "errors": 0, "total": 0}
    assert len(calls) == 5
    assert json.loads((no_icon_root / "collection-skipped.json").read_text()) == [
        {"app_id": "org.example.App", "reason": "No AppStream icon"}
    ]


def test_retries_transient_errors_but_not_missing_apps(monkeypatch):
    monkeypatch.setattr("flathub_icon_review.dataset.time.sleep", lambda _: None)
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(503 if len(requests) < 3 else 200, json={})

    client = Client(0)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handle))
    assert client.get("https://example.org").status_code == 200
    assert len(requests) == 3
    client.http.close()
    client.http = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(404))
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.get("https://example.org")
    client.http.close()


def test_transparency_views_and_features(tmp_path):
    path = tmp_path / "transparent.png"
    Image.new("RGBA", (32, 16), (255, 0, 0, 0)).save(path)
    light, dark = views(path)
    assert light.getpixel((0, 0)) == (250, 250, 250)
    assert dark.getpixel((0, 0)) == (36, 36, 36)
    assert np.isfinite(handcrafted(path)).all()


def test_resized_icons_are_grouped(tmp_path):
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
    image.resize((128, 128)).save(tmp_path / "one.png")
    image.resize((256, 256)).save(tmp_path / "two.png")
    records = [
        {"app_id": "org.example.One", "pixel_sha256": "one", "icon_path": "one.png"},
        {"app_id": "org.example.Two", "pixel_sha256": "two", "icon_path": "two.png"},
    ]
    groups = image_groups(tmp_path, records)
    assert groups[0] == groups[1]


def test_failure_metric_direction():
    result = failure_metrics([True, True, False, False], [0.9, 0.1, 0.2, 0.8])
    assert result["failure_precision"] == 0.5
    assert result["failure_recall"] == 0.5
    assert result["false_alarm_rate"] == 0.5
    always_pass = failure_metrics([True, False], [1, 1])
    assert always_pass["failure_precision"] is None
    assert always_pass["failure_recall"] == 0


def test_audit_exposes_conflicting_reviews_for_identical_images(tmp_path):
    for name, value in [("one", True), ("two", False), ("three", None)]:
        write_json(
            tmp_path / "records" / f"{name}.json",
            {
                "app_id": name,
                "pixel_sha256": "identical",
                "labels": {**dict.fromkeys(CRITERIA), CRITERIA[0]: value},
                "freshness": dict.fromkeys(CRITERIA, "unknown"),
            },
        )
    result = audit(tmp_path)
    assert result["unique_pixel_images"] == 1
    assert result["conflicting_duplicate_labels"] == [
        {
            "pixel_sha256": "identical",
            "criterion": CRITERIA[0],
            "apps": ["one", "three", "two"],
        }
    ]


def test_training_prediction_and_holdout_isolation(tmp_path):
    rng = np.random.default_rng(42)
    data = tmp_path / "data"
    data.mkdir()
    for i in range(80):
        image = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        image.save(data / f"{i}.png")
        labels = dict.fromkeys(CRITERIA, bool(i % 2))
        # Unreviewed labels must be omitted from that criterion's training set.
        labels[CRITERIA[-1]] = None
        record = {
            "app_id": f"org.example.App{i}",
            "icon_path": f"{i}.png",
            "icon_sha256": str(i),
            "pixel_sha256": str(i),
            "labels": labels,
            "freshness": dict.fromkeys(CRITERIA, "unknown"),
        }
        write_json(data / "records" / f"{i}.json", record)
    output = tmp_path / "model"
    result = train(data, output, "handcrafted", 42)
    assert result["train_apps"] + result["test_apps"] == 80
    assert "skipped" in result["criteria"][CRITERIA[-1]]
    split = json.loads((output / "split.json").read_text())["apps"]
    train_groups = {r["group"] for r in split if r["split"] == "train"}
    test_groups = {r["group"] for r in split if r["split"] == "test"}
    assert train_groups.isdisjoint(test_groups)
    prediction = predict(output / "model.joblib", data / "0.png")
    assert 0 <= prediction["criteria"][CRITERIA[0]]["pass_score"] <= 1
    assert "unavailable" in prediction["criteria"][CRITERIA[-1]]
