import json

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from flathub_icon_review.cli import main
from flathub_icon_review.dataset import write_json
from flathub_icon_review.listing import (
    audit_listing,
    listing_groups,
    listing_records,
    prepare,
    review_listing,
    train_listing,
)
from flathub_icon_review.listing_features import (
    LEARNED_CRITERIA,
    LISTING_CRITERIA,
    brand_measurements,
    category,
    inputs,
    primary_colors,
    rules,
)


def metadata():
    return {
        "name": "Kitsune",
        "summary": "Read stories",
        "branding": [
            {"type": "primary", "scheme_preference": "light", "value": "#aaddff"},
            {"type": "primary", "scheme_preference": "dark", "value": "#123456"},
        ],
    }


def test_rules_match_backend_boundaries_and_presence():
    app = metadata()
    for field, boundary in (("name", 20), ("summary", 35)):
        app[field] = "x" * boundary
        assert rules(app)[f"app-{field}-not-too-long"]["passed"]
        app[field] += "x"
        assert not rules(app)[f"app-{field}-not-too-long"]["passed"]
    assert rules(app)["branding-has-primary-brand-colors"]["passed"]
    # Upstream's automatic presence rule accepts even one primary entry.
    app["branding"] = app["branding"][:1]
    assert rules(app)["branding-has-primary-brand-colors"]["passed"]
    assert set(primary_colors(app)) == {"light"}
    assert all(not r["passed"] for r in rules({}).values())


def test_color_resolution_contrast_and_invalid_inputs(tmp_path):
    app = metadata()
    icon = tmp_path / "icon.png"
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(icon)
    measurements = brand_measurements(app, icon)
    assert measurements["dark_contrast_p50"] > measurements["light_contrast_p50"]
    assert measurements["light_minus_dark_luminance"] > 0
    app["branding"] = [{"type": "primary", "value": "#abc"}]
    assert np.array_equal(primary_colors(app)["light"], primary_colors(app)["dark"])
    app["branding"].append(
        {"type": "primary", "scheme_preference": "dark", "value": "#000"}
    )
    assert primary_colors(app)["dark"].sum() == 0
    app["branding"][0]["value"] = "not a color"
    prepared, missing = inputs(app, icon)
    assert "branding" not in prepared
    assert "Invalid light primary color" in missing["branding"]
    with pytest.raises(ValueError, match="name must be a string"):
        inputs({"name": {"en": "Name"}})
    with pytest.raises(ValueError, match="branding must be a list"):
        inputs({"branding": "#fff"})
    app = metadata()
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(icon)
    assert "visible icon" in inputs(app, icon)[1]["branding"]


def test_rules_only_review_and_actionable_hints():
    app = metadata()
    app["summary"] = "The Kitsune reader."
    result = review_listing(app)
    assert result["criteria"]["app-name-not-too-long"]["method"] == "rule"
    assert "unavailable" in result["criteria"]["app-summary-not-technical"]
    messages = " ".join(h["message"] for h in result["hints"])
    assert "full stop" in messages and "repeating" in messages and "article" in messages
    assert result["measurements"]["summary"]["repeats_name"] == 1
    app["name"] = "Kit"
    assert review_listing(app)["measurements"]["summary"]["repeats_name"] == 0


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    rng = np.random.default_rng(52)
    for i in range(160):
        icon = f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            root / icon
        )
        app = metadata()
        app.update(name=f"Name {i}", summary=f"Read stories number {i}")
        review = {
            "metadata_changed_at": {},
            "guidelines": [
                {
                    "guideline_id": key,
                    "passed": bool(i % 2),
                    "guideline": {
                        "category": "branding"
                        if category(key) == "branding"
                        else f"app-{category(key)}"
                    },
                }
                for key in LISTING_CRITERIA
            ],
        }
        write_json(
            root / "records" / f"{i:03}.json",
            {
                "app_id": f"org.example.App{i}",
                "appstream": app,
                "review": review,
                "icon_path": icon,
                "pixel_sha256": str(i),
                "icon_sha256": str(i),
            },
        )
    return root


def test_category_freshness_and_duplicate_grouping(dataset):
    path = dataset / "records/000.json"
    first = json.loads(path.read_text())
    first["review"]["metadata_changed_at"] = {
        "app-name": "2026-02-01",
        "app-icon": "2026-03-01",
    }
    for guideline in first["review"]["guidelines"]:
        guideline["updated_at"] = "2026-01-01"
    write_json(path, first)
    second_path = dataset / "records/001.json"
    second = json.loads(second_path.read_text())
    second["appstream"]["name"] = " NAME 0 "
    write_json(second_path, second)
    records = listing_records(dataset)
    assert all(value == "stale" for value in records[0]["freshness"].values())
    assert all(value == "unknown" for value in records[1]["freshness"].values())
    groups = listing_groups(dataset, records)
    assert groups[0] == groups[1]
    # Parsing new targets must not rewrite the original snapshot.
    assert "labels" not in json.loads(path.read_text())


def test_listing_training_isolation_serialization_and_legacy_icon_independence(
    dataset, tmp_path
):
    output = tmp_path / "listing"
    report = train_listing(dataset, output)
    assert sum(report[f"{s}_apps"] for s in ("train", "calibration", "test")) == 160
    split = json.loads((output / "split.json").read_text())["apps"]
    sets = {
        s: {r["group"] for r in split if r["split"] == s}
        for s in ("train", "calibration", "test")
    }
    assert sets["train"].isdisjoint(sets["calibration"] | sets["test"])
    assert sets["calibration"].isdisjoint(sets["test"])
    artifact = joblib.load(output / "model.joblib")
    assert set(artifact["models"]) == set(LEARNED_CRITERIA)
    assert set(artifact["calibrators"]) == set(LEARNED_CRITERIA)
    records = listing_records(dataset)
    prepared, _ = prepare(dataset, records)
    key = "app-name-just-the-name"
    train = [i for i, r in enumerate(split) if r["split"] == "train"]
    calibration = [i for i, r in enumerate(split) if r["split"] == "calibration"]
    model = artifact["models"][key]
    # Vocabulary and scaling must see training rows only, never held-out text.
    expected = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), lowercase=False, max_features=12000
    )
    expected.fit([records[i]["appstream"]["name"] for i in train])
    assert model[0].transformer_list[0][1][-1].vocabulary_ == expected.vocabulary_
    assert model[0].transformer_list[1][1][-1].n_samples_seen_ == len(train)
    # Calibration is reproducible from only the separate calibration partition.
    calibration_model = LogisticRegression(C=1e6, max_iter=2000, random_state=42).fit(
        model.decision_function([prepared[i]["name"] for i in calibration])[:, None],
        [int(records[i]["labels"][key]) for i in calibration],
    )
    np.testing.assert_allclose(
        calibration_model.coef_, artifact["calibrators"][key].coef_
    )
    prediction = review_listing(metadata(), output / "model.joblib", dataset / "0.png")
    for criterion in LEARNED_CRITERIA:
        assert prediction["criteria"][criterion]["calibrated"]
        assert 0 <= prediction["criteria"][criterion]["pass_score"] <= 1
        assert "brier_score" in report["criteria"][criterion]["calibrated"]
    # One-class calibration cannot justify a calibrated score.
    for i in calibration:
        path = sorted((dataset / "records").glob("*.json"))[i]
        record = json.loads(path.read_text())
        for guideline in record["review"]["guidelines"]:
            if guideline["guideline_id"] == key:
                guideline["passed"] = True
        write_json(path, record)
    fallback = train_listing(dataset, tmp_path / "fallback")
    assert "calibration_unavailable" in fallback["criteria"][key]
    fallback_result = review_listing(metadata(), tmp_path / "fallback/model.joblib")
    assert not fallback_result["criteria"][key]["calibrated"]
    artifact["version"] = -1
    joblib.dump(artifact, output / "invalid.joblib")
    with pytest.raises(ValueError, match="Unsupported listing model"):
        review_listing(metadata(), output / "invalid.joblib")


def test_unreviewed_and_missing_branding_are_not_negative_labels(dataset, tmp_path):
    for path in (dataset / "records").glob("*.json"):
        record = json.loads(path.read_text())
        record["appstream"]["branding"] = None
        for guideline in record["review"]["guidelines"]:
            if guideline["guideline_id"] == "app-name-just-the-name":
                guideline["passed"] = None
        write_json(path, record)
    audit = audit_listing(dataset)
    assert audit["criteria"]["app-name-just-the-name"]["usable_labeled_inputs"] == 0
    assert (
        audit["criteria"]["branding-good-primary-brand-colors"]["usable_labeled_inputs"]
        == 0
    )
    report = train_listing(dataset, tmp_path / "model")
    assert "skipped" in report["criteria"]["app-name-just-the-name"]
    assert "skipped" in report["criteria"]["branding-good-primary-brand-colors"]


def test_cli_json_and_input_errors(tmp_path, monkeypatch, capsys):
    path = tmp_path / "metadata.json"
    write_json(path, metadata())
    monkeypatch.setattr("sys.argv", ["icon-review", "review", str(path)])
    main()
    assert "criteria" in json.loads(capsys.readouterr().out)
    monkeypatch.setattr(
        "sys.argv",
        [
            "icon-review",
            "review",
            str(path),
            "--model",
            str(tmp_path / "missing.joblib"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "No such file" in capsys.readouterr().err


def test_combined_review_preserves_icon_scores(monkeypatch, tmp_path):
    icon_path = tmp_path / "icon.png"
    Image.new("RGBA", (32, 32), (255, 0, 0, 255)).save(icon_path)
    calls = []

    def icon_predict(model, icon):
        calls.append((model, icon))
        return {
            "criteria": {
                "app-icon-footprint": {
                    "pass_score": 0.7,
                    "calibrated": False,
                }
            }
        }

    monkeypatch.setattr("flathub_icon_review.listing.predict", icon_predict)
    model_path = tmp_path / "icon.joblib"
    result = review_listing(metadata(), icon=icon_path, icon_model=model_path)
    assert calls == [(model_path, icon_path)]
    assert result["criteria"]["app-icon-footprint"] == {
        "method": "model",
        "pass_score": 0.7,
        "calibrated": False,
    }
    with pytest.raises(ValueError, match="requires --icon"):
        review_listing(metadata(), icon_model=model_path)
