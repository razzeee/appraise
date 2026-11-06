import json

import joblib
import numpy as np
import pytest
from PIL import Image

from flathub_icon_review.cli import main
from flathub_icon_review.dataset import CRITERIA, write_json
from flathub_icon_review.features import Extractor
from flathub_icon_review.listing import train_listing
from flathub_icon_review.listing_features import LISTING_CRITERIA
from flathub_icon_review.model import predict, train


@pytest.mark.parametrize("kind", ["icon", "listing"])
@pytest.mark.parametrize("existing", [False, True])
def test_no_classifiers_preserves_output(tmp_path, kind, existing):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(25)
    for i in range(30):
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(
            data / f"{i}.png"
        )
        write_json(
            data / "records" / f"{i}.json",
            {
                "app_id": f"org.example.App{i}",
                "icon_path": f"{i}.png",
                "icon_sha256": str(i),
                "pixel_sha256": str(i),
                "labels": dict.fromkeys(CRITERIA, True),
                "freshness": dict.fromkeys(CRITERIA, "unknown"),
                "appstream": {"name": f"Name {i}", "summary": f"Read story {i}"},
                "review": {
                    "guidelines": [
                        {"guideline_id": key, "passed": True}
                        for key in LISTING_CRITERIA
                    ]
                },
            },
        )
    output = tmp_path / "model"
    original = {
        name: f"original {name}".encode()
        for name in ("model.joblib", "split.json", "evaluation.json")
    }
    if existing:
        output.mkdir()
        for name, content in original.items():
            (output / name).write_bytes(content)
    with pytest.raises(ValueError, match=f"No {kind} classifiers could be trained"):
        if kind == "icon":
            train(data, output, "handcrafted", 42)
        else:
            train_listing(data, output)
    if existing:
        assert {p.name: p.read_bytes() for p in output.iterdir()} == original
    else:
        assert not output.exists()


def test_corrupt_feature_cache_is_regenerated(tmp_path, monkeypatch):
    extractor = Extractor()
    expected = np.arange(8, dtype=np.float32)
    calls = []

    def extract(path):
        calls.append(path)
        return expected

    monkeypatch.setattr(extractor, "extract", extract)
    record = {"icon_path": "icon.png", "icon_sha256": "test"}
    extractor.cached(tmp_path, record)
    cache = next((tmp_path / "features").glob("*/*.npy"))
    cache.write_bytes(b"\x93NUMPY")
    np.testing.assert_array_equal(extractor.cached(tmp_path, record), expected)
    np.testing.assert_array_equal(np.load(cache, allow_pickle=False), expected)
    np.testing.assert_array_equal(extractor.cached(tmp_path, record), expected)
    assert len(calls) == 2  # Initial extraction and repair; subsequent read is cached.


def test_interrupted_cache_write_is_not_published(tmp_path, monkeypatch):
    extractor = Extractor()
    expected = np.arange(8, dtype=np.float32)
    monkeypatch.setattr(extractor, "extract", lambda _: expected)
    record = {"icon_path": "icon.png", "icon_sha256": "test"}
    save = np.save

    def interrupted(file, values):
        file.write(b"\x93NUMPY")
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "save", interrupted)
    with pytest.raises(KeyboardInterrupt):
        extractor.cached(tmp_path, record)
    assert not list((tmp_path / "features").glob("*/*"))
    monkeypatch.setattr(np, "save", save)
    np.testing.assert_array_equal(extractor.cached(tmp_path, record), expected)


@pytest.mark.parametrize(
    "artifact",
    [
        {"artifact_type": "listing", "version": 1, "models": {}, "calibrators": {}},
        [],
        {"features": None, "models": {}},
    ],
)
def test_wrong_icon_artifact_is_rejected_before_reading_image(tmp_path, artifact):
    model = tmp_path / "model.joblib"
    joblib.dump(artifact, model)
    with pytest.raises(ValueError, match="Expected an icon model"):
        predict(model, tmp_path / "missing-image.png")


@pytest.mark.parametrize("command", ["predict", "review"])
def test_cli_reports_wrong_icon_model_without_traceback(
    tmp_path, monkeypatch, capsys, command
):
    model = tmp_path / "listing.joblib"
    joblib.dump({"artifact_type": "listing", "models": {}}, model)
    if command == "predict":
        arguments = ["predict", str(tmp_path / "icon.png"), "--model", str(model)]
    else:
        metadata = tmp_path / "metadata.json"
        metadata.write_text(json.dumps({"name": "Example", "summary": "Read books"}))
        arguments = [
            "review",
            str(metadata),
            "--icon",
            str(tmp_path / "icon.png"),
            "--icon-model",
            str(model),
        ]
    monkeypatch.setattr("sys.argv", ["appraise", *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert "Expected an icon model" in output.err
    assert "Traceback" not in output.err
    assert not output.out
