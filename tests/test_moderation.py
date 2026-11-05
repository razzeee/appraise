import json

import pytest

from flathub_icon_review.cli import main
from flathub_icon_review.model import failure_metrics
from flathub_icon_review.moderation import moderation_payloads


def test_payloads_only_include_available_writable_targets():
    review = {
        "criteria": {
            "app-icon-contrast": {"pass_score": 0.8, "calibrated": True},
            "app-name-just-the-name": {"pass_score": 0.4},
            "app-summary-not-technical": {"unavailable": "Missing model"},
            "app-name-not-too-long": {"passed": True, "pass_score": 1},
            "branding-has-primary-brand-colors": {"passed": True},
            "unknown-criterion": {"pass_score": 1},
        }
    }
    assert moderation_payloads(review) == [
        {"guideline_id": "app-icon-contrast", "passed": True},
        {"guideline_id": "app-name-just-the-name", "passed": False},
    ]
    assert moderation_payloads(review, 0.9)[0]["passed"] is False
    assert moderation_payloads(review, 0.8)[0]["passed"] is True
    assert moderation_payloads({"criteria": {}}) == []


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_thresholds(value):
    with pytest.raises(ValueError, match="threshold"):
        moderation_payloads({"criteria": {}}, value)


@pytest.mark.parametrize("value", [-1, 2, float("nan"), None, True, "0.7"])
def test_invalid_scores(value):
    with pytest.raises(ValueError, match="Invalid pass score"):
        moderation_payloads({"criteria": {"app-icon-contrast": {"pass_score": value}}})


def test_cli_prints_endpoint_bodies(monkeypatch, capsys):
    monkeypatch.setattr(
        "flathub_icon_review.flathub.review_url",
        lambda *args: {
            "source": {"app_id": "org.example.App"},
            "criteria": {"app-icon-contrast": {"pass_score": 0.7}},
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "icon-review",
            "review-url",
            "https://flathub.org/apps/org.example.App",
            "--format",
            "moderation-json",
            "--pass-threshold",
            "0.8",
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out) == [
        {"guideline_id": "app-icon-contrast", "passed": False},
    ]


def test_local_rules_only_output_is_empty(tmp_path, monkeypatch, capsys):
    path = tmp_path / "app.json"
    path.write_text(json.dumps({"name": "Example", "summary": "Read books"}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "icon-review",
            "review",
            str(path),
            "--format",
            "moderation-json",
        ],
    )
    main()
    assert json.loads(capsys.readouterr().out) == []


def test_evaluation_matches_export_at_threshold():
    scores = [0.499999, 0.5, 0.500001]
    passed = [
        moderation_payloads({"criteria": {"app-icon-contrast": {"pass_score": score}}})[
            0
        ]["passed"]
        for score in scores
    ]
    assert passed == [False, True, True]
    assert failure_metrics(passed, scores)["confusion"] == {
        "true_fail": 1,
        "false_fail": 0,
        "missed_fail": 0,
        "true_pass": 2,
    }
