"""Grouped holdout evaluation; failures are the positive class in metrics."""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import atomic_dump, load_model
from .dataset import CRITERIA, load_records, usable_label, write_json
from .features import FEATURE_VERSION, Extractor, views


def image_groups(root: Path, records: list[dict]) -> list[str]:
    """Connect exact and near duplicates before splitting (including resized icons)."""
    hashes = []
    for record in records:
        image = np.asarray(
            views(root / record["icon_path"])[0].convert("L").resize((9, 8))
        )
        bits = (image[:, 1:] > image[:, :-1]).flatten()
        hashes.append(int.from_bytes(np.packbits(bits).tobytes(), "big"))
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, record in enumerate(records):
        for j in range(i):
            same = (
                record["app_id"] == records[j]["app_id"]
                or record["pixel_sha256"] == records[j]["pixel_sha256"]
            )
            if same or (hashes[i] ^ hashes[j]).bit_count() <= 4:
                parent[find(i)] = find(j)
    return [records[find(i)]["app_id"] for i in range(len(records))]


def failure_metrics(y_pass, pass_scores) -> dict:
    y = 1 - np.asarray(y_pass)
    scores = 1 - np.asarray(pass_scores)
    # Match moderation export: a pass score equal to the threshold is a pass.
    tn, fp, fn, tp = confusion_matrix(
        y, np.asarray(pass_scores) < 0.5, labels=[0, 1]
    ).ravel()
    return {
        "brier_score": float(brier_score_loss(y_pass, pass_scores)),
        "log_loss": float(log_loss(y_pass, pass_scores, labels=[0, 1])),
        "failures": int(y.sum()),
        "passes": int((1 - y).sum()),
        "failure_precision": float(tp / (tp + fp)) if tp + fp else None,
        "failure_recall": float(tp / (tp + fn)) if tp + fn else None,
        "false_alarm_rate": float(fp / (fp + tn)) if fp + tn else None,
        "failure_average_precision": float(average_precision_score(y, scores))
        if len(np.unique(y)) == 2
        else None,
        "confusion": {
            "true_fail": int(tp),
            "false_fail": int(fp),
            "missed_fail": int(fn),
            "true_pass": int(tn),
        },
    }


def train(root: Path, output: Path, kind: str, seed: int):
    records = [
        r
        for r in load_records(root)
        if any(usable_label(r, key) is not None for key in CRITERIA)
    ]
    if len(records) < 20:
        raise ValueError("Collect at least 20 labeled apps before training")
    groups = image_groups(root, records)
    if len(set(groups)) < 5:
        raise ValueError("Need at least five distinct image groups")
    indices = np.arange(len(records))
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            indices, groups=groups
        )
    )
    fit_idx, calibration_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            train_idx, groups=np.asarray(groups)[train_idx]
        )
    )
    calibration_idx = train_idx[calibration_idx]
    train_idx = train_idx[fit_idx]
    extractor = Extractor(kind)
    x = np.stack([extractor.cached(root, r) for r in records])
    models, calibrators, reports = {}, {}, {}
    for criterion in CRITERIA:
        labels = [usable_label(r, criterion) for r in records]
        tr = np.array([i for i in train_idx if labels[i] is not None], dtype=int)
        te = np.array([i for i in test_idx if labels[i] is not None], dtype=int)
        ca = np.array([i for i in calibration_idx if labels[i] is not None], dtype=int)
        y_calibration = np.asarray([int(labels[i]) for i in ca], dtype=int)
        y_train = np.asarray([int(labels[i]) for i in tr], dtype=int)
        y_test = np.asarray([int(labels[i]) for i in te], dtype=int)
        class_counts = np.bincount(y_train, minlength=2)
        if min(class_counts) < 5:
            reports[criterion] = {
                "skipped": "Need at least five train examples per class",
                "train_fail": int(class_counts[0]),
                "train_pass": int(class_counts[1]),
            }
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=2000, random_state=seed
            ),
        )
        model.fit(x[tr], y_train)
        models[criterion] = model
        reports[criterion] = {"train": len(tr), "calibration": len(ca), "test": len(te)}
        if min(np.bincount(y_calibration, minlength=2)) >= 5:
            calibrator = LogisticRegression(C=1e6, max_iter=2000, random_state=seed)
            calibrator.fit(model.decision_function(x[ca])[:, None], y_calibration)
            calibrators[criterion] = calibrator
            reports[criterion]["calibration_method"] = "sigmoid"
        else:
            reports[criterion]["calibration_unavailable"] = (
                "Need at least five calibration examples per class"
            )
        if len(te):
            reports[criterion]["model"] = failure_metrics(
                y_test, model.predict_proba(x[te])[:, 1]
            )
            reports[criterion]["always_pass"] = failure_metrics(
                y_test, np.ones(len(te))
            )
            if criterion in calibrators:
                reports[criterion]["calibrated"] = failure_metrics(
                    y_test,
                    calibrators[criterion].predict_proba(
                        model.decision_function(x[te])[:, None]
                    )[:, 1],
                )
        else:
            reports[criterion]["evaluation_unavailable"] = "No labeled test examples"
    if not models:
        raise ValueError(
            "No icon classifiers could be trained: need at least five training "
            "examples per class for at least one criterion. "
            "Existing output was preserved."
        )
    report = {
        "seed": seed,
        "features": extractor.spec,
        "apps": len(records),
        "image_groups": len(set(groups)),
        "train_apps": len(train_idx),
        "calibration_apps": len(calibration_idx),
        "test_apps": len(test_idx),
        "criteria": reports,
        "limitations": [
            "Exploratory current-icon/current-label pairs; freshness unknown",
            "Sigmoid calibration uses separate groups; threshold fixed at 0.5",
            "Test partition was inspected in prior experiments; exploratory evaluation",
            "Perceptual grouping is heuristic; brand-family leakage can remain",
            "Small holdout estimates are unstable; no production accuracy claim",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation.json", report)
    train_members = set(train_idx)
    calibration_members = set(calibration_idx)
    write_json(
        output / "split.json",
        {
            "seed": seed,
            "apps": [
                {
                    "app_id": r["app_id"],
                    "icon_sha256": r["icon_sha256"],
                    "labels": r["labels"],
                    "freshness": r["freshness"],
                    "group": groups[i],
                    "split": "train"
                    if i in train_members
                    else ("calibration" if i in calibration_members else "test"),
                }
                for i, r in enumerate(records)
            ],
        },
    )
    atomic_dump(
        {"features": extractor.spec, "models": models, "calibrators": calibrators},
        output / "model.joblib",
    )
    return report


def predict(model_path: Path, image_path: Path):
    # Only load model artifacts you trust: joblib is pickle-based.
    artifact = load_model(model_path, "icon")
    spec = artifact["features"]
    if spec.get("version") != FEATURE_VERSION:
        raise ValueError("Model uses an incompatible feature version")
    extractor = Extractor(spec["kind"], revision=spec.get("revision"))
    x = extractor.extract(image_path)[None, :]
    criteria = {}
    calibrators = artifact.get("calibrators", {})
    for key in CRITERIA:
        model = artifact["models"].get(key)
        if model is None:
            criteria[key] = {"unavailable": "Insufficient training labels"}
            continue
        calibrator = calibrators.get(key)
        score = (
            calibrator.predict_proba(model.decision_function(x)[:, None])[0, 1]
            if calibrator is not None
            else model.predict_proba(x)[0, 1]
        )
        criteria[key] = {
            "pass_score": float(score),
            "calibrated": calibrator is not None,
        }
    return {
        "scores_are_calibrated": bool(artifact["models"])
        and all(key in calibrators for key in artifact["models"]),
        "criteria": criteria,
    }
