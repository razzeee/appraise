"""Grouped holdout evaluation; failures are the positive class in metrics."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    tn, fp, fn, tp = confusion_matrix(y, scores >= 0.5, labels=[0, 1]).ravel()
    return {
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
    extractor = Extractor(kind)
    x = np.stack([extractor.cached(root, r) for r in records])
    models, reports = {}, {}
    for criterion in CRITERIA:
        labels = [usable_label(r, criterion) for r in records]
        tr = np.array([i for i in train_idx if labels[i] is not None], dtype=int)
        te = np.array([i for i in test_idx if labels[i] is not None], dtype=int)
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
        reports[criterion] = {"train": len(tr), "test": len(te)}
        if len(te):
            reports[criterion]["model"] = failure_metrics(
                y_test, model.predict_proba(x[te])[:, 1]
            )
            reports[criterion]["always_pass"] = failure_metrics(
                y_test, np.ones(len(te))
            )
        else:
            reports[criterion]["evaluation_unavailable"] = "No labeled test examples"
    report = {
        "seed": seed,
        "features": extractor.spec,
        "apps": len(records),
        "image_groups": len(set(groups)),
        "train_apps": len(train_idx),
        "test_apps": len(test_idx),
        "criteria": reports,
        "limitations": [
            "Exploratory current-icon/current-label pairs; freshness unknown",
            "Scores are uncalibrated; threshold fixed at 0.5",
            "Perceptual grouping is heuristic; brand-family leakage can remain",
            "Small holdout estimates are unstable; no production accuracy claim",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation.json", report)
    train_members = set(train_idx)
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
                    "split": "train" if i in train_members else "test",
                }
                for i, r in enumerate(records)
            ],
        },
    )
    joblib.dump({"features": extractor.spec, "models": models}, output / "model.joblib")
    return report


def predict(model_path: Path, image_path: Path):
    # Only load model artifacts you trust: joblib is pickle-based.
    artifact = joblib.load(model_path)
    spec = artifact["features"]
    if spec["version"] != FEATURE_VERSION:
        raise ValueError("Model uses an incompatible feature version")
    extractor = Extractor(spec["kind"], revision=spec.get("revision"))
    x = extractor.extract(image_path)[None, :]
    return {
        "scores_are_calibrated": False,
        "criteria": {
            key: {"pass_score": float(model.predict_proba(x)[0, 1])}
            if (model := artifact["models"].get(key)) is not None
            else {"unavailable": "Insufficient training labels"}
            for key in CRITERIA
        },
    }
