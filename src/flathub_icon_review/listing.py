"""Train and review app listings using local metadata and preserved review snapshots."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .artifacts import atomic_dump, load_model
from .dataset import (
    freshness_from_review,
    labels_from_review,
    load_records,
    usable_label,
    write_json,
)
from .listing_features import (
    LEARNED_CRITERIA,
    LISTING_CRITERIA,
    RULE_CRITERIA,
    VERSION,
    category,
    hints,
    inputs,
    normalized,
    rules,
    select_numeric,
    select_text,
)
from .model import failure_metrics, image_groups, predict


def listing_records(root):
    """Derive new targets without mutating or recollecting saved icon snapshots."""
    records = load_records(root)
    for record in records:
        review = record["review"]
        record["labels"] = labels_from_review(review, LISTING_CRITERIA)
        # Summary depends on name too; branding quality depends on the icon too.
        changed = (review.get("metadata_changed_at") or {}).copy()
        for target, dependency in (
            ("app-summary", "app-name"),
            ("branding", "app-icon"),
        ):
            timestamps = [changed[k] for k in (target, dependency) if changed.get(k)]
            if timestamps:

                def timestamp(value):
                    parsed = datetime.fromisoformat(value)
                    return (
                        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
                    )

                changed[target] = max(timestamps, key=timestamp)
        record["freshness"] = freshness_from_review(
            {**review, "metadata_changed_at": changed}, LISTING_CRITERIA
        )
    return records


def listing_groups(root, records):
    """Connect image families, identical normalized names, and identical summaries."""
    image = image_groups(root, records)
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, record in enumerate(records):
        keys = [("image", image[i])]
        for field in ("name", "summary"):
            text = record["appstream"].get(field, "")
            if isinstance(text, str) and normalized(text):
                keys.append((field, normalized(text)))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    return [records[find(i)]["app_id"] for i in range(len(records))]


def prepare(root, records):
    prepared, unavailable = [], []
    for record in records:
        icon = root / record["icon_path"] if record.get("icon_path") else None
        features, missing = inputs(record["appstream"], icon)
        prepared.append(features)
        unavailable.append(missing)
    return prepared, unavailable


def audit_listing(root: Path):
    records = listing_records(root)
    prepared, missing = prepare(root, records)
    counts = {}
    for key in LISTING_CRITERIA:
        counts[key] = {
            "labels": {
                name: sum(r["labels"][key] is value for r in records)
                for name, value in (
                    ("pass", True),
                    ("fail", False),
                    ("unreviewed", None),
                )
            },
            "freshness": dict(Counter(r["freshness"][key] for r in records)),
            "usable_labeled_inputs": sum(
                usable_label(r, key) is not None and category(key) in row
                for r, row in zip(records, prepared, strict=True)
            )
            if key in LEARNED_CRITERIA
            else None,
        }
    conflicts = []
    for key in LEARNED_CRITERIA:
        duplicates = {}
        for record in records:
            app = record["appstream"]
            kind = category(key)
            value = (
                [app.get("branding"), record.get("pixel_sha256")]
                if kind == "branding"
                else [app.get("name"), app.get("summary")]
                if kind == "summary"
                else app.get("name")
            )
            fingerprint = hashlib.sha256(
                json.dumps(value, sort_keys=True).encode()
            ).hexdigest()
            duplicates.setdefault(fingerprint, []).append(record)
        for fingerprint, members in duplicates.items():
            if len({usable_label(r, key) for r in members} - {None}) > 1:
                conflicts.append(
                    {
                        "criterion": key,
                        "input_sha256": fingerprint,
                        "apps": [r["app_id"] for r in members],
                    }
                )
    result = {
        "apps": len(records),
        "criteria": counts,
        "unavailable_inputs": {
            kind: dict(Counter(row[kind] for row in missing if kind in row))
            for kind in ("name", "summary", "branding")
        },
        "conflicting_duplicate_labels": conflicts,
    }
    write_json(root / "listing-audit.json", result)
    return result


def classifier(kind, seed):
    numeric = make_pipeline(
        FunctionTransformer(select_numeric),
        DictVectorizer(),
        StandardScaler(with_mean=False),
    )
    features = numeric
    if kind != "branding":
        features = FeatureUnion(
            [
                (
                    "text",
                    make_pipeline(
                        FunctionTransformer(select_text),
                        TfidfVectorizer(
                            analyzer="char",
                            ngram_range=(2, 5),
                            lowercase=False,
                            max_features=12000,
                        ),
                    ),
                ),
                ("numeric", numeric),
            ]
        )
    return make_pipeline(
        features,
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        ),
    )


def train_listing(root: Path, output: Path, seed=42):
    records = listing_records(root)
    if len(records) < 20:
        raise ValueError("Collect at least 20 apps before listing training")
    prepared, missing = prepare(root, records)
    groups = np.asarray(listing_groups(root, records))
    if len(set(groups)) < 5:
        raise ValueError("Need at least five distinct listing groups")
    indices = np.arange(len(records))
    development, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            indices, groups=groups
        )
    )
    fit, calibration = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            development, groups=groups[development]
        )
    )
    partitions = {
        "train": development[fit],
        "calibration": development[calibration],
        "test": test,
    }
    models, calibrators, reports = {}, {}, {}
    for key in LEARNED_CRITERIA:
        kind = category(key)
        selected = {
            name: [
                i
                for i in values
                if kind in prepared[i] and usable_label(records[i], key) is not None
            ]
            for name, values in partitions.items()
        }
        x = {
            name: [prepared[i][kind] for i in values]
            for name, values in selected.items()
        }
        y = {
            name: np.array([int(usable_label(records[i], key)) for i in values])
            for name, values in selected.items()
        }
        report = {name: len(values) for name, values in selected.items()}
        report["class_counts"] = {
            name: {"fail": int((values == 0).sum()), "pass": int((values == 1).sum())}
            for name, values in y.items()
        }
        reports[key] = report
        if min(report["class_counts"]["train"].values()) < 5:
            report["skipped"] = "Need at least five training examples per class"
            continue
        model = classifier(kind, seed).fit(x["train"], y["train"])
        models[key] = model
        if min(report["class_counts"]["calibration"].values()) >= 5:
            calibrator = LogisticRegression(C=1e6, max_iter=2000, random_state=seed)
            calibrator.fit(
                model.decision_function(x["calibration"])[:, None], y["calibration"]
            )
            calibrators[key] = calibrator
            report["calibration_method"] = "sigmoid"
        else:
            report["calibration_unavailable"] = (
                "Need at least five calibration examples per class"
            )
        if x["test"]:
            report["model"] = failure_metrics(
                y["test"], model.predict_proba(x["test"])[:, 1]
            )
            report["always_pass"] = failure_metrics(y["test"], np.ones(len(x["test"])))
            if key in calibrators:
                report["calibrated"] = failure_metrics(
                    y["test"],
                    calibrators[key].predict_proba(
                        model.decision_function(x["test"])[:, None]
                    )[:, 1],
                )
        else:
            report["evaluation_unavailable"] = "No labeled test examples"
    if not models:
        raise ValueError(
            "No listing classifiers could be trained: need usable inputs and at least "
            "five training examples per class. Existing output was preserved."
        )
    for key in RULE_CRITERIA:
        labeled = [i for i in test if usable_label(records[i], key) is not None]
        reports[key] = {"method": "rule", "test": len(labeled)}
        if labeled:
            reports[key]["rule"] = failure_metrics(
                [int(usable_label(records[i], key)) for i in labeled],
                [int(rules(records[i]["appstream"])[key]["passed"]) for i in labeled],
            )
    report = {
        "schema_version": VERSION,
        "seed": seed,
        "apps": len(records),
        "image_and_text_groups": len(set(groups)),
        **{f"{name}_apps": len(values) for name, values in partitions.items()},
        "criteria": reports,
        "unavailable_inputs": {
            kind: sum(kind in row for row in missing)
            for kind in ("name", "summary", "branding")
        },
        "limitations": [
            "Current metadata and public labels have unknown review freshness",
            "English/default-language text only; no claim of multilingual support",
            "Heuristic image and exact-text grouping can miss related app families",
            "Listing partitions differ from the earlier icon experiments",
            "Calibration does not guarantee reliable probabilities on new data",
            "Fixed 0.5 threshold; no threshold tuning on test labels",
            "Existing collection includes only apps with downloadable icons",
        ],
    }
    membership = {int(i): name for name, values in partitions.items() for i in values}
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "split.json",
        {
            "seed": seed,
            "apps": [
                {
                    "app_id": r["app_id"],
                    "group": str(groups[i]),
                    "split": membership[i],
                    "labels": r["labels"],
                    "freshness": r["freshness"],
                    "metadata_sha256": hashlib.sha256(
                        json.dumps(
                            {
                                k: r["appstream"].get(k)
                                for k in ("name", "summary", "branding")
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest(),
                    "icon_sha256": r["icon_sha256"],
                }
                for i, r in enumerate(records)
            ],
        },
    )
    write_json(output / "evaluation.json", report)
    atomic_dump(
        {
            "artifact_type": "listing",
            "version": VERSION,
            "models": models,
            "calibrators": calibrators,
        },
        output / "model.joblib",
    )
    return report


def review_listing(
    app: dict,
    model_path: Path | None = None,
    icon: Path | None = None,
    icon_model: Path | None = None,
):
    prepared, unavailable = inputs(app, icon)
    criteria = rules(app)
    artifact = {"models": {}, "calibrators": {}}
    if model_path is not None:
        artifact = load_model(model_path, "listing")
        if (
            artifact.get("artifact_type") != "listing"
            or artifact.get("version") != VERSION
        ):
            raise ValueError("Unsupported listing model; train with train-listing")
    for key in LEARNED_CRITERIA:
        kind = category(key)
        if kind not in prepared:
            criteria[key] = {"unavailable": unavailable[kind]}
        elif key not in artifact["models"]:
            criteria[key] = {"unavailable": "No trained model for this criterion"}
        else:
            model = artifact["models"][key]
            row = [prepared[kind]]
            calibrator = artifact["calibrators"].get(key)
            score = (
                calibrator.predict_proba(model.decision_function(row)[:, None])[0, 1]
                if calibrator is not None
                else model.predict_proba(row)[0, 1]
            )
            criteria[key] = {
                "method": "model",
                "pass_score": float(score),
                "calibrated": calibrator is not None,
            }
    if icon_model is not None:
        if icon is None:
            raise ValueError("--icon-model requires --icon")
        criteria.update(
            {
                key: {"method": "model", **value} if "pass_score" in value else value
                for key, value in predict(icon_model, icon)["criteria"].items()
            }
        )
    return {
        "criteria": criteria,
        "hints": hints(app),
        "measurements": {
            kind: {key: float(value) for key, value in row["numeric"].items()}
            for kind, row in prepared.items()
        },
    }
