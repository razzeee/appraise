import argparse
import json
import math
from pathlib import Path

import httpx

from .dataset import audit, collect


def main():
    parser = argparse.ArgumentParser(
        description="Appraise: ML-assisted quality checks for Flathub app listings"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("collect", help="Collect/resume a seeded API sample")
    fetch.add_argument("--data", type=Path, default=Path("data/sample"))
    fetch.add_argument(
        "--limit", type=int, default=300, help="0 means the whole catalog"
    )
    fetch.add_argument("--seed", type=int, default=42)
    fetch.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Minimum seconds between sequential HTTP requests",
    )
    fetch.add_argument("--app-id", action="append", default=[])
    inspect = commands.add_parser("audit", help="Count labels and duplicate icons")
    inspect.add_argument("--data", type=Path, default=Path("data/sample"))
    fit = commands.add_parser("train", help="Train classifiers with grouped evaluation")
    fit.add_argument("--data", type=Path, default=Path("data/sample"))
    fit.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    fit.add_argument(
        "--features", choices=["handcrafted", "clip"], default="handcrafted"
    )
    fit.add_argument("--seed", type=int, default=42)
    infer = commands.add_parser("predict", help="Score an icon using a local model")
    infer.add_argument("image", type=Path)
    infer.add_argument(
        "--model", type=Path, default=Path("artifacts/baseline/model.joblib")
    )
    listing_audit = commands.add_parser(
        "audit-listing", help="Audit name, summary, and branding labels"
    )
    listing_audit.add_argument("--data", type=Path, default=Path("data/sample"))
    listing_fit = commands.add_parser(
        "train-listing", help="Train calibrated listing classifiers"
    )
    listing_fit.add_argument("--data", type=Path, default=Path("data/sample"))
    listing_fit.add_argument("--output", type=Path, default=Path("artifacts/listing"))
    listing_fit.add_argument("--seed", type=int, default=42)
    review = commands.add_parser(
        "review", help="Review English AppStream metadata and an optional icon"
    )
    review.add_argument(
        "metadata", type=Path, nargs="?", help="Local AppStream JSON file"
    )
    review.add_argument("--app-id", help="Review an app from the collected dataset")
    review.add_argument("--data", type=Path, default=Path("data/sample"))
    review.add_argument(
        "--model",
        type=Path,
        help="Trained listing model; omitted means rules and hints only",
    )
    review.add_argument(
        "--icon", type=Path, help="Local icon for brand-color measurements"
    )
    review.add_argument(
        "--icon-model", type=Path, help="Also run the existing icon classifier"
    )
    live = commands.add_parser(
        "review-url", help="Fetch a Flathub app page's current data and run all models"
    )
    live.add_argument("url", help="HTTPS Flathub app page URL")
    live.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/listing/model.joblib"),
        help="Listing model (default: artifacts/listing/model.joblib)",
    )
    live.add_argument(
        "--icon-model",
        type=Path,
        default=Path("artifacts/calibrated-clip/model.joblib"),
        help="Icon model (default: artifacts/calibrated-clip/model.joblib)",
    )
    for command in (review, live):
        command.add_argument(
            "--format",
            choices=("review", "moderation-json"),
            default="review",
            help="Full review or an array of quality-moderation POST bodies",
        )
        command.add_argument(
            "--pass-threshold",
            type=float,
            default=0.5,
            help="Moderation JSON: pass when score >= this value (default: 0.5)",
        )
    args = parser.parse_args()
    if args.command in ("review", "review-url") and (
        not math.isfinite(args.pass_threshold) or not 0 <= args.pass_threshold <= 1
    ):
        parser.error("--pass-threshold must be between 0 and 1")
    if args.command == "review-url":
        from .flathub import review_url

        try:
            result = review_url(args.url, args.model, args.icon_model)
        except (ValueError, OSError, httpx.HTTPError) as error:
            parser.error(str(error))
    elif args.command in ("audit-listing", "train-listing", "review"):
        from .listing import audit_listing, review_listing, train_listing

        try:
            if args.command == "audit-listing":
                result = audit_listing(args.data)
            elif args.command == "train-listing":
                result = train_listing(args.data, args.output, args.seed)
            else:
                if bool(args.metadata) == bool(args.app_id):
                    parser.error("review needs either a metadata JSON file or --app-id")
                icon = args.icon
                if args.app_id:
                    from .dataset import load_records

                    record = next(
                        (
                            r
                            for r in load_records(args.data)
                            if r["app_id"] == args.app_id
                        ),
                        None,
                    )
                    if record is None:
                        parser.error(
                            f"App {args.app_id!r} not collected; "
                            "run collect --app-id first"
                        )
                    app = record["appstream"]
                    if icon is None and record.get("icon_path"):
                        icon = args.data / record["icon_path"]
                else:
                    app = json.loads(args.metadata.read_text())
                result = review_listing(app, args.model, icon, args.icon_model)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    elif args.command == "collect":
        if args.limit < 0 or args.interval < 0:
            parser.error("limit and interval must be nonnegative")
        result = collect(args.data, args.limit, args.seed, args.interval, args.app_id)
    elif args.command == "audit":
        result = audit(args.data)
    elif args.command == "train":
        from .model import train

        try:
            result = train(args.data, args.output, args.features, args.seed)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    else:
        from .model import predict

        try:
            result = predict(args.model, args.image)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    if args.command in ("review", "review-url") and args.format == "moderation-json":
        from .moderation import moderation_payloads

        try:
            result = moderation_payloads(result, args.pass_threshold)
        except ValueError as error:
            parser.error(str(error))
    print(json.dumps(result, indent=2))
    if args.command == "collect" and result["errors"]:
        raise SystemExit(1)
