import argparse
import json
from pathlib import Path

from .dataset import audit, collect


def main():
    parser = argparse.ArgumentParser(description="Flathub icon-review experiments")
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
    args = parser.parse_args()
    if args.command == "collect":
        if args.limit < 0 or args.interval < 0:
            parser.error("limit and interval must be nonnegative")
        result = collect(args.data, args.limit, args.seed, args.interval, args.app_id)
    elif args.command == "audit":
        result = audit(args.data)
    elif args.command == "train":
        from .model import train

        result = train(args.data, args.output, args.features, args.seed)
    else:
        from .model import predict

        result = predict(args.model, args.image)
    print(json.dumps(result, indent=2))
    if args.command == "collect" and result["errors"]:
        raise SystemExit(1)
