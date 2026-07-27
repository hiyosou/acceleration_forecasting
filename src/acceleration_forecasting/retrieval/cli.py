from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_TREND_DIR,
    DEFAULT_WAVEFORM_DIR,
)


def _print_result(result):
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(
        description="100m振動波形のAutoencoder・ベクトルDB・類似検索"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="波形抽出と推移結合を行います。")
    prepare.add_argument("--waveform-dir", default=str(DEFAULT_WAVEFORM_DIR))
    prepare.add_argument("--trend-dir", default=str(DEFAULT_TREND_DIR))
    prepare.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))

    resplit = subparsers.add_parser(
        "resplit", help="既存成果物をdataset_id単位で再分割します。"
    )
    resplit.add_argument(
        "--source-artifact-dir", default=str(DEFAULT_ARTIFACT_DIR)
    )
    resplit.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_DIR.parent / "artifacts_dataset_split"),
    )
    resplit.add_argument("--development-ratio", type=float, default=0.8)
    resplit.add_argument("--train-ratio", type=float, default=0.9)
    resplit.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser("train", help="Autoencoderを学習します。")
    train.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    train.add_argument("--device", default=None)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--patience", type=int, default=10)

    build = subparsers.add_parser("build-db", help="80%%側をSQLiteへ登録します。")
    build.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    build.add_argument("--device", default=None)
    build.add_argument("--batch-size", type=int, default=512)

    search = subparsers.add_parser("search", help="manifestレコードから上位3日を検索します。")
    search.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    search.add_argument("--record-id", required=True)
    search.add_argument("--device", default=None)
    search.add_argument("--top-k", type=int, default=3)
    search.add_argument("--output", default="")
    search.add_argument(
        "--no-strict-time",
        action="store_true",
        help="推論日以降に完成するガイド候補も許可します。",
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="隔離したinference正解で予測結果を評価します。"
    )
    evaluate.add_argument("--artifact-dir", required=True)
    evaluate.add_argument("--prediction-file", required=True)
    evaluate.add_argument("--output", default="")
    verify = subparsers.add_parser(
        "verify-migration", help="Verify existing retrieval artifacts without modifying them."
    )
    verify.add_argument(
        "--artifact-dir",
        default=str(Path(__file__).resolve().parents[3].parent / "acceleration_retrieval" / "artifacts_dataset_split"),
    )
    verify.add_argument("--output", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_dataset

        result = prepare_dataset(
            args.waveform_dir,
            args.trend_dir,
            args.artifact_dir,
        )
    elif args.command == "resplit":
        from .splitting import create_dataset_split

        result = create_dataset_split(
            args.source_artifact_dir,
            args.artifact_dir,
            development_ratio=args.development_ratio,
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
    elif args.command == "train":
        from .training import train_autoencoder

        result = train_autoencoder(
            args.artifact_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
    elif args.command == "build-db":
        from .pipeline import build_database

        result = build_database(
            args.artifact_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.command == "search":
        from .pipeline import save_search_result, search_manifest_record

        result = search_manifest_record(
            args.artifact_dir,
            args.record_id,
            device=args.device,
            top_k=args.top_k,
            strict_time=not args.no_strict_time,
        )
        if args.output:
            save_search_result(result, args.output)
    elif args.command == "evaluate":
        from .evaluation import evaluate_predictions

        result = evaluate_predictions(
            args.artifact_dir,
            args.prediction_file,
            output_file=args.output or None,
        )
    else:
        from .migration import verify_migration

        result = verify_migration(args.artifact_dir, args.output or None)
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
