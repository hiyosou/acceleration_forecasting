from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common.config import REPOSITORY_ROOT


DEFAULT_RETRIEVAL = REPOSITORY_ROOT.parent / "acceleration_retrieval" / "artifacts_dataset_split"
DEFAULT_ARTIFACTS = REPOSITORY_ROOT / "artifacts"


def build_parser():
    parser = argparse.ArgumentParser(description="検索拡張型・最大加速度18か月予測")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-generation", help="生成用データと固定ガイドを構築")
    prepare.add_argument("--retrieval-artifact-dir", default=str(DEFAULT_RETRIEVAL))
    prepare.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "datasets"))
    prepare.add_argument("--device", default=None)
    prepare.add_argument("--max-train", type=int)
    prepare.add_argument("--max-validation", type=int)
    prepare.add_argument("--max-inference", type=int)
    prepare.add_argument("--no-progress", action="store_true")

    residual = commands.add_parser("prepare-residual", help="Build guide-baseline residual datasets")
    residual.add_argument("--source-dataset-dir", required=True)
    residual.add_argument("--output-dir", required=True)
    residual.add_argument("--temperature", type=float, default=0.1)
    residual.add_argument("--clip-quantile-low", type=float, default=0.5)
    residual.add_argument("--clip-quantile-high", type=float, default=99.5)

    train = commands.add_parser("train", help="MLPまたはU-Net拡散モデルを学習")
    train.add_argument("--model", choices=("mlp", "unet"), required=True)
    train.add_argument("--dataset-dir", default=str(DEFAULT_ARTIFACTS / "datasets"))
    train.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACTS / "models"))
    train.add_argument("--device", default=None)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--no-resume", action="store_true")
    train.add_argument("--no-progress", action="store_true")
    train.add_argument("--dropout", type=float, default=0.1)

    select = commands.add_parser("select-model", help="validationで正式モデルを選択")
    select.add_argument("--dataset-dir", default=str(DEFAULT_ARTIFACTS / "datasets"))
    select.add_argument("--model-dir", default=str(DEFAULT_ARTIFACTS / "models"))
    select.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "model_selection"))
    select.add_argument("--device", default=None)
    select.add_argument("--num-samples", type=int, default=100)
    select.add_argument("--max-records", type=int)
    select.add_argument("--no-progress", action="store_true")
    select.add_argument("--model", choices=("mlp", "unet"), action="append")
    select.add_argument("--mae-limit", type=float)
    select.add_argument("--coverage-min", type=float)
    select.add_argument("--interval-width-limit", type=float)

    predict = commands.add_parser("predict", help="選択モデルで正式inferenceを実行")
    predict.add_argument("--dataset-dir", default=str(DEFAULT_ARTIFACTS / "datasets"))
    predict.add_argument("--selection-file", default=str(DEFAULT_ARTIFACTS / "model_selection" / "selected_model.json"))
    predict.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "predictions"))
    predict.add_argument("--device", default=None)
    predict.add_argument("--num-samples", type=int, default=100)
    predict.add_argument("--sampling-steps", type=int, default=50)
    predict.add_argument("--save-samples", action="store_true")
    predict.add_argument("--max-records", type=int)
    predict.add_argument("--use-selected-model", action="store_true", help=argparse.SUPPRESS)
    predict.add_argument("--no-progress", action="store_true")

    evaluate = commands.add_parser("evaluate", help="正解を読み評価・可視化")
    evaluate.add_argument("--dataset-dir", default=str(DEFAULT_ARTIFACTS / "datasets"))
    evaluate.add_argument("--prediction-dir", default=str(DEFAULT_ARTIFACTS / "predictions"))
    evaluate.add_argument("--output-dir", default=str(DEFAULT_ARTIFACTS / "evaluation"))
    evaluate.add_argument("--plot-output-dir")
    evaluate.add_argument(
        "--plot-style", choices=("detailed", "clean"), default="detailed",
        help="detailed: ガイド詳細を表示、clean: ガイド凡例・注記を非表示",
    )
    evaluate.add_argument(
        "--single-sample-index", type=int,
        help="保存済み生成系列の指定indexを代表例として重ね描画（0始まり）",
    )
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--plot", action="store_true")
    evaluate.add_argument("--plot-max-targets", type=int, default=100)
    evaluate.add_argument("--y-max", type=float, default=5.0)
    evaluate.add_argument("--dpi", type=int, default=150)
    evaluate.add_argument("--no-progress", action="store_true")

    compare = commands.add_parser("compare", help="Compare one-anchor and residual predictions")
    compare.add_argument("--baseline-evaluation-dir", required=True)
    compare.add_argument("--residual-evaluation-dir", required=True)
    compare.add_argument("--baseline-prediction-dir", required=True)
    compare.add_argument("--residual-prediction-dir", required=True)
    compare.add_argument("--dataset-dir", required=True)
    compare.add_argument("--output-dir", required=True)
    compare.add_argument("--max-images", type=int, default=100)
    compare.add_argument("--dpi", type=int, default=150)

    summary = commands.add_parser("summarize-residual", help="Write residual experiment ledger")
    summary.add_argument("--selection-file", required=True)
    summary.add_argument("--evaluation-dir", required=True)
    summary.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prepare-generation":
        from .datasets.build_generation_dataset import prepare_generation_dataset
        result = prepare_generation_dataset(
            args.retrieval_artifact_dir, args.output_dir, device=args.device,
            max_train=args.max_train, max_validation=args.max_validation,
            max_inference=args.max_inference,
            progress=not args.no_progress,
        )
    elif args.command == "prepare-residual":
        from .datasets.residual_dataset import prepare_residual_dataset
        result = prepare_residual_dataset(
            args.source_dataset_dir, args.output_dir,
            temperature=args.temperature,
            clip_quantile_low=args.clip_quantile_low,
            clip_quantile_high=args.clip_quantile_high,
        )
    elif args.command == "train":
        from .generation.train import train_model
        result = train_model(
            args.dataset_dir, args.artifact_dir, args.model, device=args.device,
            epochs=args.epochs, batch_size=args.batch_size, resume=not args.no_resume,
            progress=not args.no_progress,
            dropout=args.dropout,
        )
    elif args.command == "select-model":
        from .generation.select_model import select_model
        result = select_model(
            args.dataset_dir, args.model_dir, args.output_dir, device=args.device,
            num_samples=args.num_samples, max_records=args.max_records,
            progress=not args.no_progress,
            candidates=tuple(args.model or ("mlp", "unet")),
            mae_limit=args.mae_limit, coverage_min=args.coverage_min,
            interval_width_limit=args.interval_width_limit,
        )
    elif args.command == "predict":
        from .generation.predict import predict
        result = predict(
            args.dataset_dir, args.selection_file, args.output_dir, device=args.device,
            num_samples=args.num_samples, sampling_steps=args.sampling_steps,
            save_samples=args.save_samples, max_records=args.max_records,
            progress=not args.no_progress,
        )
    elif args.command == "evaluate":
        from .evaluation.evaluate import evaluate
        result = evaluate(
            args.dataset_dir, args.prediction_dir, args.output_dir,
            bootstrap_iterations=args.bootstrap, plot=args.plot,
            plot_max_targets=args.plot_max_targets, y_max=args.y_max, dpi=args.dpi,
            plot_output_dir=args.plot_output_dir,
            plot_style=args.plot_style,
            single_sample_index=args.single_sample_index,
            progress=not args.no_progress,
        )
    elif args.command == "compare":
        from .evaluation.compare_predictions import compare_predictions
        result = compare_predictions(
            args.baseline_evaluation_dir, args.residual_evaluation_dir,
            args.baseline_prediction_dir, args.residual_prediction_dir,
            args.dataset_dir, args.output_dir, max_images=args.max_images, dpi=args.dpi,
        )
    else:
        from .evaluation.experiment_summary import summarize_residual_experiment
        result = summarize_residual_experiment(
            args.selection_file, args.evaluation_dir, args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
