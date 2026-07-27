# Acceleration Forecasting

Railway vibration waveforms are retrieved by a learned 256-dimensional embedding and
their associated acceleration-max trends guide an 18-month conditional diffusion model.

The repository contains both the migrated retrieval package and the forecasting pipeline.
Existing retrieval artifacts are referenced read-only and are not regenerated.

## Commands

The migrated retrieval artifacts remain read-only in the legacy project. First verify
their hashes and database counts, then prepare forecasting arrays, train both models,
select on `model_validation`, predict `inference`, and evaluate only after prediction.

```powershell
uv run python -m acceleration_forecasting.retrieval.cli verify-migration
uv run python -m acceleration_forecasting.cli prepare-generation
uv run python -m acceleration_forecasting.cli train --model mlp --device cuda
uv run python -m acceleration_forecasting.cli train --model unet --device cuda
uv run python -m acceleration_forecasting.cli select-model --device cuda
uv run python -m acceleration_forecasting.cli predict --use-selected-model --num-samples 100 --save-samples --device cuda
uv run python -m acceleration_forecasting.cli evaluate --bootstrap 1000 --plot
```

For an end-to-end smoke test, add `--max-train 1000 --max-validation 200
--max-inference 100` to `prepare-generation`, and use the CLI epoch and record-limit
options before running the complete dataset. Run the automated tests with:

```powershell
uv run pytest -q
```

Generated datasets, checkpoints, databases, samples, and images live under
`artifacts/` and are intentionally excluded from Git. Source code, configuration,
schemas, and tests remain version controlled.
