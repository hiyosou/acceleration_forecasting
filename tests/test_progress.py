import pandas as pd

from acceleration_forecasting.cli import build_parser
from acceleration_forecasting.common.progress import progress_bar, progress_message
from acceleration_forecasting.evaluation.bootstrap import dataset_bootstrap


def test_progress_uses_stderr_and_can_be_disabled(capsys):
    list(progress_bar(range(2), enabled=True, desc="progress-test"))
    progress_message("done", enabled=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "progress-test" in captured.err
    assert "done" in captured.err

    list(progress_bar(range(2), enabled=False, desc="hidden"))
    progress_message("hidden", enabled=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_progress_does_not_change_bootstrap_results():
    frame = pd.DataFrame({
        "dataset_id": ["a", "a", "b"], "MAE": [1.0, 2.0, 3.0],
        "RMSE": [1.0, 2.0, 3.0], "coverage_p10_p90": [1, 0, 1],
        "mean_interval_width": [1, 1, 1], "peak_value_error": [1, 2, 3],
    })
    enabled = dataset_bootstrap(frame, iterations=10, seed=42, progress=True)
    disabled = dataset_bootstrap(frame, iterations=10, seed=42, progress=False)
    pd.testing.assert_frame_equal(enabled, disabled)


def test_all_commands_accept_no_progress():
    parser = build_parser()
    commands = [
        ["prepare-generation", "--no-progress"],
        ["train", "--model", "mlp", "--no-progress"],
        ["select-model", "--no-progress"],
        ["predict", "--no-progress"],
        ["evaluate", "--no-progress"],
    ]
    for arguments in commands:
        assert parser.parse_args(arguments).no_progress is True
