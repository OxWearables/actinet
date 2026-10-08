import sys
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from actinet import accPlot
from actinet.utils import plot_utils


@pytest.fixture
def comparison_results():
    rows = []
    for participant, age, sex in [("P1", "18-40", "F"), ("P2", "41-60", "M")]:
        for model, offset in [("accelerometer", 1.0), ("actinet", 0.25)]:
            truth = np.array(["sleep", "light", "sleep", "light"])
            predicted = truth.copy()
            rows.append(
                {
                    "Participant": participant,
                    "Age Band": age,
                    "Sex": sex,
                    "Model": model,
                    "Accuracy": 0.8 + 0.1 * (model == "actinet"),
                    "Macro F1": 0.75 + 0.1 * (model == "actinet"),
                    "Cohen Kappa": 0.7 + 0.1 * (model == "actinet"),
                    "True": truth,
                    "Predicted": predicted,
                    "True_dict": {"sleep": 4.0, "light": 2.0},
                    "Pred_dict": {"sleep": 4.0 + offset, "light": 2.0 - offset / 2},
                }
            )
    frame = pd.DataFrame(rows)
    frame["Age Band"] = pd.Categorical(frame["Age Band"], categories=["18-40", "41-60"])
    frame["Sex"] = pd.Categorical(frame["Sex"], categories=["F", "M"])
    return frame


def test_plot_time_series_handles_timezone_imputation_and_irregularity():
    index = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 01:30", "2024-01-02 00:00"],
        tz="Europe/London",
        name="time",
    )
    frame = pd.DataFrame(
        {
            "acc": [10, 20, 30, 40],
            "sleep": [1, 0, 0, 1],
            "light": [0, 1, 1, 0],
            "imputed": [0, 1, 0, 0],
        },
        index=index,
    )
    figure = accPlot.plotTimeSeries(frame, title="Activity", showFirstNDays=1)
    assert figure._suptitle.get_text() == "Activity"
    assert len(figure.axes) == 2
    plt.close(figure)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        accPlot.plotTimeSeries(frame.reset_index())
    assert accPlot.str2bool("YES") is True
    assert accPlot.str2bool("false") is False


def test_accplot_cli_writes_default_and_explicit_files(tmp_path, monkeypatch):
    index = pd.date_range("2024-01-01", periods=4, freq="30min", name="time")
    frame = pd.DataFrame({"acc": [1, 2, 3, 4], "sleep": [1, 1, 0, 0]}, index=index)
    source = tmp_path / "series.csv.gz"
    frame.to_csv(source)
    monkeypatch.setattr(
        sys, "argv", ["accPlot", str(source), "--showFileName", "true", "--showFirstNDays", "1"]
    )
    accPlot.main()
    assert (tmp_path / "series-plot.png").exists()


def test_basic_performance_boxplots_and_save(tmp_path, comparison_results, monkeypatch):
    monkeypatch.setattr(plot_utils.plt, "show", MagicMock())
    monkeypatch.setattr(plot_utils.plt, "savefig", MagicMock())
    plot_utils.plot_model_performance(comparison_results, modulus=1)
    plot_utils.plot_difference_boxplots(comparison_results)
    ax = plot_utils.plot_boxplots(
        comparison_results, "Age Band", title="By age", show_legend=True
    )
    assert ax.get_title() == "By age"
    ax = plot_utils.plot_boxplots(comparison_results, "Sex", show_legend=False)
    assert ax.get_legend() is None
    save_path = tmp_path / "plots" / "panel.pdf"
    plot_utils.plot_boxplots_panel(
        comparison_results, by=["Age Band", "Sex"], save_path=str(save_path)
    )
    assert save_path.exists()
    plt.close("all")


def test_confusion_matrix_data_and_plots(tmp_path, comparison_results, monkeypatch):
    extracted = plot_utils.build_confusion_matrix_data(
        comparison_results, age_band="18-40", sex="F"
    )
    assert extracted[-1] == 1
    assert all(len(array) == 4 for array in extracted[:-1])
    figure, ax = plt.subplots()
    plot_utils.plot_confusion_matrix(
        ["sleep", "light"], ["sleep", "sleep"], ["sleep", "light"], "Matrix", ax=ax
    )
    assert ax.get_title() == "Matrix"
    monkeypatch.setattr(plot_utils.plt, "show", MagicMock())
    output = tmp_path / "confusion.png"
    plot_utils.generate_confusion_matrices(
        comparison_results, ["sleep", "light"], save_path=str(output), fontsize=8
    )
    assert output.exists()
    plt.close("all")


def test_bland_altman_variants_and_panel(tmp_path, comparison_results, monkeypatch):
    figure, ax = plt.subplots()
    plot_utils.bland_altman_plot(
        [1, 2, 3], [1.5, 2.0, 4.0], "sleep", "Walmsley2020", show_y_label=True, ax=ax
    )
    assert "Sleep" in ax.get_title()
    assert ax.get_ylabel() == "ActiNet - Baseline"

    figure, axes = plt.subplots(1, 2)
    for comparison in [False, "bbaa", "actinet"]:
        plot_utils.generate_bland_altman_plots(
            comparison_results,
            ["sleep", "light"],
            "Walmsley2020",
            compare_to_true=comparison,
            axs=axes,
            fontsize=8,
        )
    with pytest.raises(ValueError, match="compare_to_true"):
        plot_utils._plot_ba(
            "sleep", [1], [1], [1], [1], "Walmsley2020", "invalid", axes[0], 0, 8
        )
    with pytest.raises(ValueError, match="external axes"):
        plot_utils.generate_bland_altman_plots(
            comparison_results,
            ["sleep", "light"],
            "Walmsley2020",
            group_by="Sex",
            axs=axes,
        )
    monkeypatch.setattr(plot_utils.plt, "savefig", MagicMock())
    monkeypatch.setattr(plot_utils.plt, "show", MagicMock())
    plot_utils.generate_bland_altman_panel(
        comparison_results, ["sleep", "light"], "Walmsley2020", str(tmp_path / "panel.png"), 8
    )
    plt.close("all")


def test_error_plots_for_population_and_groups(tmp_path, comparison_results, monkeypatch):
    monkeypatch.setattr(plot_utils.plt, "show", MagicMock())
    plot_utils.plot_errors(
        comparison_results,
        ["sleep", "light"],
        "Walmsley2020",
        save_path=str(tmp_path / "errors.png"),
        fontsize=8,
    )
    assert (tmp_path / "errors.png").exists()
    plot_utils.plot_errors(
        comparison_results,
        ["sleep"],
        "Walmsley2020",
        group_by="Sex",
        fontsize=8,
    )
    plt.close("all")
