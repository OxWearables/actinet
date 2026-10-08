import datetime

import numpy as np
import pandas as pd
import pytest

from actinet import circadian, summarisation
from actinet.utils import summary_utils, utils


def test_frequency_date_parsers_and_safe_indexer(capsys):
    index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:01",
            "2024-01-01 00:00:02",
            "2024-01-01 00:00:12",
        ]
    )
    assert utils.infer_freq(index) == pd.Timedelta(seconds=1)
    parsed = utils.date_parser("2024-01-01 12:00:00.000+0000 [Europe/London]")
    assert parsed == pd.Timestamp("2024-01-01 12:00:00", tz="Europe/London")
    assert utils.custom_date_parser("2024-01-01 12:00:00.000000+0100") == datetime.datetime(
        2024, 1, 1, 11
    )
    values = np.array([10, 20, 30])
    np.testing.assert_array_equal(utils.safe_indexer(values, [2, 0]), [30, 10])
    assert utils.safe_indexer(None, [0]) is None
    utils.to_screen("hello", verbose=True)
    assert "hello" in capsys.readouterr().out
    utils.to_screen("hidden", verbose=False)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "frame,length,columns,expected",
    [
        (pd.DataFrame({"x": [1], "y": [2], "z": [3]}), 1, ["x", "y", "z"], True),
        (pd.DataFrame({"x": [1], "y": [2]}), 1, ["x", "y", "z"], False),
        (pd.DataFrame({"x": [1, 2], "y": [2, 3], "z": [3, 4]}), 1, ["x", "y", "z"], False),
        (pd.DataFrame({"x": [np.nan], "y": [2], "z": [3]}), 1, ["x", "y", "z"], False),
    ],
)
def test_is_good_window(frame, length, columns, expected):
    assert utils.is_good_window(frame, length, columns) is expected


def test_resize_interpolates_each_axis():
    values = np.array([[[0.0], [10.0]]])
    resized = utils.resize(values, 3)
    np.testing.assert_allclose(resized[:, :, 0], [[0.0, 5.0, 10.0]])


def test_day_filters_do_not_mutate_input(hourly_acceleration, capsys):
    original = hourly_acceleration.copy()
    first_removed = utils.drop_first_last_days(hourly_acceleration, "first")
    last_removed = utils.drop_first_last_days(hourly_acceleration, "last")
    both_removed = utils.drop_first_last_days(hourly_acceleration, "both")
    assert set(first_removed.index.date) == {datetime.date(2024, 1, 2)}
    assert set(last_removed.index.date) == {datetime.date(2024, 1, 1)}
    assert both_removed.empty

    flagged = utils.flag_wear_below_days(hourly_acceleration, "24h")
    assert flagged.isna().all(axis=None)
    pd.testing.assert_frame_equal(hourly_acceleration, original)

    empty = hourly_acceleration.iloc[:0]
    assert utils.drop_first_last_days(empty).empty
    assert "No data to drop" in capsys.readouterr().out
    assert utils.flag_wear_below_days(empty).empty
    assert "No data to exclude" in capsys.readouterr().out


def test_wear_statistics_are_numerically_pinned(hourly_acceleration):
    stats = utils.calculate_wear_stats(hourly_acceleration)
    assert stats == {
        "StartTime": "2024-01-01 00:00:00",
        "EndTime": "2024-01-02 23:00:00",
        "WearStartTime": "2024-01-01 00:00:00",
        "WearEndTime": "2024-01-02 23:00:00",
        "WearTime(days)": pytest.approx(46 / 24),
        "NonwearTime(days)": pytest.approx(2 / 24),
        "Covers24hOK": 0,
    }
    daily = summary_utils.calculate_daily_wear_stats(hourly_acceleration)
    assert daily["WearTime(hours)"].tolist() == [23.0, 23.0]
    empty = hourly_acceleration.iloc[:0]
    assert utils.calculate_wear_stats(empty)["WearTime(days)"] == 0
    assert summary_utils.calculate_daily_wear_stats(empty).empty


def test_impute_missing_uses_matching_clock_times():
    index = pd.to_datetime(
        ["2024-01-01 12:00", "2024-01-08 12:00", "2024-01-15 12:00"]
    )
    data = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0], "b": [100.0, 300.0, np.nan]}, index=index
    )
    result = summary_utils.impute_missing(data, extrapolate=False)
    expected = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [100.0, 300.0, 200.0]}, index=index
    )
    pd.testing.assert_frame_equal(result, expected)


def test_impute_missing_pads_to_complete_days():
    index = pd.date_range("2024-01-01 12:00", periods=3, freq="1h", name="time")
    result = summary_utils.impute_missing(pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=index))
    assert result.index[0] == pd.Timestamp("2024-01-01")
    assert result.index[-1] == pd.Timestamp("2024-01-01 23:00")


def test_ecdf_and_daily_summaries_pin_business_values():
    index = pd.date_range("2024-01-01", periods=8, freq="6h", name="time")
    acc = pd.Series([0, 10, 20, 30, 40, 50, 60, 70], index=index, name="acc")
    summary = summary_utils.calculateECDF(acc, {})
    assert summary["acc-ecdf-1mg"] == pytest.approx(1 / 8)
    assert summary["acc-ecdf-20mg"] == pytest.approx(3 / 8)
    assert summary["acc-ecdf-100mg"] == 1

    adjusted = acc.fillna(0)
    daily_enmo = summary_utils.summarize_daily_enmo(acc, adjusted, min_wear_per_day=0)
    np.testing.assert_allclose(daily_enmo["ENMO(mg)"], [15, 55])
    labels = ["sleep", "light"]
    activities = pd.DataFrame(
        {"sleep": [1, 1, 0, 0, 1, 0, 0, 0], "light": [0, 0, 1, 1, 0, 1, 1, 1]},
        index=index,
    )
    daily = summary_utils.summarize_daily_activity(
        activities, activities.copy(), labels, min_wear_per_day=0
    )
    assert daily.iloc[0].to_dict() == {
        "Sleep(hours)": 12.0,
        "Light(hours)": 12.0,
        "Sleep Adjusted(hours)": 12.0,
        "Light Adjusted(hours)": 12.0,
    }
    shifted = adjusted.copy()
    shifted.index = pd.date_range(index[0], periods=8, freq="3h")
    with pytest.raises(ValueError, match="same frequency"):
        summary_utils.summarize_daily_enmo(acc, shifted)
    with pytest.raises(ValueError, match="same frequency"):
        summary_utils.summarize_daily_activity(activities, activities.iloc[::2], labels)


def test_activity_summary_regression(epoch_predictions):
    labels = ["sleep", "sedentary", "light", "moderate-vigorous"]
    summary, daily = summarisation.get_activity_summary(
        epoch_predictions,
        labels,
        exclude_daily_wear_below=None,
        intensityDistribution=True,
        circadianMetrics=False,
        verbose=False,
    )
    assert summary["FirstDay(0=mon,6=sun)"] == 0
    assert summary["acc-overall-avg"] == pytest.approx(235.0)
    assert summary["sleep-overall-avg"] == pytest.approx(0.25)
    assert summary["day0-recorded-sleep(hrs)"] == pytest.approx(6.0)
    assert summary["acc-ecdf-100mg"] == pytest.approx(11 / 48)
    assert daily.index.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert daily["Sleep(hours)"].tolist() == [6.0, 6.0]


def test_activity_summary_reads_csv(tmp_path, epoch_predictions):
    path = tmp_path / "epochs.csv.gz"
    epoch_predictions.to_csv(path)
    summary, daily = summarisation.get_activity_summary(
        str(path), ["sleep", "sedentary", "light", "moderate-vigorous"], verbose=False
    )
    assert summary["acc-overall-avg"] == pytest.approx(235.0)
    assert len(daily) == 2


def test_circadian_metrics_on_known_daily_wave():
    index = pd.date_range("2024-01-01", periods=24 * 12 * 3, freq="5min")
    hour = index.hour + index.minute / 60
    acc = 100 + 50 * np.cos(2 * np.pi * (hour - 14) / 24)
    frame = pd.DataFrame(
        {"acc": acc, "sleep": (hour < 8).astype(float), "light": (hour >= 8).astype(float)},
        index=index,
    )
    psd = circadian.calculatePSD(frame, 300, False, ["sleep", "light"], {})
    assert psd["PSD_sleep"] >= 0
    assert "PSD_acc" not in psd
    psd_acc = circadian.calculatePSD(frame, 300, True, ["sleep", "light"], {})
    assert psd_acc["PSD_acc"] >= 0
    centered = frame.copy()
    centered["acc"] = centered["acc"] - centered["acc"].mean()
    fourier = circadian.calculateFourierFreq(centered, 300, True, ["sleep", "light"], {})
    assert fourier["fourier-frequency_acc"] == pytest.approx(2.15825217198, abs=1e-6)
    sleep_fourier = circadian.calculateFourierFreq(
        frame, 300, False, ["sleep", "light"], {}
    )
    assert np.isfinite(sleep_fourier["fourier-frequency_sleep"])
    m10l5 = circadian.calculateM10L5(frame, 300, {})
    assert m10l5["M10L5"] > 0
