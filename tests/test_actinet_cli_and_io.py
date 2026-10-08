import hashlib
import io
import json
import sys
import warnings
from unittest.mock import MagicMock, mock_open

import actipy
import joblib
import numpy as np
import pandas as pd
import pytest

from actinet import accPlot, actinet, summarisation
from actinet.models import ActivityClassifier
from actinet.utils import summary_utils
from actinet.utils import utils as actinet_utils


def _acceleration_frame(periods=6):
    index = pd.date_range("2024-01-01", periods=periods, freq="1s", name="time")
    return pd.DataFrame({"x": np.zeros(periods), "y": np.zeros(periods), "z": np.ones(periods)}, index=index)


def test_summary_path_hash_and_json_encoder(tmp_path, capsys):
    path = tmp_path / "summary.json"
    info = {
        "Filename": "sample.cwa",
        "Filesize(MB)": np.float32(1.5),
        "WearTime(days)": np.float64(1),
        "NonwearTime(days)": np.int64(0),
        "ReadOK": np.bool_(True),
        "array": np.array([1, 2]),
    }
    actinet.save_and_print_summary(path, info, verbose=True)
    saved = json.loads(path.read_text())
    assert saved["array"] == [1, 2]
    assert "Summary Stats" in capsys.readouterr().out

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"actinet")
    assert actinet.md5(payload) == hashlib.md5(b"actinet").hexdigest()
    directory, filename, extension = actinet.resolve_path("folder/sample.csv.gz")
    assert str(directory) == "folder"
    assert filename == "sample"
    assert extension == ".csv"
    with pytest.raises(TypeError):
        json.dumps({"unsupported": object()}, cls=actinet.NpEncoder)


def test_validate_time_interval_edge_cases():
    assert actinet.validate_time_interval() == (None, None)
    with pytest.raises(ValueError, match="Invalid start"):
        actinet.validate_time_interval(pd.NaT)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        actinet.validate_time_interval("2024-01-01", index=pd.Index([1, 2]))
    aware = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    start, end = actinet.validate_time_interval("2024-01-01 00:00+00:00", "2024-01-01 02:00+00:00", aware)
    assert start.tzinfo is not None and end.tzinfo is not None
    with pytest.raises(ValueError, match="does not overlap"):
        actinet.validate_time_interval(end_time="2023-12-31 00:00+00:00", index=aware)


def test_read_csv_by_column_indices_and_pickle(tmp_path, monkeypatch):
    frame = _acceleration_frame()
    csv_frame = frame.reset_index().rename(columns={"time": "timestamp", "x": "axis_x"})
    csv = tmp_path / "sample.csv"
    csv_frame.to_csv(csv, index=False)
    monkeypatch.setattr(
        actipy,
        "process",
        lambda data, *args, **kwargs: (data, {"ResampleRate": 1}),
    )
    loaded, info = actinet.read(
        csv,
        csv_txyz_idxs="0,1,2,3",
        dateFormat="%Y-%m-%d %H:%M:%S",
        sample_rate=1,
        verbose=False,
    )
    assert loaded.columns.tolist() == ["x", "y", "z"]
    assert info["Device"] == ".csv"
    assert info["ReadOK"] == 1

    pickle = tmp_path / "sample.pkl"
    frame.to_pickle(pickle)
    loaded, info = actinet.read(pickle, sample_rate=1, verbose=False)
    pd.testing.assert_frame_equal(loaded, frame)
    assert info["Device"] == ".pkl"


@pytest.mark.parametrize(
    "indices,match",
    [
        ("0,1,2", "4 comma-separated"),
        ("0,-1,2,3", "non-negative"),
        ("0,1,2,10", "out of range"),
    ],
)
def test_read_rejects_invalid_csv_indices(tmp_path, indices, match):
    csv = tmp_path / "sample.csv"
    _acceleration_frame().reset_index().to_csv(csv, index=False)
    with pytest.raises(ValueError, match=match):
        actinet.read(csv, csv_txyz_idxs=indices, sample_rate=1, verbose=False)


def test_read_device_warnings_unknown_format_and_empty_selection(tmp_path, monkeypatch):
    frame = _acceleration_frame()
    cwa = tmp_path / "sample.cwa"
    cwa.write_bytes(b"fixture")
    monkeypatch.setattr(
        actipy,
        "read_device",
        lambda *args, **kwargs: (frame, {"SampleRate": 1}),
    )
    with warnings.catch_warnings(record=True) as caught:
        loaded, info = actinet.read(cwa, csv_txyz_idxs="0,1,2,3", verbose=False)
    assert len(loaded) == len(frame)
    assert info["ResampleRate"] == 1
    assert any("only supported for CSV" in str(item.message) for item in caught)

    unknown = tmp_path / "sample.xyz"
    unknown.touch()
    with pytest.raises(ValueError, match="Unknown file format"):
        actinet.read(unknown, verbose=False)

    monkeypatch.setattr(
        actinet,
        "validate_time_interval",
        MagicMock(side_effect=[(None, None), (pd.Timestamp("2024-01-03"), None)]),
    )
    with pytest.raises(ValueError, match="does not contain any data"):
        actinet.read(cwa, verbose=False)


def test_load_local_classifier_and_errors(tmp_path, monkeypatch):
    path = tmp_path / "classifier.joblib.lzma"
    classifier = ActivityClassifier(labels=["sleep"])
    joblib.dump(classifier, path)
    load_model = MagicMock()
    monkeypatch.setattr(ActivityClassifier, "load_model", load_model)
    loaded = actinet.load_classifier(str(path), model_repo_path=str(tmp_path), verbose=True)
    assert loaded.labels == ["sleep"]
    load_model.assert_called_once_with(str(tmp_path))
    with pytest.raises(ValueError, match="Unknown classifier"):
        actinet.load_classifier("not-a-classifier", verbose=False)

    broken = tmp_path / "broken.joblib.lzma"
    broken.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Error loading"):
        actinet.load_classifier(str(broken), verbose=False)


def test_download_known_classifier_and_detect_corruption(monkeypatch):
    classifier = ActivityClassifier(labels=["sleep"])
    monkeypatch.setattr(actinet, "__classifiers__", {"known": {"version": "test-model", "md5": "expected"}})
    response = MagicMock()
    response.__enter__.return_value = io.BytesIO(b"model")
    response.__exit__.return_value = False
    monkeypatch.setattr(actinet.urllib.request, "urlopen", MagicMock(return_value=response))
    monkeypatch.setattr("builtins.open", mock_open())
    monkeypatch.setattr(actinet.shutil, "copyfileobj", MagicMock())
    monkeypatch.setattr(actinet, "md5", lambda path: "expected")
    monkeypatch.setattr(joblib, "load", lambda path: classifier)
    monkeypatch.setattr(ActivityClassifier, "load_model", MagicMock())
    assert actinet.load_classifier("known", force_download=True, verbose=True) is classifier
    monkeypatch.setattr(actinet, "md5", lambda path: "wrong")
    with pytest.raises(ValueError, match="corrupted"):
        actinet.load_classifier("known", force_download=True, verbose=False)


def test_cli_successful_prediction_workflow(tmp_path, monkeypatch, capsys):
    data = _acceleration_frame(90)
    info = {
        "Filename": "sample.cwa",
        "Filesize(MB)": 1.0,
        "WearTime(days)": 1.0,
        "NonwearTime(days)": 0.0,
        "ReadOK": 1,
    }
    monkeypatch.setattr(actinet, "read", MagicMock(return_value=(data, info)))
    monkeypatch.setattr(actinet_utils, "drop_first_last_days", MagicMock(side_effect=lambda x, _: x))
    monkeypatch.setattr(actinet_utils, "flag_wear_below_days", MagicMock(side_effect=lambda x, _: x))
    monkeypatch.setattr(actinet_utils, "calculate_wear_stats", lambda data: {"WearTime(days)": 1.0})
    daily_wear = pd.DataFrame({"WearTime(hours)": [24.0]}, index=pd.DatetimeIndex(["2024-01-01"], name="Date"))
    monkeypatch.setattr(summary_utils, "calculate_daily_wear_stats", lambda data: daily_wear)
    classifier = MagicMock()
    classifier.labels = ["sleep", "light"]
    classifier.window_sec = 30
    predictions = pd.DataFrame(
        {"acc": [1.0, 2.0, 3.0], "sleep": [1, 0, 0], "light": [0, 1, 1]},
        index=pd.date_range("2024-01-01", periods=3, freq="30s", name="time"),
    )
    classifier.predict_from_frame.return_value = predictions
    monkeypatch.setattr(actinet, "load_classifier", MagicMock(return_value=classifier))
    daily_activity = pd.DataFrame({"Sleep(hours)": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"], name="Date"))
    monkeypatch.setattr(
        summarisation,
        "get_activity_summary",
        lambda *args, **kwargs: (
            {"acc-overall-avg": 2.0, "sleep-overall-avg": 1 / 3, "light-overall-avg": 2 / 3},
            daily_activity,
        ),
    )
    figure = MagicMock()
    monkeypatch.setattr(accPlot, "plotTimeSeries", MagicMock(return_value=figure))
    argv = [
        "actinet",
        "sample.cwa",
        "--outdir",
        str(tmp_path),
        "--pytorch-device",
        "cpu",
        "--exclude-first-last",
        "first",
        "--exclude-wear-below",
        "1h",
        "--require-sleep-above",
        "30min",
        "--single-sleep-block",
        "--plot-activity",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    actinet.main()
    output = tmp_path / "sample"
    assert (output / "sample-timeSeries.csv.gz").exists()
    assert (output / "sample-outputSummary.json").exists()
    assert (output / "sample-Daily.csv.gz").exists()
    figure.savefig.assert_called_once()
    classifier.predict_from_frame.assert_called_once_with(data, None, True, "30min", True)
    assert "Done! (" in capsys.readouterr().out


def test_cli_cache_only_and_missing_filepath(monkeypatch, capsys):
    loader = MagicMock()
    monkeypatch.setattr(actinet, "load_classifier", loader)
    monkeypatch.setattr(sys, "argv", ["actinet", "--cache-classifier", "--quiet"])
    assert actinet.main() is None
    loader.assert_called_once()
    assert "Done! (" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["actinet", "--quiet"])
    with pytest.raises(ValueError, match="provide a file"):
        actinet.main()
