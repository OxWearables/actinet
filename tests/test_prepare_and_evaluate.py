import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from actinet import evaluate, prepare


class ImmediateParallel:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, jobs):
        results = []
        for function, args, kwargs in jobs:
            results.append(function(*args, **kwargs))
        return results


def _annotated_data(periods=8):
    index = pd.date_range("2024-01-01", periods=periods, freq="1s", name="time")
    return pd.DataFrame(
        {
            "x": np.arange(periods, dtype=float),
            "y": np.arange(periods, dtype=float) + 1,
            "z": np.ones(periods),
            "annotation": ["walk"] * (periods // 2) + ["sleep"] * (periods - periods // 2),
        },
        index=index,
    )


def test_load_data_reads_csv_and_parquet(tmp_path, monkeypatch):
    source = _annotated_data(4)
    csv = tmp_path / "sample.csv"
    source.to_csv(csv)
    process = MagicMock(side_effect=lambda data, *args, **kwargs: (data, {"ok": 1}))
    monkeypatch.setattr(prepare.actipy, "process", process)
    loaded = prepare.load_data(str(csv), sample_rate=1, resample_rate=1)
    assert loaded.shape == source.shape
    assert process.call_args.kwargs["resample_hz"] == 1

    parquet_data = source.copy()
    parquet_data.iloc[0, 0] = np.nan
    monkeypatch.setattr(prepare.pd, "read_parquet", lambda path: parquet_data.copy())
    loaded = prepare.load_data("sample.parquet", sample_rate=1)
    assert len(loaded) == 3


def test_make_windows_and_labels_numerical_outputs():
    data = _annotated_data(8)
    annotations = pd.DataFrame(
        {"label:test": ["walking", "sleep"]}, index=pd.Index(["walk", "sleep"], name="annotation")
    )
    X, y, times = prepare.make_windows(
        data, annotations, "test", winsec=4, sample_rate=1, resample_rate=2
    )
    assert X.shape == (2, 8, 3)
    np.testing.assert_array_equal(y, ["walking", "sleep"])
    assert times.shape == (2,)

    damaged = data.copy()
    damaged.loc[damaged.index[:4], "annotation"] = np.nan
    labels, label_times = prepare.make_labels(
        damaged, annotations, "test", sample_rate=1, winsec=4
    )
    assert labels[0] == "nan"
    assert labels[1] == "sleep"
    assert label_times.shape == (2,)

    damaged.loc[damaged.index[4], "x"] = np.nan
    labels, _ = prepare.make_labels(damaged, annotations, "test", sample_rate=1, winsec=4)
    assert all(pd.isna(value) or str(value) == "nan" for value in labels)


@pytest.mark.parametrize("method", ["nn", "linear"])
def test_load_all_and_make_windows_combines_and_saves(tmp_path, monkeypatch, method):
    annotation_file = tmp_path / "annotations.csv"
    pd.DataFrame({"annotation": ["walk"], "label:test": ["walking"]}).to_csv(
        annotation_file, index=False
    )
    monkeypatch.setattr(prepare, "Parallel", ImmediateParallel)
    monkeypatch.setattr(prepare, "load_data", lambda *args, **kwargs: _annotated_data(4))
    monkeypatch.setattr(
        prepare,
        "make_windows",
        lambda *args, **kwargs: (
            np.ones((1, 3, 3)),
            np.array(["walking"]),
            np.array([[pd.Timestamp("2024-01-01")]]),
        ),
    )
    output = tmp_path / f"arrays-{method}"
    X, y, times, participants = prepare.load_all_and_make_windows(
        ["P001.csv", "P002.csv"],
        str(annotation_file),
        out_dir=str(output),
        anno_label="test",
        downsampling_method=method,
        n_jobs=1,
    )
    assert X.shape == (2, 3, 3)
    np.testing.assert_array_equal(participants, ["P001", "P002"])
    assert (output / "X.npy").exists()
    assert json.loads((output / "info.json").read_text())["downsampling_method"] == method


def test_load_all_rejects_unknown_downsampling_method(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame({"annotation": ["walk"], "label:test": ["walking"]}).to_csv(
        annotations, index=False
    )
    monkeypatch.setattr(prepare, "Parallel", ImmediateParallel)
    with pytest.raises(ValueError, match="Invalid downsampling"):
        prepare.load_all_and_make_windows(["P001.csv"], str(annotations), downsampling_method="cubic")


def test_extract_accelerometer_features_invokes_missing_participants(monkeypatch):
    monkeypatch.setattr(prepare, "Parallel", ImmediateParallel)
    monkeypatch.setattr(prepare, "glob", lambda pattern: [])
    run = MagicMock(return_value="complete")
    monkeypatch.setattr(prepare.subprocess, "run", run)
    prepare.extract_accelerometer_features(n_jobs=1)
    assert run.call_count == 151
    assert run.call_args.kwargs == {"shell": True, "capture_output": True}


def test_prepare_participant_accelerometer_data(monkeypatch):
    raw = _annotated_data(4)
    features = prepare.MODEL_CONFIG["Walmsley2020"]["rf_features"]
    feature_frame = pd.DataFrame(
        np.arange(2 * len(features)).reshape(2, len(features)),
        columns=features,
        index=pd.date_range("2024-01-01", periods=2),
    )
    annotations = pd.DataFrame(
        {"label:Walmsley2020": ["light", "sleep"]}, index=["walk", "sleep"]
    )
    reads = iter([raw, feature_frame, annotations])
    monkeypatch.setattr(prepare.pd, "read_csv", lambda *args, **kwargs: next(reads))
    monkeypatch.setattr(
        prepare,
        "make_labels",
        lambda *args, **kwargs: (
            np.array(["light", np.nan], dtype=object),
            np.array([[1], [2]]),
        ),
    )
    X, y, times, participants = prepare.prepare_participant_accelerometer_data(
        7, "annotations.csv", "Walmsley2020", verbose=True
    )
    assert X.shape == (1, len(features))
    np.testing.assert_array_equal(y, ["light"])
    np.testing.assert_array_equal(participants, ["P007"])


def test_prepare_accelerometer_data_combines_participants(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "Parallel", ImmediateParallel)
    monkeypatch.setattr(prepare, "tqdm", lambda values, **kwargs: list(values)[:2])
    monkeypatch.setattr(
        prepare,
        "prepare_participant_accelerometer_data",
        lambda pid, *args: (
            np.array([[pid, pid + 1]]),
            np.array(["label"]),
            np.array([[pid]]),
            np.array([f"P{pid:03}"]),
        ),
    )
    output = tmp_path / "rf"
    X, y, times, participants = prepare.prepare_accelerometer_data(
        "annotations.csv", "Walmsley2020", str(output), 1
    )
    assert X.shape == (2, 2)
    assert y.tolist() == ["label", "label"]
    assert (output / "pid.npy").exists()


def test_download_data_and_make_acc_df(tmp_path, monkeypatch, capsys):
    archive = tmp_path / "download" / "data.zip"
    extract = tmp_path / "extracted"
    retrieve = MagicMock(side_effect=lambda url, path: Path(path).touch())
    monkeypatch.setattr(prepare.request, "urlretrieve", retrieve)
    zip_context = MagicMock()
    monkeypatch.setattr(prepare.zipfile, "ZipFile", MagicMock(return_value=zip_context))
    prepare.download_data("https://example.test/data.zip", str(archive), str(extract), verbose=True)
    retrieve.assert_called_once()
    zip_context.__enter__.return_value.extractall.assert_called_with(str(extract))
    prepare.download_data("unused", str(archive), str(extract), verbose=True)
    assert "Skipping download" in capsys.readouterr().out

    arrays = tmp_path / "arrays"
    arrays.mkdir()
    np.save(arrays / "X.npy", np.array([[[1, 2, 3], [4, 5, 6]]]))
    np.save(arrays / "Y.npy", np.array(["walk"]))
    np.save(arrays / "pid.npy", np.array(["P001"]))
    output = tmp_path / "acc.csv"
    prepare.make_acc_df(str(arrays), str(output), sample_rate=2)
    frame = pd.read_csv(output)
    assert frame[["x", "y", "z"]].values.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert frame["label"].tolist() == ["walk", "walk"]
    prepare.make_acc_df(str(arrays), str(output))
    assert "already exists" in capsys.readouterr().out


class FixedSplitter:
    def split(self, X, y, groups):
        yield np.array([0, 1]), np.array([2, 3])
        yield np.array([2, 3]), np.array([0, 1])


def test_evaluate_preprocessing_runs_group_folds(monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "StratifiedGroupKFold", lambda n_splits: FixedSplitter())
    classifier = MagicMock()
    classifier.predict.side_effect = [np.array([0, 1]), np.array([0, 1])]
    X = np.arange(8).reshape(4, 2)
    y = np.array(["a", "b", "a", "b"])
    groups = np.array(["g1", "g1", "g2", "g2"])
    result = evaluate.evaluate_preprocessing(
        classifier, X, y, groups, np.arange(4), weights_path="weights-{}.pt", verbose=True
    )
    np.testing.assert_array_equal(result, y)
    assert classifier.fit.call_count == 2
    assert "Fold 1 Test Scores" in capsys.readouterr().out


def test_evaluate_models_returns_and_saves_fold_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(evaluate, "StratifiedGroupKFold", lambda n_splits: FixedSplitter())
    actinet_classifier = MagicMock()
    rf_classifier = MagicMock()
    actinet_classifier.predict.side_effect = [np.array([0, 1]), np.array([0, 1])]
    rf_classifier.predict.side_effect = [np.array([0, 1]), np.array([0, 1])]
    X = np.arange(8).reshape(4, 2)
    y = np.array(["a", "b", "a", "b"])
    groups = np.array(["g1", "g1", "g2", "g2"])
    times = pd.date_range("2024-01-01", periods=4, freq="30s").to_numpy()
    actinet_result, rf_result = evaluate.evaluate_models(
        actinet_classifier,
        rf_classifier,
        X,
        X,
        y,
        y,
        groups,
        groups,
        times,
        times,
        weights_path="weights-{}.pt",
        out_dir=str(tmp_path),
        verbose=True,
    )
    assert len(actinet_result) == len(rf_result) == 2
    assert (tmp_path / "actinet_results.pkl").exists()
    assert (tmp_path / "rf_results.pkl").exists()
    assert "Actinet performance" in capsys.readouterr().out
