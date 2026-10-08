from collections import OrderedDict
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from actinet import models


def test_make_windows_handles_exact_long_short_and_resampling(monkeypatch):
    index = pd.date_range("2024-01-01", periods=26, freq="1s", name="time")
    data = pd.DataFrame(
        {"x": np.arange(26), "y": np.arange(26) + 1, "z": np.ones(26)}, index=index
    )
    monkeypatch.setattr(models.sslmodel, "SAMPLE_RATE", 1)
    windows, times = models.make_windows(data, 10, 10, return_index=True, verbose=False)
    assert windows.shape == (3, 10, 3)
    np.testing.assert_array_equal(windows[0, :, 0], np.arange(10))
    np.testing.assert_array_equal(windows[2, :, 0], [20, 21, 22, 23, 24, 25, 20, 21, 22, 23])
    assert times.name == "time"

    tiny = data.iloc[:4]
    bad = models.make_windows(tiny, 10, 10, verbose=False)
    assert np.isnan(bad).all()

    monkeypatch.setattr(models.sslmodel, "SAMPLE_RATE", 2)
    resized = models.make_windows(data.iloc[:10], 10, 10, verbose=False)
    assert resized.shape == (1, 20, 3)


def test_raw_to_df_pins_one_hot_labels_enmo_and_missingness():
    raw = np.array(
        [
            [[0, 0, 1], [0, 0, 1]],
            [[0, 0, 2], [0, 0, 2]],
            [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        ],
        dtype=float,
    )
    time = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:03"])
    result = models.raw_to_df(raw, np.array([0, 1, np.nan]), time, ["sleep", "light"], reindex=False)
    np.testing.assert_allclose(result.iloc[0], [0, 1, 0])
    np.testing.assert_allclose(result.iloc[1], [1000, 0, 1])
    assert result.iloc[2].isna().all()

    reindexed = models.raw_to_df(raw, np.array([0, 1, np.nan]), time, ["sleep", "light"], freq="1min")
    assert len(reindexed) == 4
    assert reindexed.iloc[2].isna().all()


def test_load_hmm_params_accepts_dict_file_and_none(tmp_path, capsys):
    params = {
        "prior": np.array([0.5, 0.5]),
        "emission": np.eye(2),
        "transition": np.eye(2),
        "labels": np.array([0, 1]),
    }
    from_dict = models.load_hmm_params(params.copy(), True, False)
    assert from_dict.ignore_transition_gaps is True
    assert models.load_hmm_params(None, False, True).handle_sleep_transitions is True
    path = tmp_path / "params.npz"
    np.savez(path, **params)
    loaded = models.load_hmm_params(str(path), False, False, verbose=True)
    np.testing.assert_allclose(loaded.prior, params["prior"])
    assert "Loading hmm_params" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError):
        models.load_hmm_params(str(tmp_path / "missing.npz"), False, False)
    with pytest.raises(TypeError):
        models.load_hmm_params([], False, False)


def test_activity_classifier_predict_and_frame_conversion(monkeypatch):
    classifier = models.ActivityClassifier(
        labels=["sleep", "sedentary"], window_sec=5, batch_size=2, verbose=True
    )
    classifier.model = MagicMock()
    classifier.hmm = MagicMock()
    classifier.hmm.predict.return_value = np.array([1])
    monkeypatch.setattr(
        models.sslmodel,
        "predict",
        lambda *args, **kwargs: (np.array([np.nan]), np.array([0]), np.array([np.nan])),
    )
    good = np.zeros((5, 3), dtype=float)
    good[:, 2] = 1
    raw = np.stack([good, np.full_like(good, np.nan)])
    times = pd.date_range("2024-01-01", periods=2, freq="5s")
    predicted = classifier.predict(raw, times, hmm_smothing=True)
    np.testing.assert_allclose(predicted[:1], [1])
    assert np.isnan(predicted[1])
    classifier.hmm.predict.assert_called_once()
    assert "Activity Classifier" in str(classifier)

    frame_index = pd.date_range("2024-01-01", periods=10, freq="1s", name="time")
    frame = pd.DataFrame(np.tile([0.0, 0.0, 1.0], (10, 1)), columns=["x", "y", "z"], index=frame_index)
    monkeypatch.setattr(classifier, "predict", lambda X, *args: np.zeros(len(X)))
    output = classifier.predict_from_frame(frame, sample_freq=None, hmm_smothing=False)
    assert output.columns.tolist() == ["acc", "sedentary", "sleep"]
    assert len(output) == 2
    assert output["sedentary"].tolist() == [1, 1]


def test_activity_classifier_requires_loaded_model():
    classifier = models.ActivityClassifier(labels=["sleep"])
    with pytest.raises(Exception, match="has not been loaded"):
        classifier.predict(np.zeros((1, 900, 3)))


def test_activity_classifier_load_and_save(tmp_path, monkeypatch, capsys):
    network = MagicMock()
    monkeypatch.setattr(models.sslmodel, "get_sslnet", MagicMock(return_value=network))
    classifier = models.ActivityClassifier(labels=["sleep", "light"], verbose=True)
    classifier.load_model("local-repo")
    assert classifier.model is network
    network.to.assert_called_with("cpu")
    assert "Using pytorch device" in capsys.readouterr().out

    path = tmp_path / "classifier.joblib.lzma"
    classifier.model = MagicMock()
    classifier.save(path)
    saved = joblib.load(path)
    assert saved.model is None
    assert saved.device == "cpu"
    assert saved.batch_size == 512


def test_activity_classifier_fit_aggregates_validation_predictions(tmp_path, monkeypatch):
    classifier = models.ActivityClassifier(labels=["a", "b"], window_sec=5)
    classifier.model_weights = OrderedDict(weight=torch.tensor([1.0]))
    classifier.hmm = MagicMock()
    model = MagicMock()
    model.state_dict.return_value = OrderedDict(weight=torch.tensor([2.0]))
    monkeypatch.setattr(classifier, "load_model", lambda *args: setattr(classifier, "model", model))
    splitter = MagicMock()
    splitter.split.return_value = [(np.array([0, 1]), np.array([2, 3]))]
    monkeypatch.setattr(models, "GroupShuffleSplit", MagicMock(return_value=splitter))
    monkeypatch.setattr(models, "DataLoader", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(
        models.sslmodel,
        "predict",
        lambda *args, **kwargs: (
            np.array([0, 1]),
            np.array([[2.0, 1.0], [1.0, 2.0]]),
            np.array(["g2", "g2"]),
        ),
    )
    weights = tmp_path / "existing.pt"
    weights.touch()
    X = np.zeros((4, 10, 3), dtype=float)
    y = np.array(["a", "b", "a", "b"])
    groups = np.array(["g1", "g1", "g2", "g2"])
    times = np.arange(4)
    result = classifier.fit(X, y, groups, times, str(weights), n_splits=1)
    assert result is classifier
    classifier.hmm.fit.assert_called_once()
    np.testing.assert_allclose(classifier.model_weights["weight"], [2.0])
    model.to.assert_called_with("cpu")


def test_rf_classifier_delegates_fit_predict_and_roundtrips(tmp_path):
    classifier = models.RFActivityClassifier(winsec=30, labels=["sleep", "sedentary"])
    classifier.model = MagicMock()
    classifier.hmm = MagicMock()
    classifier.model.oob_decision_function_ = np.eye(2)
    classifier.fit(np.zeros((2, 3)), np.array([0, 1]), np.array(["a", "a"]), [0, 30])
    classifier.hmm.fit.assert_called_once()
    classifier.model.predict.return_value = np.array([0, 1])
    classifier.hmm.predict.return_value = np.array([1, 1])
    np.testing.assert_array_equal(classifier.predict(np.zeros((2, 3)), [0, 30]), [1, 1])
    assert str(classifier) == str(classifier.model)

    path = tmp_path / "rf.joblib.lzma"
    classifier.model = models.BalancedRandomForestClassifier()
    classifier.hmm = models.hmm.HMM(labels=np.array([0, 1]))
    classifier.save(path)
    restored = models.RFActivityClassifier(winsec=5, labels=["x"])
    restored.load(path)
    assert restored.winsec == 30
    assert restored.labels == ["sedentary", "sleep"]
