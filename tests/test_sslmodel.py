from collections import OrderedDict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from actinet import sslmodel


def test_axis_transforms_preserve_shape_and_vector_magnitude(monkeypatch):
    sample = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    switch = sslmodel.RandomSwitchAxis()
    for choice in range(1, 7):
        monkeypatch.setattr(sslmodel.random, "randint", lambda *args, value=choice: value)
        result = switch(sample)
        assert result.shape == sample.shape
        assert sorted(map(tuple, result.tolist())) == sorted(map(tuple, sample.tolist()))

    np.random.seed(4)
    array = np.random.randn(3, 20).astype("f4")
    rotated = sslmodel.RotationAxis()(array)
    np.testing.assert_allclose(
        np.linalg.norm(rotated, axis=0), np.linalg.norm(array, axis=0), rtol=1e-5
    )
    monkeypatch.setattr(sslmodel.random, "randint", lambda *args: 2)
    decimated = sslmodel.RandomDecimation()(sample)
    assert decimated.shape == sample.shape


def test_normal_dataset_labels_participants_transpose_and_augmentation(monkeypatch):
    X = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    dataset = sslmodel.NormalDataset(X, y=[1, 0], pid=np.array(["a", "b"]))
    sample, label, participant = dataset[torch.tensor(1)]
    assert sample.shape == (3, 4)
    assert label.item() == 0
    assert participant == "b"
    plain = sslmodel.NormalDataset(X, transpose_channels_first=False)
    assert plain[0][0].shape == (4, 3)
    assert np.isnan(plain[0][1])
    assert np.isnan(plain[0][2])
    augmented = sslmodel.NormalDataset(X, augmentation=True)
    monkeypatch.setattr(augmented, "transform", lambda value: value + 1)
    np.testing.assert_allclose(augmented[0][0], torch.from_numpy(X.astype("f4")[0].T) + 1)


def test_early_stopping_saves_improvements_and_stops(tmp_path):
    messages = []
    path = tmp_path / "checkpoint.pt"
    stopping = sslmodel.EarlyStopping(
        patience=2, verbose=True, delta=0.01, path=str(path), trace_func=messages.append
    )
    model = nn.Linear(2, 2)
    stopping(1.0, model)
    stopping(1.1, model)
    stopping(1.2, model)
    assert stopping.early_stop is True
    assert path.exists()
    assert any("Validation loss decreased" in message for message in messages)
    stopping(0.5, model)
    assert stopping.counter == 0

    wrapped = MagicMock()
    wrapped.module = model
    sslmodel.EarlyStopping(path=str(tmp_path / "wrapped.pt")).save_checkpoint(1.0, wrapped)
    assert (tmp_path / "wrapped.pt").exists()


def test_get_sslnet_validates_and_loads_local_repository(monkeypatch):
    with pytest.raises(ValueError, match="window"):
        sslmodel.get_sslnet(window_sec=12)
    with pytest.raises(ValueError, match="class labels"):
        sslmodel.get_sslnet(num_labels=0)
    net = nn.Linear(2, 2)
    hub_load = MagicMock(return_value=net)
    monkeypatch.setattr(sslmodel.torch.hub, "load", hub_load)
    result = sslmodel.get_sslnet("v1", local_repo_path="repo", window_sec=5, num_labels=2)
    assert result is net
    assert hub_load.call_args.kwargs["source"] == "local"


def test_get_sslnet_uses_cached_repo_and_ordered_weights(tmp_path, monkeypatch, capsys):
    cached = tmp_path / "OxWearables_ssl-wearables_v1.0.0"
    cached.mkdir()
    monkeypatch.setattr(sslmodel, "torch_cache_path", tmp_path)
    monkeypatch.setattr(sslmodel, "verbose", True)
    net = nn.Linear(2, 2)
    weights = OrderedDict((key, value.clone()) for key, value in net.state_dict().items())
    hub_load = MagicMock(return_value=net)
    monkeypatch.setattr(sslmodel.torch.hub, "load", hub_load)
    monkeypatch.setattr(sslmodel.torch.hub, "set_dir", MagicMock())
    assert sslmodel.get_sslnet(pretrained_weights=weights, num_labels=2) is net
    assert hub_load.call_args.kwargs["source"] == "local"
    assert "Using local" in capsys.readouterr().out


def test_get_sslnet_creates_cache_and_selects_github(tmp_path, monkeypatch):
    cache = tmp_path / "new-cache"
    monkeypatch.setattr(sslmodel, "torch_cache_path", cache)
    hub_load = MagicMock(return_value=nn.Linear(2, 2))
    monkeypatch.setattr(sslmodel.torch.hub, "load", hub_load)
    monkeypatch.setattr(sslmodel.torch.hub, "set_dir", MagicMock())
    sslmodel.get_sslnet(tag="v9", num_labels=2)
    assert cache.is_dir()
    assert hub_load.call_args.kwargs["source"] == "github"


def test_model_dictionary_roundtrip(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([1.0])}, path)
    loaded = sslmodel.get_model_dict(path, "cpu")
    torch.testing.assert_close(loaded["weight"], torch.tensor([1.0]))


def test_predict_returns_labels_logits_and_empty_arrays(monkeypatch):
    monkeypatch.setattr(sslmodel, "verbose", False)
    X = np.array([[[1.0, 0, 0], [2.0, 0, 0]], [[-1.0, 0, 0], [-2.0, 0, 0]]])
    dataset = sslmodel.NormalDataset(X, y=np.array([0, 1]), pid=np.array([10, 11]))
    loader = DataLoader(dataset, batch_size=2)

    class Net(nn.Module):
        def forward(self, inputs):
            score = inputs[:, 0].mean(axis=1)
            return torch.stack([score, -score], dim=1)

    truth, prediction, participants = sslmodel.predict(Net(), loader, "cpu")
    np.testing.assert_array_equal(truth, [0, 1])
    np.testing.assert_array_equal(prediction, [0, 1])
    np.testing.assert_array_equal(participants, [10, 11])
    _, logits, _ = sslmodel.predict(Net(), loader, "cpu", output_logits=True)
    np.testing.assert_allclose(logits, [[1.5, -1.5], [-1.5, 1.5]])

    empty = DataLoader(sslmodel.NormalDataset(np.empty((0, 2, 3))), batch_size=2)
    outputs = sslmodel.predict(Net(), empty, "cpu")
    assert all(value.size == 0 for value in outputs)


def test_training_validation_and_inverse_weights(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sslmodel, "verbose", False)
    torch.manual_seed(2)
    X = np.random.default_rng(2).normal(size=(6, 4, 3)).astype("f4")
    y = np.array([0, 0, 0, 1, 1, 1])
    loader = DataLoader(sslmodel.NormalDataset(X, y=y), batch_size=3)
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
    result = sslmodel.train(
        model,
        loader,
        loader,
        "cpu",
        class_weights="balanced",
        weights_path=str(tmp_path / "weights.pt"),
        num_epoch=2,
        patience=1,
    )
    assert result is model
    assert (tmp_path / "weights.pt").exists()
    loss, accuracy = sslmodel._validate_model(model, loader, "cpu", nn.CrossEntropyLoss())
    assert np.isfinite(loss)
    assert 0 <= accuracy <= 1
    assert sslmodel.get_inverse_class_weights(np.array([0, 0, 1])) == [1.5, 3.0]
    assert "Inverse class weights" in capsys.readouterr().out
