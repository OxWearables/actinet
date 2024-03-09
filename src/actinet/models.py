import numpy as np
import pandas as pd
import copy
import os
import joblib
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax

from actinet import hmm
from actinet import sslmodel
from actinet.utils.utils import safe_indexer, resize, infer_freq


class ActivityClassifier:
    def __init__(
        self,
        device="cpu",
        batch_size=512,
        window_sec=30,
        weights_path=None,
        labels=[],
        repo_tag="v1.0.0",
        hmm_params=None,
        verbose=False,
    ):
        self.device = device
        self.repo_tag = repo_tag
        self.batch_size = batch_size
        self.window_sec = window_sec
        self.labels = labels
        self.verbose = verbose

        self.model_weights = (
            sslmodel.get_model_dict(weights_path, device) if weights_path else None
        )
        self.model = None

        self.load_hmm_params(hmm_params)

    def __str__(self):
        return (
            "Activity Classifier\n"
            "class_labels: {self.labels}\n"
            "window_length: {self.window_sec}\n"
            "batch_size: {self.batch_size}\n"
            "device: {self.device}\n"
            "hmm: {self.hmms}\n"
            "model: {model}".format(
                self=self, model=self.model or "Model has not been loaded."
            )
        )

    def fit(
        self,
        X,
        Y,
        groups=None,
        T=None,
        weights_path="models/weights.pt",
        model_repo_path=None,
        n_splits=5,
    ):
        sslmodel.verbose = self.verbose

        Y = LabelEncoder().fit_transform(Y)

        if self.verbose:
            print("Training SSL")

        y_prob_splits = []
        y_true_splits = []
        t_splits = []

        if n_splits < 3:
            splitter = GroupShuffleSplit(n_splits=n_splits, random_state=42)
            split_iterator = splitter.split(X, Y, groups)
        else:
            splitter = StratifiedGroupKFold(n_splits)
            split_iterator = splitter.split(X, Y, groups)

        for i, (train_idx, val_idx) in enumerate(split_iterator):
            if self.verbose:
                print(f"Training split {i+1}/{n_splits}")

            x_train = X[train_idx]
            x_val = X[val_idx]

            y_train = Y[train_idx]
            y_val = Y[val_idx]

            group_train = safe_indexer(groups, train_idx)
            group_val = safe_indexer(groups, val_idx)

            t_val = safe_indexer(T, val_idx)

            train_dataset = sslmodel.NormalDataset(
                x_train, y_train, pid=group_train, augmentation=True
            )
            val_dataset = sslmodel.NormalDataset(x_val, y_val, pid=group_val)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=1,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=1,
            )

            self.load_model(model_repo_path)

            if self.model_weights is None or not os.path.exists(weights_path):
                sslmodel.train(
                    self.model,
                    train_loader,
                    val_loader,
                    self.device,
                    weights_path=weights_path,
                    class_weights="balanced",
                )
                self.model.load_state_dict(torch.load(weights_path, self.device))

            # train HMM with predictions of the validation set
            y_val, y_val_pred, _ = sslmodel.predict(
                self.model, val_loader, self.device, output_logits=True
            )
            y_val_pred_sf = softmax(y_val_pred, axis=1)

            y_true_splits.append(y_val)
            y_prob_splits.append(y_val_pred_sf)
            t_splits.append(t_val)

        y_prob_splits = np.vstack(y_prob_splits)
        y_true_splits = np.hstack(y_true_splits)
        t_splits = np.hstack(t_splits)

        if self.verbose:
            print("Training HMM")

        self.hmms.fit(y_prob_splits, y_true_splits, t_splits, self.window_sec)

        # move model to cpu to get a device-less state dict (prevents device conflicts when loading on cpu/gpu later)
        self.model.to("cpu")
        self.model_weights = self.model.state_dict()

        return self

    def predict_from_frame(self, data, sample_freq, hmm_smothing=True):
        sample_freq = sample_freq or infer_freq(data.index)
        X, T = make_windows(
            data,
            self.window_sec,
            self.window_sec * sample_freq,
            return_index=True,
            verbose=self.verbose,
        )

        Y = raw_to_df(
            X, self.predict(X, hmm_smothing, T), T, self.labels, reindex=False
        )

        return Y

    def predict(self, X, hmm_smothing=True, T=None):
        if self.model is None:
            raise Exception("Model has not been loaded for ActivityClassifier.")

        self.model.to(self.device)

        # check X quality
        ok = np.flatnonzero(~np.asarray([np.isnan(x).any() for x in X]))

        X_ = X[ok]
        T_ = safe_indexer(T, ok)

        sslmodel.verbose = self.verbose

        dataset = sslmodel.NormalDataset(X_)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        _, Y_, _ = sslmodel.predict(
            self.model, dataloader, self.device, output_logits=False
        )

        if hmm_smothing:
            interval = self.window_sec if T_ is not None else None
            Y_ = self.hmms.predict(Y_, T_, interval)

        Y = np.full(len(X), fill_value=np.nan)
        Y[ok] = Y_

        return Y

    def load_model(self, model_repo_path=None):
        self.model = sslmodel.get_sslnet(
            tag=self.repo_tag,
            local_repo_path=model_repo_path,
            pretrained_weights=self.model_weights or True,
            window_sec=self.window_sec,
            num_labels=len(self.labels),
        )
        self.model.to(self.device)

    def load_hmm_params(self, hmm_params):
        if isinstance(hmm_params, str):
            if os.path.exists(hmm_params):
                if self.verbose:
                    print(f"Loading hmm_params from {hmm_params}")

                hmm_params = dict(np.load(hmm_params, allow_pickle=True))

            else:
                raise FileNotFoundError(
                    "Path to file with saved hmm parameters cannot be found."
                )

        elif hmm_params is None:
            hmm_params = dict()

        elif not isinstance(hmm_params, dict):
            raise TypeError(
                "Invalid type for HMM parameters. Expected str, dict, or None."
            )

        self.hmms = hmm.HMM(**hmm_params)

    def save(self, output_path):
        classifier = copy.deepcopy(self)
        classifier.model = None
        classifier.device = "cpu"
        classifier.batch_size = 512

        joblib.dump(classifier, output_path, compress=("lzma", 3))


def make_windows(data, window_sec, window_len, return_index=False, verbose=True):
    """Split data into windows"""

    if verbose:
        print("Defining windows...")

    X, T = [], []
    acc_cols = ["x", "y", "z"]
    ssl_window_len = int(sslmodel.SAMPLE_RATE * window_sec)

    for t, x in tqdm(
        data.resample(f"{window_sec}s", origin="start"),
        mininterval=5,
        disable=not verbose,
    ):
        n = len(x)
        x = x[acc_cols].to_numpy()

        if n == window_len:
            x = x
        elif n > window_len:
            x = x[:window_len]
        elif n < window_len and n > window_len / 2:
            x = np.pad(x, ((0, window_len - n), (0, 0)), mode="wrap")
        else:
            x = np.full((window_len, 3), np.nan)

        X.append(x)
        T.append(t)

    X = np.asarray(X)

    if window_len != ssl_window_len:
        X = resize(X, ssl_window_len)

    if return_index:
        T = pd.DatetimeIndex(T, name=data.index.name)
        return X, T

    return X


def raw_to_df(data, labels, time, classes, reindex=True, freq="30S"):
    """
    Construct a DataFrome from the raw data, prediction labels and time Numpy arrays.

    :param data: Numpy windowed acc data, shape (rows, window_len, 3)
    :param labels: Either a scalar label array with shape (rows, ),
                    or the probabilities for each class if label_proba==True with shape (rows, len(classes)).
    :param time: Numpy time array, shape (rows, )
    :param classes: Array with the categorical class labels.
                    The index of this array should correspond to the labels value when label_proba==False.
    :param reindex: Reindex the dataframe to fill missing values
    :param freq: Reindex frequency
    :return: Dataframe
        Index: DatetimeIndex
        Columns: acc, classes
    :rtype: pd.DataFrame
    """
    label_matrix = np.zeros((len(time), len(classes)), dtype=np.float32)
    a_matrix = np.zeros(len(time), dtype=np.float32)

    for i, data in enumerate(data):
        if np.isnan(labels[i]):
            label_matrix[i, :] = np.nan
            a_matrix[i] = np.nan
            continue

        label = int(labels[i])
        label_matrix[i, label] = 1

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        enmo = (np.sqrt(x**2 + y**2 + z**2) - 1) * 1000  # in milli gravity
        enmo[enmo < 0] = 0
        a_matrix[i] = np.mean(enmo)

    datadict = {
        **{"time": time, "acc": a_matrix},
        **{classes[i]: label_matrix[:, i] for i in range(len(classes))},
    }

    df = pd.DataFrame(datadict)
    df = df.set_index("time")

    if reindex:
        newindex = pd.date_range(df.index[0], df.index[-1], freq=freq)
        df = df.reindex(newindex, method="nearest", fill_value=np.nan, tolerance="5S")

    return df
