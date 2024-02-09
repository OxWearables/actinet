import numpy as np
import pandas as pd
import copy
import os
import joblib
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax

from actinet import hmm
from actinet import sslmodel
from actinet.utils.utils import safe_indexer


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
        self.window_len = int(np.ceil(self.window_sec * sslmodel.SAMPLE_RATE))
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
    ):
        sslmodel.verbose = self.verbose

        Y = LabelEncoder().fit_transform(Y)

        if self.verbose:
            print("Training SSL")

        # prepare training and validation sets
        folds = GroupShuffleSplit(1, test_size=0.2, random_state=41).split(
            X, Y, groups=groups
        )
        train_idx, val_idx = next(folds)

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

        sslmodel.train(
            self.model, train_loader, val_loader, self.device, weights_path=weights_path
        )
        self.model.load_state_dict(torch.load(weights_path, self.device))

        if self.verbose:
            print("Training HMM")

        # train HMM with predictions of the validation set
        y_val, y_val_pred, group_val = sslmodel.predict(
            self.model, val_loader, self.device, output_logits=True
        )
        y_val_pred_sf = softmax(y_val_pred, axis=1)

        self.hmms.fit(y_val_pred_sf, y_val, t_val, 1 / sslmodel.SAMPLE_RATE)

        # move model to cpu to get a device-less state dict (prevents device conflicts when loading on cpu/gpu later)
        self.model.to("cpu")
        self.model_weights = self.model.state_dict()

        return self

    def predict_from_frame(self, data):

        def fn(chunk):
            """Process the chunk. Apply padding if length is not enough."""
            n = len(chunk)
            x = chunk[["x", "y", "z"]].to_numpy()
            if n == self.window_len:
                x = x
            elif n > self.window_len:
                x = x[: self.window_len]
            elif n < self.window_len and n > self.window_len / 2:
                m = self.window_len - n
                x = np.pad(x, ((0, m), (0, 0)), mode="wrap")
            else:
                x = np.full((self.window_len, 3), fill_value=np.nan)
            return x

        X, T = make_windows(
            data, self.window_sec, fn=fn, return_index=True, verbose=self.verbose
        )

        Y = raw_to_df(X, self._predict(X), T, self.labels, reindex=False)

        return Y

    def _predict(self, X):
        if self.model is None:
            raise Exception("Model has not been loaded for ActivityClassifier.")

        sslmodel.verbose = self.verbose

        dataset = sslmodel.NormalDataset(X)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        _, y_pred, _ = sslmodel.predict(
            self.model, dataloader, self.device, output_logits=False
        )

        y_pred = self.hmms.predict(y_pred)

        return y_pred

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


def make_windows(data, window_sec, fn=None, return_index=False, verbose=True):
    """Split data into windows"""

    if verbose:
        print("Defining windows...")

    if fn is None:

        def fn(x):
            return x

    X, T = [], []
    for t, x in tqdm(
        data.resample(f"{window_sec}s", origin="start"),
        mininterval=5,
        disable=not verbose,
    ):
        x = fn(x)
        X.append(x)
        T.append(t)

    X = np.asarray(X)

    if return_index:
        T = pd.DatetimeIndex(T, name=data.index.name)
        return X, T

    return X


def raw_to_df(data, labels, time, classes, label_proba=False, reindex=True, freq="30S"):
    """
    Construct a DataFrome from the raw data, prediction labels and time Numpy arrays.

    :param data: Numpy windowed acc data, shape (rows, window_len, 3)
    :param labels: Either a scalar label array with shape (rows, ),
                    or the probabilities for each class if label_proba==True with shape (rows, len(classes)).
    :param time: Numpy time array, shape (rows, )
    :param classes: Array with the categorical class labels.
                    The index of this array should correspond to the labels value when label_proba==False.
    :param label_proba: If True, assume 'labels' contains the raw class probabilities.
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
        if not label_proba:
            label = int(labels[i])
            label_matrix[i, label] = 1

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        enmo = (np.sqrt(x**2 + y**2 + z**2) - 1) * 1000  # in milli gravity
        enmo[enmo < 0] = 0
        a_matrix[i] = np.mean(enmo)

    if label_proba:
        datadict = {
            **{"time": time, "acc": a_matrix},
            **{classes[i]: label_matrix[:, i] for i in range(len(classes))},
        }
    else:
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
