import numpy as np
import pandas as pd
import copy
import os
import joblib
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, cast, overload

from actinet import hmm
from actinet import sslmodel
from actinet.utils.utils import safe_indexer, resize, infer_freq
from actinet.utils.sleep_utils import removeSpuriousSleep

TimeSequence = Union[np.ndarray, pd.Index, Sequence[Any]]


class ActivityClassifier:
    """
    Implement a ResNet-18 based Activity Classifier with model saving/loading and optional HMM smoothing.
    """

    def __init__(
        self,
        device: Any = (
            "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        ),
        batch_size: int = 512,
        window_sec: int = 30,
        weights_path: Optional[str] = None,
        labels: Optional[Sequence[Any]] = None,
        repo_tag: str = "v1.0.0",
        hmm_params: Optional[Union[str, Mapping[str, Any]]] = None,
        hmm_ignore_transition_gaps: bool = False,
        hmm_handle_sleep_transitions: bool = False,
        verbose: bool = False,
    ) -> None:
        self.device = device
        self.repo_tag = repo_tag
        self.batch_size = batch_size
        self.window_sec = window_sec
        self.labels = list(np.unique(labels if labels is not None else []))
        self.verbose = verbose

        self.model_weights: Any = (
            sslmodel.get_model_dict(weights_path, device) if weights_path else None
        )
        self.model: Any = None

        self.hmm = load_hmm_params(
            hmm_params,
            hmm_ignore_transition_gaps,
            hmm_handle_sleep_transitions,
            verbose,
        )

    def __str__(self) -> str:
        return (
            "Activity Classifier\n"
            "class_labels: {self.labels}\n"
            "window_length: {self.window_sec}\n"
            "batch_size: {self.batch_size}\n"
            "device: {self.device}\n"
            "hmm: {self.hmm}\n"
            "model: {model}".format(
                self=self, model=self.model or "Model has not been loaded."
            )
        )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        T: Optional[TimeSequence] = None,
        weights_path: str = "models/weights.pt",
        model_repo_path: Optional[str] = None,
        n_splits: int = 5,
    ) -> "ActivityClassifier":
        """
        Fit the ActivityClassifier to the provided data by training the model.

        :param X: The training accelerometer data [x,y,z] with shape (rows, window_len, 3)
        :type X: numpy.ndarray
        :param Y: Ground truth labels of the training data with shape (rows, )
        :type Y: numpy.ndarray
        :param groups: Participant labels for the training data with shape (rows, )
        :type groups: numpy.ndarray, optional
        :param T: Time at each observation with shape (rows, )
        :type T: numpy.ndarray, optional
        :param weights_path: Path to save the model weights
        :type weights_path: str, optional
        :param model_repo_path: Path to the ssl-wearables model repository (https://github.com/OxWearables/ssl-wearables)
        :type model_repo_path: str, optional
        :param n_splits: Number of splits for cross-validation
        :type n_splits: int, optional
        """
        sslmodel.verbose = self.verbose

        Y = LabelEncoder().fit_transform(Y)

        if self.verbose:
            print("Training SSL")

        y_prob_splits: List[Any] = []
        y_true_splits: List[Any] = []
        group_splits: List[Any] = []
        t_splits: List[Any] = []

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
                self.model.load_state_dict(
                    torch.load(weights_path, map_location=self.device)
                )

            # train HMM with predictions of the validation set
            y_val, y_val_pred, _ = sslmodel.predict(
                self.model, val_loader, self.device, output_logits=True
            )
            y_val_pred_sf = softmax(y_val_pred, axis=1)

            y_true_splits.append(y_val)
            y_prob_splits.append(y_val_pred_sf)
            group_splits.append(group_val)
            t_splits.append(t_val)

        y_prob = np.vstack(y_prob_splits)
        y_true = np.hstack(y_true_splits)
        group_values = np.hstack(group_splits)
        time_values = np.hstack(t_splits)

        if self.verbose:
            print("Training HMM")

        self.hmm.fit(
            y_prob,
            y_true,
            group_values,
            time_values,
            interval=self.window_sec,
        )

        # move model to cpu to get a device-less state dict (prevents device conflicts when loading on cpu/gpu later)
        self.model.to("cpu")
        self.model_weights = self.model.state_dict()

        return self

    def predict_from_frame(
        self,
        data: pd.DataFrame,
        sample_freq: Optional[float],
        hmm_smothing: bool = True,
        sleep_tolerance: Optional[str] = None,
        remove_naps: bool = False,
    ) -> pd.DataFrame:
        """
        Use the ActivityClassifier to make predictions on input accelerometer data.

        :param data: The input accelerometer data [x,y,z]
        :type data: pandas.DataFrame
        :param sample_freq: Sampling frequency of the accelerometer data
        :type sample_freq: int or float
        :param hmm_smothing: Whether to apply HMM smoothing to the predictions
        :type hmm_smothing: bool, optional
        :param sleep_tolerance: Time threshold for sleep periods to be considered valid (e.g., '30min')
        :type sleep_tolerance: str, optional
        :param remove_naps: Whether to remove nap periods from the predictions
        :type remove_naps: bool, optional

        :raises ValueError: If the sample frequency cannot be inferred or the
            data contains no valid prediction window.
        """
        if sample_freq is None or sample_freq is False:
            if len(data.index) < 2:
                raise ValueError(
                    "Input data must contain at least two timestamps to infer "
                    "the sample frequency."
                )
            sample_period = infer_freq(data.index)
            if pd.isna(sample_period) or sample_period <= pd.Timedelta(0):
                raise ValueError("Could not infer a valid sample frequency.")
            sample_freq = 1 / sample_period.total_seconds()

        X, T = make_windows(
            data,
            self.window_sec,
            int(self.window_sec * sample_freq),
            return_index=cast(Literal[True], True),
            verbose=self.verbose,
        )

        if X.ndim != 3 or not (~np.isnan(X).any(axis=(1, 2))).any():
            raise ValueError(
                "Input data does not contain enough valid samples for a "
                "prediction window."
            )

        Y = raw_to_df(
            X,
            self.predict(X, T, hmm_smothing, sleep_tolerance, remove_naps),
            T,
            self.labels,
            reindex=False,
        )

        return Y

    def predict(
        self,
        X: np.ndarray,
        T: Optional[TimeSequence] = None,
        hmm_smothing: bool = True,
        sleep_tol: Optional[str] = None,
        remove_naps: bool = False,
    ) -> np.ndarray:
        """
        Use the ActivityClassifier to make predictions on input accelerometer data.

        :param X: The input accelerometer data [x,y,z] with shape (rows, window_len, 3)
        :type X: numpy.ndarray
        :param T: Time at each observation with shape (rows, )
        :type T: numpy.ndarray, optional
        :param hmm_smothing: Whether to apply HMM smoothing to the predictions
        :type hmm_smothing: bool, optional
        :param sleep_tol: Time threshold for sleep periods to be considered valid (e.g., '30min')
        :type sleep_tol: str, optional
        """
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
            Y_ = self.hmm.predict(Y_, T_, interval)

        Y = np.full(len(X), fill_value=np.nan)
        Y[ok] = Y_

        Y = removeSpuriousSleep(Y, self.labels, self.window_sec, sleep_tol, remove_naps)

        return Y

    def load_model(self, model_repo_path: Optional[str] = None) -> None:
        """
        Load SSL model reposiotory from specified path. (https://github.com/OxWearables/ssl-wearables)

        :param model_repo_path: Path to the ssl-wearables model repository
        :type model_repo_path: str, optional
        """
        self.model = sslmodel.get_sslnet(
            tag=self.repo_tag,
            local_repo_path=model_repo_path,
            pretrained_weights=self.model_weights or True,
            window_sec=self.window_sec,
            num_labels=len(self.labels),
        )
        self.model.to(self.device)

        if self.verbose:
            print(f"Using pytorch device: {self.device}")

    def save(self, output_path: str) -> None:
        """
        Save the ActivityClassifier model to a .lzma file.

        :param output_path: lzma file location
        :type output_path: str
        """
        classifier = copy.deepcopy(self)
        classifier.model = None
        classifier.device = "cpu"
        classifier.batch_size = 512

        joblib.dump(classifier, output_path, compress=("lzma", 3))


class RFActivityClassifier:
    """
    Implement a basic Balanced Random Forest classifier with optional HMM smoothing.
    """

    def __init__(
        self,
        winsec: Optional[float] = None,
        hmm_params: Optional[Union[str, Mapping[str, Any]]] = None,
        hmm_ignore_transition_gaps: bool = False,
        hmm_handle_sleep_transitions: bool = False,
        labels: Optional[Sequence[Any]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:

        self.model = BalancedRandomForestClassifier(
            oob_score=True, verbose=verbose, **kwargs
        )
        self.labels = list(np.unique(labels if labels is not None else []))
        self.hmm = load_hmm_params(
            hmm_params,
            hmm_ignore_transition_gaps,
            hmm_handle_sleep_transitions,
            verbose,
        )
        self.winsec = winsec

    def __str__(self) -> str:
        return str(self.model)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        T: Optional[TimeSequence] = None,
    ) -> None:
        if self.winsec is None:
            raise ValueError("winsec must be set before fitting the classifier")
        self.model.fit(X, Y)
        self.hmm.fit(self.model.oob_decision_function_, Y, groups, T, self.winsec)

    def predict(
        self,
        X: np.ndarray,
        T: Optional[TimeSequence] = None,
        hmm_smothing: bool = True,
        sleep_tol: Optional[str] = None,
        remove_naps: bool = False,
    ) -> np.ndarray:
        if self.winsec is None:
            raise ValueError("winsec must be set before making predictions")
        y_pred = self.model.predict(X)

        if hmm_smothing:
            y_pred = self.hmm.predict(y_pred, T, self.winsec)

        y_pred = removeSpuriousSleep(
            y_pred, self.labels, self.winsec, sleep_tol, remove_naps
        )

        return y_pred

    def save(self, output_path: str) -> None:
        classifier = copy.deepcopy(self)

        joblib.dump(classifier, output_path, compress=("lzma", 3))

    def load(self, input_path: str) -> None:
        classifier = joblib.load(input_path)
        self.model = classifier.model
        self.labels = classifier.labels
        self.hmm = classifier.hmm
        self.winsec = classifier.winsec


@overload
def make_windows(
    data: pd.DataFrame,
    window_sec: int,
    window_len: int,
    return_index: Literal[True],
    verbose: bool = True,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    ...


@overload
def make_windows(
    data: pd.DataFrame,
    window_sec: int,
    window_len: int,
    return_index: Literal[False] = False,
    verbose: bool = True,
) -> np.ndarray:
    ...


def make_windows(
    data: pd.DataFrame,
    window_sec: int,
    window_len: int,
    return_index: bool = False,
    verbose: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, pd.DatetimeIndex]]:
    """Split data into windows"""

    if verbose:
        print("Defining windows...")

    windows: List[np.ndarray] = []
    times: List[Any] = []
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

        windows.append(x)
        times.append(t)

    X = np.asarray(windows)

    if window_len != ssl_window_len:
        X = resize(X, ssl_window_len)

    if return_index:
        time_index = pd.DatetimeIndex(times, name=data.index.name)
        return X, time_index

    return X


def raw_to_df(
    data: np.ndarray,
    labels: np.ndarray,
    time: Sequence[Any],
    classes: Sequence[str],
    reindex: bool = True,
    freq: str = "30S",
) -> pd.DataFrame:
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


def load_hmm_params(
    hmm_params: Optional[Union[str, Mapping[str, Any]]],
    ignore_transition_gaps: bool,
    handle_sleep_transitions: bool,
    verbose: bool = False,
) -> hmm.HMM:
    if isinstance(hmm_params, str):
        if os.path.exists(hmm_params):
            if verbose:
                print(f"Loading hmm_params from {hmm_params}")

            params: Dict[str, Any] = dict(np.load(hmm_params, allow_pickle=True))

        else:
            raise FileNotFoundError(
                "Path to file with saved hmm parameters cannot be found."
            )

    elif hmm_params is None:
        params = {}

    elif not isinstance(hmm_params, Mapping):
        raise TypeError("Invalid type for HMM parameters. Expected str, dict, or None.")

    else:
        params = dict(hmm_params)

    params.update(
        {
            "ignore_transition_gaps": ignore_transition_gaps,
            "handle_sleep_transitions": handle_sleep_transitions,
        }
    )

    return hmm.HMM(**params)
