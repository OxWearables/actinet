import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from actinet import hmm
from actinet import sslmodel


class ActivityClassifier:
    def __init__(
        self,
        device="cpu",
        batch_size=512,
        window_sec=30,
        weights_path="state_dict.pt",
        labels=[],
        repo_tag="v1.0.0",
        hmm_params=None,
        verbose=False,
    ):
        self.device = device
        self.weights_path = weights_path
        self.repo_tag = repo_tag
        self.batch_size = batch_size
        self.window_sec = window_sec
        self.state_dict = None
        self.label_encoder = LabelEncoder().fit(labels)
        self.window_len = int(np.ceil(self.window_sec * sslmodel.SAMPLE_RATE))

        self.verbose = verbose

        hmm_params = hmm_params or dict()
        self.hmms = hmm.HMM(**hmm_params)

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

        Y_labels = self.label_encoder.inverse_transform(self._predict(X))

        Y = raw_to_df(X, Y_labels, T, self.label_encoder.classes_, reindex=False)

        return Y

    def _predict(self, X, groups=None):
        sslmodel.verbose = self.verbose

        dataset = sslmodel.NormalDataset(X, name="prediction")
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = sslmodel.get_sslnet(
            tag=self.repo_tag,
            pretrained=False,
            window_sec=self.window_sec,
            num_labels=len(self.label_encoder.classes_),
        )
        model.load_state_dict(self.state_dict)
        model.to(self.device)

        _, y_pred, _ = sslmodel.predict(
            model, dataloader, self.device, output_logits=False
        )

        y_pred = self.hmms.predict(y_pred, groups=groups)

        return y_pred


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
