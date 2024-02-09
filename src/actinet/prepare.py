import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import warnings
import actipy

from actinet.utils.utils import is_good_window, resize


def load_data(datafile, sample_rate=100, annot_type=str):
    """
    Load data from a file and process it using actipy.

    Args:
        datafile (str): Path to the data file.
        sample_rate (int): Sampling rate of the data. Defaults to 100.
        annot_type (type): Type of annotation. Defaults to <class 'str'>.

    Returns:
        pd.DataFrame: Processed data.

    """
    if ".parquet" in datafile:
        data = pd.read_parquet(datafile)
        data.dropna(inplace=True)

    else:
        data = pd.read_csv(
            datafile,
            index_col=[0],
            parse_dates=True,
            dtype={"x": "f4", "y": "f4", "z": "f4", "annotation": annot_type},
        )

    data, _ = actipy.process(data, sample_rate, verbose=False)
    return data


def resize(x, length, axis=1):
    """
    Resize the temporal length of the data using linear interpolation.

    Args:
        x (ndarray): Data to be resized.
        length (int): New length of the data.
        axis (int): Axis along which to perform the interpolation. Defaults to 1.

    Returns:
        ndarray: Resized data.

    """
    length_orig = x.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    x = interp1d(t_orig, x, kind="linear", axis=axis, assume_sorted=True)(t_new)
    return x


def make_windows(
    data, anno_dict, anno_label, winsec=30, sample_rate=100, resample_rate=30
):
    """
    Create windows from the input data.

    Args:
        data (pd.DataFrame): Input data.
        anno_dict (pd.DataFrame): Annotation dictionary.
        anno_label (string): Column in annotation dictionary to convert raw annotations to activities.
        winsec (int): Window size in seconds.
        sample_rate (int): Sampling rate of the data.
        resample_rate (int): Resampling rate. Needs to be 30Hz for SSL model.

    Returns:
        tuple: Tuple containing the windowed data, labels, and timestamps.

    """
    X, Y, T = [], [], []

    for t, w in data.resample(f"{winsec}s", origin="start"):

        if len(w) < 1:
            continue

        t = t.to_numpy()

        x = w[["x", "y", "z"]].to_numpy()

        annot = w["annotation"]

        if pd.isna(annot).all():  # skip if annotation is NA
            continue

        if not is_good_window(x, sample_rate, winsec):  # skip if bad window
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unable to sort modes")
            y = (
                anno_dict.loc[annot.dropna(), f"label:{anno_label}"]
                .mode(dropna=False)
                .iloc[0]
            )

        X.append(x)
        Y.append(y)
        T.append(t)

    X = np.stack(X)
    Y = np.stack(Y)
    T = np.stack(T)

    if resample_rate != sample_rate:
        X = resize(X, int(resample_rate * winsec))

    return X, Y, T


def is_good_window(x, sample_rate, winsec):
    """
    Check if a window is considered good based on its length and the presence of NaN values.

    Args:
        x (ndarray): Window data.
        sample_rate (int): Sampling rate of the data.
        winsec (int): Length of the window in seconds.

    Returns:
        bool: True if the window is considered good, False otherwise.

    """
    # Check window length is correct
    window_len = sample_rate * winsec
    if len(x) != window_len:
        return False

    # Check no nans
    if np.isnan(x).any():
        return False

    return True


def load_all_and_make_windows(
    datafiles,
    annofile,
    out_dir=None,
    anno_label="Walmsley2020",
    sample_rate=100,
    winsec=30,
    resample_rate=30,
    n_jobs=1,
):
    """
    Load data from multiple files, create windows, and save the results.

    Args:
        datafiles (list): List of data file paths.
        annofile (str): Path to the annotation file.
        out_dir (str | None): Output directory to save the results. Outputs not saved if None.
        n_jobs (int): Number of parallel jobs.

    Returns:
        tuple: Tuple containing the windowed data, labels, timestamps, and participant IDs.

    """

    def worker(datafile):
        X, Y, T = make_windows(
            load_data(datafile, sample_rate),
            anno_dict,
            anno_label,
            winsec,
            sample_rate,
            resample_rate,
        )
        pid = os.path.basename(datafile).split(".")[
            0
        ]  # participant ID based on file name
        pid = np.asarray([pid] * len(X))
        return X, Y, T, pid

    anno_dict = pd.read_csv(annofile, index_col="annotation", dtype="string")

    X, Y, T, P = zip(
        *Parallel(n_jobs=n_jobs)(
            delayed(worker)(datafile)
            for datafile in tqdm(datafiles, desc="Load and making windows: ")
        )
    )

    X = np.vstack(X)
    Y = np.hstack(Y)
    T = np.hstack(T)
    P = np.hstack(P)

    if out_dir:
        # Save arrays for future use
        os.makedirs(out_dir, exist_ok=True)
        np.save(f"{out_dir}/X.npy", X)
        np.save(f"{out_dir}/Y.npy", Y)
        np.save(f"{out_dir}/T.npy", T)
        np.save(f"{out_dir}/pid.npy", P)

    return X, Y, T, P
