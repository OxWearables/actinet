import json
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import warnings
import actipy
from glob import glob
import subprocess

from actinet.utils.utils import is_good_window, resize

ACC_COLS = ["x", "y", "z"]


def load_data(
    datafile,
    sample_rate=100,
    annot_type=str,
    lowpass_hz=None,
    resample_rate=None,
):
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

    data, _ = actipy.process(
        data,
        sample_rate,
        verbose=False,
        lowpass_hz=lowpass_hz,
        resample_hz=resample_rate,
    )
    return data


def make_windows(
    data,
    anno_dict,
    anno_label,
    winsec=30,
    sample_rate=100,
    resample_rate=30,
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

        x = w[ACC_COLS]

        annot = w["annotation"]

        if pd.isna(annot).all():  # skip if annotation is NA
            continue

        if not is_good_window(x, sample_rate * winsec, ACC_COLS):  # skip if bad window
            continue

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unable to sort modes")
            y = (
                anno_dict.loc[annot.dropna(), f"label:{anno_label}"]
                .mode(dropna=False)
                .iloc[0]
            )

        X.append(x.to_numpy())
        Y.append(y)
        T.append(t)

    X = np.stack(X)
    Y = np.stack(Y)
    T = np.stack(T)

    if resample_rate != sample_rate:
        X = resize(X, int(resample_rate * winsec))

    return X, Y, T


def load_all_and_make_windows(
    datafiles,
    annofile,
    out_dir=None,
    anno_label="Walmsley2020",
    sample_rate=100,
    winsec=30,
    resample_rate=30,
    lowpass_hz=None,
    downsampling_method="nn",
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
        if downsampling_method == "nn":
            X, Y, T = make_windows(
                load_data(
                    datafile,
                    sample_rate,
                    resample_rate=resample_rate,
                    lowpass_hz=lowpass_hz,
                ),
                anno_dict,
                anno_label,
                winsec,
                resample_rate,
                resample_rate,
            )
        elif downsampling_method == "linear":
            X, Y, T = make_windows(
                load_data(
                    datafile,
                    sample_rate,
                    lowpass_hz=lowpass_hz,
                ),
                anno_dict,
                anno_label,
                winsec,
                sample_rate,
                resample_rate,
            )
        else:
            raise ValueError("Invalid downsampling method")

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
        out_path = os.path.join(
            out_dir, f"prepared/downsampling_{downsampling_method}_lowpass_{lowpass_hz}"
        )

        info = {
            "sample_rate": sample_rate,
            "winsec": winsec,
            "resample_rate": resample_rate,
            "lowpass_hz": lowpass_hz,
            "downsampling_method": downsampling_method,
            "anno_label": anno_label,
        }

        # Save arrays for future use
        os.makedirs(out_path, exist_ok=True)
        np.save(f"{out_path}/X.npy", X)
        np.save(f"{out_path}/Y.npy", Y)
        np.save(f"{out_path}/T.npy", T)
        np.save(f"{out_path}/pid.npy", P)

        with open(f"{out_path}/info.json", "w") as f:
            json.dump(info, f, indent=4)

    return X, Y, T, P


def make_labels(
    data,
    anno_dict,
    anno_label,
    sample_rate=100,
    winsec=30,
):
    Y, T = [], []
    for t, w in data.resample(f"{winsec}s", origin="start"):
        if len(w) < 1:
            continue

        t = t.to_numpy()

        annot = w["annotation"]

        if pd.isna(annot).all():  # skip if annotation is NA
            y = np.nan

        elif not is_good_window(
            w, sample_rate * winsec, ACC_COLS
        ):  # skip if bad window
            y = np.nan

        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Unable to sort modes")
                y = (
                    anno_dict.loc[annot.dropna(), f"label:{anno_label}"]
                    .mode(dropna=False)
                    .iloc[0]
                )

        Y.append(y)
        T.append(t)

    Y = np.stack(Y)
    T = np.stack(T)

    return Y, T


def extract_accelerometer_features(n_jobs):
    def process_file(file_number):
        filename = f"P{file_number:03}.csv.gz"

        if len(glob(f"data/capture24/bbaa/P{file_number:03}*")) != 4:
            command = f'accProcess data/capture24/{filename} --csvTimeFormat "yyyy-MM-dd HH:mm:ss.SSSSSS" --csvTimeXYZTempColsIndex "0,1,2,3" -o "data/capture24/bbaa" --activityClassification False --deleteIntermediateFiles False'
            process = subprocess.run(command, shell=True, capture_output=True)

            return process

    Parallel(n_jobs=n_jobs)(
        delayed(process_file)(file_number) for file_number in tqdm(range(1, 152))
    )


def prepare_participant_accelerometer_data(pid, annotations_file, verbose=False):
    raw_file_name = f"data/capture24/P{pid:03}.csv.gz"
    features_file_name = f"data/capture24/bbaa/P{pid:03}-epoch.csv.gz"

    raw_file = pd.read_csv(
        raw_file_name,
        index_col="time",
        parse_dates=True,
        dtype={"x": "f4", "y": "f4", "z": "f4", "annotation": str},
    )
    features_file = pd.read_csv(features_file_name, index_col="time")

    if verbose:
        print(
            f"First accelerometer timestamp: {features_file.index[0]}\n"
            + f"First actipy timestamp: {raw_file.index[0]}\n"
            + f"Last accelerometer timestamp: {features_file.index[-1]}\n"
            + f"Last actipy timestamp: {raw_file.index[-1]}\n"
        )

    Y, T = make_labels(
        raw_file,
        pd.read_csv(annotations_file, index_col="annotation", dtype="string"),
        "Walmsley2020",
    )

    features = (
        ["enmoTrunc", "enmoAbs", "xMean", "yMean", "zMean", "xRange"]
        + ["yRange", "zRange", "xStd", "yStd", "zStd", "xyCov", "xzCov", "yzCov"]
        + ["mean", "sd", "coefvariation", "median", "min", "max", "25thp", "75thp"]
        + ["autocorr", "corrxy", "corrxz", "corryz", "avgroll", "avgpitch"]
        + ["avgyaw", "sdroll", "sdpitch", "sdyaw", "rollg", "pitchg", "yawg"]
        + ["fmax", "pmax", "fmaxband", "pmaxband", "entropy", "fft1", "fft2"]
        + ["fft3", "fft4", "fft5", "fft6", "fft7", "fft8", "fft9", "fft10", "MAD"]
        + ["MPD", "skew", "kurt", "avgArmAngel", "avgArmAngelAbsDiff", "f1", "p1"]
        + ["f2", "p2", "f625", "p625", "totalPower"]
    )

    X = features_file[features].to_numpy()
    Y, T = Y[: len(X)], T[: len(X)]
    P = np.array([f"P{pid:03}"] * len(X))

    mask = ~(pd.isna(Y) | (Y == "nan"))
    X, Y, T, P = X[mask], Y[mask], T[mask], P[mask]

    return X, Y, T, P


def prepare_accelerometer_data(annotation_file, out_dir, n_jobs):
    X, Y, T, P = zip(
        *Parallel(n_jobs=n_jobs)(
            delayed(prepare_participant_accelerometer_data)(
                file_number, annotation_file
            )
            for file_number in tqdm(range(1, 152))
        )
    )

    X = np.vstack(X)
    Y = np.hstack(Y)
    T = np.hstack(T)
    P = np.hstack(P)

    if out_dir:
        out_path = os.path.join(out_dir, f"prepared/accelerometer")
        # Save arrays for future use
        os.makedirs(out_path, exist_ok=True)
        np.save(f"{out_path}/X.npy", X)
        np.save(f"{out_path}/Y.npy", Y)
        np.save(f"{out_path}/T.npy", T)
        np.save(f"{out_path}/pid.npy", P)

    return X, Y, T, P
