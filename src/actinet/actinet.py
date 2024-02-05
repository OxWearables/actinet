import os
import pathlib
import urllib
import shutil
import time
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import joblib

import actipy

from actinet import __model_version__
from actinet import __model_md5__
from actinet.accPlot import plotTimeSeries
from actinet.models import ActivityClassifier
from actinet.sslmodel import SAMPLE_RATE
from actinet.summarisation import getActivitySummary, ACTIVITY_LABELS
from actinet.utils.utils import infer_freq

BASE_URL = "https://zenodo.org/records/10616280/files/"


def main():

    parser = argparse.ArgumentParser(
        description="A tool to predict activities from accelerometer data using a self-supervised Resnet 18 model",
        add_help=True,
    )
    parser.add_argument("filepath", help="Enter file to be processed")
    parser.add_argument(
        "--outdir",
        "-o",
        help="Enter folder location to save output files",
        default="outputs/",
    )
    parser.add_argument(
        "--model-path", "-m", help="Enter custom model file to use", default=None
    )
    parser.add_argument(
        "--force-download", action="store_true", help="Force download of model file"
    )
    parser.add_argument(
        "--pytorch-device",
        "-d",
        help="Pytorch device to use, e.g.: 'cpu' or 'cuda:0'",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--sample-rate",
        "-r",
        help="Sample rate for measurement, otherwise inferred.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--plot-activity",
        "-p",
        action="store_true",
        help="Plot the predicted activity labels",
    )
    parser.add_argument(
        "--cache-ssl",
        action="store_true",
        help="Download and cache ssl module for offline usage",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = parser.parse_args()

    before = time.time()

    verbose = not args.quiet

    if args.cache_ssl:
        model = ActivityClassifier(weights_path=True, ssl_repo=None, verbose=verbose)

        after = time.time()
        print(f"Done! ({round(after - before,2)}s)")

        return

    # Load file
    data, info = read(
        args.filepath,
        resample_hz=SAMPLE_RATE,
        sample_rate=args.sample_rate,
        verbose=verbose,
    )

    # Output paths
    basename = resolve_path(args.filepath)[1]
    outdir = os.path.join(args.outdir, basename)
    os.makedirs(outdir, exist_ok=True)

    # Run model
    if verbose:
        print("Loading model...")
    model_path = pathlib.Path(__file__).parent / f"{__model_version__}.joblib.lzma"
    check_md5 = args.model_path is None
    model: ActivityClassifier = load_model(
        args.model_path or model_path, check_md5, args.force_download
    )

    model.verbose = verbose
    model.device = args.pytorch_device

    if verbose:
        print("Running activity classifier...")
    Y = model.predict_from_frame(data)

    # Save predicted activities
    timeSeriesFile = f"{outdir}/{basename}-timeSeries.csv.gz"
    Y.to_csv(timeSeriesFile)

    if verbose:
        print("Time series output written to:", timeSeriesFile)

    # Summary
    summary = getActivitySummary(Y, True, True, verbose)

    # Join the actipy processing info, with acitivity summary data
    outputSummary = {**summary, **info}

    # Save output summary
    outputSummaryFile = f"{outdir}/{basename}-outputSummary.json"
    with open(outputSummaryFile, "w") as f:
        json.dump(outputSummary, f, indent=4, cls=NpEncoder)

    if verbose:
        print("Output summary written to:", outputSummaryFile)

    # Print
    if verbose:
        print("\nSummary Stats\n---------------------")
        print(
            {
                key: outputSummary[key]
                for key in [
                    "Filename",
                    "Filesize(MB)",
                    "WearTime(days)",
                    "NonwearTime(days)",
                    "ReadOK",
                    "acc-overall-avg(mg)",
                ]
                + [f"{label}-week-avg" for label in ACTIVITY_LABELS]
            }
        )

    # Plot activity profile
    if args.plot_activity:
        plotFile = f"{outdir}/{basename}-timeSeries-plot.png"
        fig = plotTimeSeries(Y)
        fig.savefig(plotFile, dpi=200, bbox_inches="tight")

        if verbose:
            print("Output plot written to:", plotFile)

    after = time.time()
    print(f"Done! ({round(after - before,2)}s)")


def read(filepath, resample_hz="uniform", sample_rate=None, verbose=True):

    p = pathlib.Path(filepath)
    ftype = p.suffixes[0].lower()
    fsize = round(p.stat().st_size / (1024 * 1024), 1)

    if ftype in (".csv", ".pkl"):

        if ftype == ".csv":
            data = pd.read_csv(
                filepath,
                usecols=["time", "x", "y", "z"],
                parse_dates=["time"],
                index_col="time",
                dtype={"x": "f4", "y": "f4", "z": "f4"},
            )
        elif ftype == ".pkl":
            data = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unknown file format: {ftype}")

        if sample_rate in (None, False):
            freq = infer_freq(data.index)
            sample_rate = int(np.round(pd.Timedelta("1s") / freq))

        # Quick fix: Drop duplicate indices. TODO: Maybe should be handled by actipy.
        data = data[~data.index.duplicated(keep="first")]

        data, info = actipy.process(
            data,
            sample_rate,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=resample_hz,
            verbose=verbose,
        )

        info = {
            **{
                "Filename": filepath,
                "Device": ftype,
                "Filesize(MB)": fsize,
                "SampleRate": sample_rate,
            },
            **info,
        }

    elif ftype in (".cwa", ".gt3x", ".bin"):

        data, info = actipy.read_device(
            filepath,
            lowpass_hz=None,
            calibrate_gravity=True,
            detect_nonwear=True,
            resample_hz=resample_hz,
            verbose=verbose,
        )

    if "ResampleRate" not in info:
        info["ResampleRate"] = info["SampleRate"]

    return data, info


def resolve_path(path):
    """Return parent folder, file name and file extension"""
    p = pathlib.Path(path)
    extension = p.suffixes[0]
    filename = p.name.rsplit(extension)[0]
    dirname = p.parent
    return dirname, filename, extension


def load_model(model_path, check_md5=True, force_download=False):
    """Load trained model. Download if not exists."""

    pth = pathlib.Path(model_path)

    if force_download or not pth.exists():

        url = f"{BASE_URL}{__model_version__}.joblib.lzma"

        print(f"Downloading {url}...")

        with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)

    if check_md5:
        assert md5(pth) == __model_md5__, (
            "Model file is corrupted. Please run with --force-download "
            "to download the model file again."
        )

    return joblib.load(pth)


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    main()
