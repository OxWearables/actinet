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
import sys

import actipy

from actinet import __version__
from actinet import __classifier_version__
from actinet import __classifier_md5__
from actinet.accPlot import plotTimeSeries
from actinet.models import ActivityClassifier
from actinet.summarisation import getActivitySummary
from actinet.utils.utils import infer_freq, drop_first_last_days, flag_wear_below_days, calculate_wear_stats

BASE_URL = "https://wearables-files.ndph.ox.ac.uk/files/actinet/models/"


def main():

    parser = argparse.ArgumentParser(
        description="A tool to predict activities from accelerometer data using a self-supervised Resnet 18 model",
        add_help=True,
    )
    parser.add_argument(
        "filepath", nargs="?", default="", help="Enter file to be processed"
    )
    parser.add_argument(
        "--outdir",
        "-o",
        help="Enter folder location to save output files",
        default="outputs/",
    )
    parser.add_argument(
        "--classifier-path",
        "-c",
        help="Enter custom acitivty classifier file to use",
        default=None,
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download of classifier file",
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
        "--exclude-first-last", 
        "-e",
        help="Exclude first, last or both days of data. Default: None (no exclusion)",
        type=str, 
        choices=['first', 'last', 'both'], 
        default=None
    ) 
    parser.add_argument(
        "--exclude-wear-below", 
        "-w",
        help="Exclude days with wear time below threshold. Pass values as strings, e.g.: '12H', '30min'. Default: None (no exclusion)",
        type=str, 
        default=None)
    parser.add_argument(
        "--plot-activity",
        "-p",
        action="store_true",
        help="Plot the predicted activity labels",
    )
    parser.add_argument(
        "--cache-classifier",
        action="store_true",
        help="Download and cache classifier file and model modules for offline usage",
    )
    parser.add_argument(
        "--model-repo-path", "-m", help="Enter repository of ssl model", default=None
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = parser.parse_args()

    before = time.time()

    verbose = not args.quiet

    classifier_path = (
        pathlib.Path(__file__).parent / f"{__classifier_version__}.joblib.lzma"
    )

    if args.cache_classifier:
        load_classifier(
            classifier_path=classifier_path,
            model_repo_path=None,
            check_md5=True,
            force_download=True,
            verbose=verbose,
        )

        after = time.time()
        print(f"Done! ({round(after - before,2)}s)")

        return

    else:
        if not args.filepath:
            raise ValueError("Please provide a file to process.")

    # Info contains high-level summary of the data and results
    info = {}
    info['ActiNetVersion'] = __version__
    info['ActiNetArgs'] = vars(args)

    # Load file
    data, info_read = read(
        args.filepath,
        resample_hz=None,
        sample_rate=args.sample_rate,
        verbose=verbose,
    )
    info.update(info_read)

    # Exclusion: first/last days
    if args.exclude_first_last is not None:
        data = drop_first_last_days(data, args.exclude_first_last)
  
    # Exclusion: days with wear time below threshold
    if args.exclude_wear_below is not None:
        data = flag_wear_below_days(data, args.exclude_wear_below)
   
    # Update wear time stats after exclusions
    info.update(calculate_wear_stats(data))

    # Output paths
    basename = resolve_path(args.filepath)[1]
    outdir = os.path.join(args.outdir, basename)
    outputSummaryFile = f"{outdir}/{basename}-outputSummary.json"
    os.makedirs(outdir, exist_ok=True)

     # If no data, save info and exit
    if len(data) == 0 or data.isna().any(axis=1).all():  # TODO: check na only on x,y,z cols?
        # Save info as outputSummary.json
        with open(outputSummaryFile, "w") as f:
            json.dump(info, f, indent=4, cls=NpEncoder)

        # Print
        if verbose:
            print("\nSummary Stats\n---------------------")
            print(
                json.dumps(
                    {
                        key: info[key]
                        for key in [
                            "Filename",
                            "Filesize(MB)",
                            "WearTime(days)",
                            "NonwearTime(days)",
                            "ReadOK"
                        ]
                    },
                    indent=4,
                    cls=NpEncoder,
                )
            )
        print("No data to process. Exiting early...")
        sys.exit(0)

    # Run classifier
    if verbose:
        print("Loading classifier...")

    check_md5 = args.classifier_path is None
    classifier: ActivityClassifier = load_classifier(
        args.classifier_path or classifier_path,
        args.model_repo_path,
        check_md5,
        args.force_download,
        verbose,
    )

    classifier.verbose = verbose
    classifier.device = args.pytorch_device

    if verbose:
        print("Running activity classifier...")
    Y = classifier.predict_from_frame(data, args.sample_rate)

    # Save predicted activities
    timeSeriesFile = f"{outdir}/{basename}-timeSeries.csv.gz"
    Y.to_csv(timeSeriesFile)

    if verbose:
        print("Time series output written to:", timeSeriesFile)

    # Plot activity profile
    if args.plot_activity:
        plotFile = f"{outdir}/{basename}-timeSeries-plot.png"
        fig = plotTimeSeries(Y)
        fig.savefig(plotFile, dpi=200, bbox_inches="tight")

        if verbose:
            print("Output plot written to:", plotFile)

    # Summary
    summary = getActivitySummary(Y, list(classifier.labels), True, True, verbose)

    # Join the actipy processing info, with acitivity summary data
    outputSummary = {**summary, **info}

    # Save output summary
    with open(outputSummaryFile, "w") as f:
        json.dump(outputSummary, f, indent=4, cls=NpEncoder)

    if verbose:
        print("Output summary written to:", outputSummaryFile)

    # Print
    if verbose:
        print("\nSummary Stats\n---------------------")
        print(
            json.dumps(
                {
                    key: outputSummary[key]
                    for key in [
                        "Filename",
                        "Filesize(MB)",
                        "WearTime(days)",
                        "NonwearTime(days)",
                        "ReadOK",
                    ]
                    + [
                        f"{label}-overall-avg"
                        for label in ["acc"] + list(classifier.labels)
                    ]
                },
                indent=4,
                cls=NpEncoder,
            )
        )

    after = time.time()
    print(f"Done! ({round(after - before,2)}s)")


def read(
    filepath, resample_hz="uniform", sample_rate=None, lowpass_hz=None, verbose=True
):

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
            lowpass_hz=lowpass_hz,
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
            lowpass_hz=lowpass_hz,
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


def load_classifier(
    classifier_path,
    model_repo_path=None,
    check_md5=True,
    force_download=False,
    verbose=True,
):
    """Load trained classifier. Download if not exists."""

    pth = pathlib.Path(classifier_path)

    if force_download or not pth.exists():

        url = f"{BASE_URL}{__classifier_version__}.joblib.lzma"

        if verbose:
            print(f"Downloading {url}...")

        with urllib.request.urlopen(url) as f_src, open(pth, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)

    if check_md5:
        assert md5(pth) == __classifier_md5__, (
            "Classifier file is corrupted. Please run with --force-download "
            "to download the model file again."
        )

    classifier: ActivityClassifier = joblib.load(pth)

    if model_repo_path and pathlib.Path(model_repo_path).exists() and verbose:
        print(f"Loading model repository from {model_repo_path}.")

    classifier.load_model(model_repo_path)

    return classifier


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
        if pd.isnull(obj):  # handles pandas NAType
            return np.nan
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    main()
