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
from actinet import __classifiers__
from actinet.accPlot import plotTimeSeries
from actinet.models import ActivityClassifier
from actinet.summarisation import get_activity_summary
from actinet.utils.summary_utils import calculate_daily_wear_stats
from actinet.utils.utils import infer_freq, drop_first_last_days, flag_wear_below_days, calculate_wear_stats

BASE_URL = "https://wearables-files.ndph.ox.ac.uk/files/models/actinet/"


def main():
    parser = argparse.ArgumentParser(
        description="A tool to predict activities from accelerometer data using a self-supervised ResNet-18 model",
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
        "--classifier",
        "-c",
        help="Enter custom activity classifier file to use. Default: walmsley (Walmsley2020 annotations of activity intensity).",
        default="walmsley",
    )
    parser.add_argument(
        "--no-hmm",
        action="store_true",
        help="Disable HMM post-processing",
    )
    parser.add_argument(
        "--require-sleep-above",
        help="Require sleep blocks to exceed a minimum duration, otherwise be classified as sedentary. Pass values as strings, e.g.: '2H', '30min'. Default: None (no requirement)",
        type=str, 
        default=None
    )
    parser.add_argument(
        "--single-sleep-block",
        action="store_true",
        help="Recognise only one sleep block per day, all other sleep blocks will be converted to sedentary",
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
        "--csv-start-row",
        help="Row number to start reading a CSV file. Default: 1 (First row)",
        type=int,
        default=1,
    )
    parser.add_argument("--txyz",
                        help="Use this option to specify the column names for time, x, y, z " +
                             "in the input file, in that order. Use a comma-separated string. " +
                             "Only needed for CSV files, can be ignored for other file types. " +
                             "Default: 'time,x,y,z'",
                        type=str, default="time,x,y,z"
    )
    parser.add_argument('--csv-date-format',
                        default="%Y-%m-%d %H:%M:%S.%f",
                        type=str, 
                        help="Date time format for csv file when reading a csv file. " +
                             "See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes for more possible codes." +
                             "Default: '%%Y-%%m-%%d %%H:%%M:%%S.%%f' (e.g. '2023-10-01 12:34:56.789')"
    )
    parser.add_argument('--calibration-stdtol-min',
                        default=None,
                        type=float,
                        help="Minimum standard deviation tolerance (g) for detecting stationary periods for calibration. Default: None"
    )
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

    if args.cache_classifier:
        load_classifier(
            classifier=args.classifier,
            model_repo_path=None,
            force_download=True,
            verbose=verbose,
        )

        after = time.time()
        print(f"Done! ({round(after - before, 2)}s)")

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
        args.txyz,
        args.csv_start_row-1,  # -1 to convert to zero-based index
        args.csv_date_format,
        args.calibration_stdtol_min,
        resample_hz="uniform",
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

    daily_wear_stats = calculate_daily_wear_stats(data)

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

    classifier: ActivityClassifier = load_classifier(
        args.classifier,
        args.model_repo_path,
        args.force_download,
        verbose
    )

    classifier.verbose = verbose
    classifier.device = args.pytorch_device

    if verbose:
        print("Running activity classifier...")
    Y = classifier.predict_from_frame(data, args.sample_rate, not args.no_hmm,
                                      args.require_sleep_above, args.single_sleep_block)

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
    summary, daily_summary = get_activity_summary(Y, list(classifier.labels), args.exclude_wear_below,
                                                  True, True, verbose)

    # Join the actipy processing info, with acitivity summary data
    outputSummary = {**summary, **info}

    # Save output summary
    with open(outputSummaryFile, "w") as f:
        json.dump(outputSummary, f, indent=4, cls=NpEncoder)

    if verbose:
        print("Output summary written to:", outputSummaryFile)

    daily_summary = pd.concat([daily_wear_stats, daily_summary], axis=1)

    daily_summary.insert(0, 'Filename', info['Filename'])  # add filename for reference
    daily_summary.to_csv(f"{outdir}/{basename}-Daily.csv.gz")

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
    filepath, usecols=None, skipRows=0, dateFormat=None, calibration_stdtol_min=None,
    resample_hz="uniform", sample_rate=None, lowpass_hz=None, verbose=True
):
    p = pathlib.Path(filepath)
    fsize = round(p.stat().st_size / (1024 * 1024), 1)
    
    ftype = p.suffix.lower()
    if ftype in (".gz", ".xz", ".lzma", ".bz2", ".zip"):  # if file is compressed, check the next extension
        ftype = pathlib.Path(p.stem).suffix.lower()

    if ftype in (".csv", ".pkl"):
        if ftype == ".csv":
            tcol, xcol, ycol, zcol = usecols.split(',')
            
            data = pd.read_csv(
                filepath,
                usecols=[tcol, xcol, ycol, zcol],
                parse_dates=[tcol],
                date_format=dateFormat,
                index_col=tcol,
                dtype={xcol: "f4", ycol: "f4", zcol: "f4"},
                skiprows=skipRows,
            )

            # rename to standard names
            data = data.rename(columns={xcol: 'x', ycol: 'y', zcol: 'z'})
            data.index.name = 'time'

        elif ftype == ".pkl":
            data = pd.read_pickle(filepath)

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
            calibrate_gravity_kwargs={'stdtol_min': calibration_stdtol_min},
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
                "ReadOK": 1,
            },
            **info,
        }

    elif ftype in (".cwa", ".gt3x", ".bin"):

        data, info = actipy.read_device(
            filepath,
            lowpass_hz=lowpass_hz,
            calibrate_gravity=True,
            calibrate_gravity_kwargs={'stdtol_min': calibration_stdtol_min},
            detect_nonwear=True,
            resample_hz=resample_hz,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown file format: {ftype}")

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
    classifier,
    model_repo_path=None,
    force_download=False,
    verbose=True,
):
    """Load trained classifier. Download if not exists."""

    if classifier in __classifiers__.keys():
        classifier_version = __classifiers__[classifier]['version']
        classifier_md5 = __classifiers__[classifier]['md5']

    else:
        classifier_version = classifier
        classifier_md5 = None

    classifier_path = pathlib.Path(__file__).parent / f"{classifier_version}.joblib.lzma"

    if force_download or not classifier_path.exists():
        url = f"{BASE_URL}{classifier_version}.joblib.lzma"

        if verbose:
            print(f"Downloading {url}...")

        with urllib.request.urlopen(url) as f_src, open(classifier_path, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)

        if classifier_md5:
            digest = md5(classifier_path)
            if digest != classifier_md5:
                raise ValueError(
                    "Classifier file is corrupted. "
                    "Run again with --force-download to redownload."
                )

    classifier: ActivityClassifier = joblib.load(classifier_path)

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
