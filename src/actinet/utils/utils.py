import datetime
import re
import numpy as np
import pandas as pd
from scipy import stats


def infer_freq(x):
    """Like pd.infer_freq but more forgiving"""
    freq, _ = stats.mode(np.diff(x), keepdims=False)
    freq = pd.Timedelta(freq)
    return freq


def toScreen(msg, verbose=True):
    """
    Print msg str prepended with current time

    :param str mgs: Message to be printed to screen
    :return: None. Prints msg str prepended with current time.
    """
    if verbose:
        timeFormat = "%Y-%m-%d %H:%M:%S"
        print(f"\n{datetime.datetime.now().strftime(timeFormat)}\t{msg}")


def date_parser(t):
    """
    Parse date a date string of the form e.g.
    2020-06-14 19:01:15.123+0100 [Europe/London]
    """
    tz = re.search(r"(?<=\[).+?(?=\])", t)
    if tz is not None:
        tz = tz.group()
    t = re.sub(r"\[(.*?)\]", "", t)
    return pd.to_datetime(t, utc=True).tz_convert(tz)
