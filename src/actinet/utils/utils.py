import datetime
import re
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from typing import Union


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


def safe_indexer(array, indexes):
    return array[indexes] if array is not None else None


def is_good_window(x, window_len, columns):
    """
    Check if a window is considered good based on its length, the presence of NaN values in specified columns.

    Args:
        x (pd.DataFrame): Window data.
        window_len (int): The index length of the data.
        columns (list): List of relevant columns to check for NaN values.

    Returns:
        bool: True if the window is considered good, False otherwise.

    """
    if not all(col in x.columns for col in columns):
        return False

    # Check window length is correct
    if len(x[columns]) != window_len:
        return False

    # Check no nans
    if pd.isna(x[columns]).any(axis=None):

        return False

    return True


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


def drop_first_last_days(
    x: Union[pd.Series, pd.DataFrame],
    first_or_last='both'
):
    """
    Drop the first day, last day, or both from a time series.

    Parameters:
    - x (pd.Series or pd.DataFrame): A pandas Series or DataFrame with a DatetimeIndex representing time series data.
    - first_or_last (str, optional): A string indicating which days to drop. Options are 'first', 'last', or 'both'. Default is 'both'.

    Returns:
    - pd.Series or pd.DataFrame: A pandas Series or DataFrame with the values of the specified days dropped.

    Example:
        # Drop the first day from the series
        series = drop_first_last_days(series, first_or_last='first')
    """
    if len(x) == 0:
        print("No data to drop")
        return x

    if first_or_last == 'first':
        x = x[x.index.date != x.index.date[0]]
    elif first_or_last == 'last':
        x = x[x.index.date != x.index.date[-1]]
    elif first_or_last == 'both':
        x = x[(x.index.date != x.index.date[0]) & (x.index.date != x.index.date[-1])]
    return x


def flag_wear_below_days(
    x: Union[pd.Series, pd.DataFrame],
    min_wear: str = '12H'
):
    """
    Set days containing less than the specified minimum wear time (`min_wear`) to NaN.

    Parameters:
    - x (pd.Series or pd.DataFrame): A pandas Series or DataFrame with a DatetimeIndex representing time series data.
    - min_wear (str): A string representing the minimum wear time required per day (e.g., '8H' for 8 hours).

    Returns:
    - pd.Series or pd.DataFrame: A pandas Series or DataFrame with days having less than `min_wear` of valid data set to NaN.

    Example:
        # Exclude days with less than 12 hours of valid data
        series = exclude_wear_below_days(series, min_wear='12H')
    """
    if len(x) == 0:
        print("No data to exclude")
        return x

    min_wear = pd.Timedelta(min_wear)
    dt = infer_freq(x.index)
    ok = x.notna()
    if isinstance(ok, pd.DataFrame):
        ok = ok.all(axis=1)
    ok = (
        ok
        .groupby(x.index.date)
        .sum() * dt
        >= min_wear
    )
    # keep ok days, rest is set to NaN
    x = x.copy()  # make a copy to avoid modifying the original data
    x[np.isin(x.index.date, ok[~ok].index)] = np.nan
    return x


def calculate_wear_stats(data: pd.DataFrame):
    """
    Calculate wear time and related information from raw accelerometer data.

    Parameters:
    - data (pd.DataFrame): A pandas DataFrame of raw accelerometer data with columns 'x', 'y', 'z' and a DatetimeIndex.

    Returns:
    - dict: A dictionary containing various wear time stats.

    Example:
        info = calculate_wear_stats(data)
    """

    TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

    n_data = len(data)

    if n_data == 0:
        start_time = None
        end_time = None
        wear_start_time = None
        wear_end_time = None
        nonwear_duration = 0.0
        wear_duration = 0.0
        covers24hok = 0

    else:
        na = data.isna().any(axis=1)  # TODO: check na only on x,y,z cols?
        dt = infer_freq(data.index).total_seconds()
        start_time = data.index[0].strftime(TIME_FORMAT)
        end_time = data.index[-1].strftime(TIME_FORMAT)
        wear_start_time = data.first_valid_index()
        if wear_start_time is not None:
            wear_start_time = wear_start_time.strftime(TIME_FORMAT)
        wear_end_time = data.last_valid_index()
        if wear_end_time is not None:
            wear_end_time = wear_end_time.strftime(TIME_FORMAT)
        nonwear_duration = na.sum() * dt / (60 * 60 * 24)
        wear_duration = n_data * dt / (60 * 60 * 24) - nonwear_duration 
        coverage = (~na).groupby(na.index.hour).mean()
        covers24hok = int(len(coverage) == 24 and coverage.min() >= 0.01)

    return {
        'StartTime': start_time,
        'EndTime': end_time,
        'WearStartTime': wear_start_time,
        'WearEndTime': wear_end_time,
        'WearTime(days)': wear_duration,
        'NonwearTime(days)': nonwear_duration,
        'Covers24hOK': covers24hok
    }
