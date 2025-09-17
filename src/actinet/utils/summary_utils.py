import pandas as pd
from pandas.tseries.frequencies import to_offset
import numpy as np
from actinet.utils.utils import infer_freq


def impute_missing(data, extrapolate=True):
    """Impute missing/nonwear segments

    Impute non-wear data segments using the average of similar time-of-day values
    with one minute granularity on different days of the measurement. This
    imputation accounts for potential wear time diurnal bias where, for example,
    if the device was systematically less worn during sleep in an individual,
    the crude average vector magnitude during wear time would be a biased
    overestimate of the true average. See
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169649#sec013

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param bool verbose: Print verbose output

    :return: Update DataFrame <e> columns nan values with time-of-day imputation
    :rtype: void
    """

    if extrapolate:
        # padding at the boundaries to have full 24h
        data = data.reindex(
            pd.date_range(
                data.index[0].floor("D"),
                data.index[-1].ceil("D"),
                freq=to_offset(pd.infer_freq(data.index)),
                inclusive="left",
                name="time",
            ),
            method="nearest",
            tolerance=pd.Timedelta("1m"),
            limit=1,
        )

    def fillna(subframe):
        # Transform will first pass the subframe column-by-column as a Series.
        # After passing all columns, it will pass the entire subframe again as a DataFrame.
        # Processing the entire subframe is optional (return value can be omitted). See 'Notes' in transform doc.
        if isinstance(subframe, pd.Series):
            x = subframe.to_numpy()
            nan = np.isnan(x)
            nanlen = len(x[nan])
            if 0 < nanlen < len(x):  # check x contains a NaN and is not all NaN
                x[nan] = np.nanmean(x)
                return x  # will be cast back to a Series automatically
            else:
                return subframe

    data = (
        data
        # first attempt imputation using same day of week
        .groupby([data.index.weekday, data.index.hour, data.index.minute])
        .transform(fillna)
        # then try within weekday/weekend
        .groupby([data.index.weekday >= 5, data.index.hour, data.index.minute])
        .transform(fillna)
        # finally, use all other days
        .groupby([data.index.hour, data.index.minute])
        .transform(fillna)
    )

    return data


def calculateECDF(x, summary):
    """Calculate activity intensity empirical cumulative distribution

    The input data must not be imputed, as ECDF requires different imputation
    where nan/non-wear data segments are IMPUTED FOR EACH INTENSITY LEVEL. Here,
    the average of similar time-of-day values is imputed with one minute
    granularity on different days of the measurement. Following intensity levels
    are calculated:
    1mg bins from 1-20mg
    5mg bins from 25-100mg
    25mg bins from 125-500mg
    100mg bins from 500-2000mg

    :param pandas.DataFrame e: Pandas dataframe of epoch data
    :param str inputCol: Column to calculate intensity distribution on
    :param dict summary: Output dictionary containing all summary metrics

    :return: Updated summary file
    :rtype: dict
    """

    levels = np.concatenate(
        [
            np.linspace(1, 20, 20),  # 1mg bins from 1-20mg
            np.linspace(25, 100, 16),  # 5mg bins from 25-100mg
            np.linspace(125, 500, 16),  # 25mg bins from 125-500mg
            np.linspace(600, 2000, 15),  # 100mg bins from 500-2000mg
        ]
    ).astype("int")

    whrnan = x.isna().to_numpy()
    ecdf = x.to_numpy().reshape(-1, 1) <= levels.reshape(1, -1)
    ecdf[whrnan] = np.nan

    ecdf = (
        pd.DataFrame(ecdf, index=x.index, columns=levels)
        .groupby([x.index.hour, x.index.minute])
        .mean()  # first average is across same time of day
        .mean()  # second average is within each level
    )

    # Write to summary
    for level, val in ecdf.items():
        summary[f"{x.name}-ecdf-{level}mg"] = val

    return summary


def summarize_daily_enmo(
    acc: pd.Series,
    acc_adjusted: pd.Series,
    min_wear_per_day: float = 21 * 60
):
    """
    Summarize daily ENMO information from raw accelerometer data.

    Parameters:
    - data (pd.Series): A pandas Series of ENMO values for each window.
    - data_adjusted (pd.Series): A pandas Series of ENMO values for each window, with missingness imputed.
    - min_wear_per_day (float, optional): The minimum required wear time (in minutes) for a day to be considered valid. Defaults to 21 hours.

    Returns:
    - pd.DataFrame: A DataFrame with dates as index and daily wear statistics as columns:
        - 'ENMO(mg)': Total ENMO in mg
        - 'ENMO Adjusted(mg)': Total ENMO in mg, adjusted for non-wear

    Example:
        summary = summarize_daily_enmo(acc, acc_adjusted, min_wear_per_day=21*60)
    """

    def _is_enough(x, min_wear=None, dt=None):
        if min_wear is None:
            return True  # no minimum wear time, then default to True
        if dt is None:
            dt = infer_freq(x.index).total_seconds()
        return x.notna().sum() * dt / 60 > min_wear

    def _mean(x, min_wear=None, dt=None):
        if not _is_enough(x, min_wear, dt):
            return np.nan
        return x.mean()

    dt = infer_freq(acc.index).total_seconds()

    if infer_freq(acc_adjusted.index) != infer_freq(acc.index):
        raise ValueError("Data and Data_adjusted must have the same frequency")

    daily = acc.resample('D').agg(_mean, min_wear=min_wear_per_day, dt=dt).rename('ENMO(mg)')
    daily_adj = acc_adjusted.resample('D').agg(_mean, min_wear=min_wear_per_day, dt=dt).rename('ENMO Adjusted(mg)')
    
    summary = pd.concat([daily, daily_adj], axis=1)

    return summary


def summarize_daily_activity(data: pd.DataFrame, data_adjusted: pd.DataFrame, 
                             labels: list, min_wear_per_day: float = 21 * 60):
    """
    Summarize daily activity information from predicted label outputs.

    Parameters:
    - data (pd.DataFrame): A pandas DataFrame of predicted activity labels with a DatetimeIndex.
    - data_adjusted (pd.DataFrame): A pandas DataFrame of predicted activity labels with a DatetimeIndex, with missingness imputed.
    - labels (list): List of activity state labels (e.g., ['sleep', 'sedentary', 'light', 'moderate-vigorous']).
    - min_wear_per_day (float, optional): The minimum required wear time (in minutes) for a day to be considered valid. Defaults to 21 hours.

    Returns:
    - pd.DataFrame: A DataFrame with dates as index and daily activity statistics as columns:
        - 'Sleep (hours)': Total sleep duration in hours
        - 'Sedentary (hours)': Sedentary activity duration in hours
        - 'Light (hours)': Light activity duration in hours
        - 'MVPA (hours)': Moderate activity duration in hours
        - 'Sleep Adjusted (hours)': Total sleep duration in hours, adjusted for non-wear
        - 'Sedentary Adjusted (hours)': Sedentary activity duration in hours, adjusted for non-wear
        - 'Light Adjusted (hours)': Light activity duration in hours, adjusted for non-wear
        - 'MVPA Adjusted (hours)': Moderate activity duration in hours, adjusted for non-wear
    """
    if infer_freq(data_adjusted.index) != infer_freq(data.index):
        raise ValueError("Data and Data_adjusted must have the same frequency")
    
    dt = infer_freq(data.index).total_seconds()
    
    def _is_enough(x, min_wear=None, dt=None):
        if min_wear is None:
            return True  # no minimum wear time, then default to True
        if dt is None:
            dt = infer_freq(x.index).total_seconds()
        return x.notna().sum() * dt / 60 > min_wear

    def _total_hrs(x, min_wear=None, dt=None):
        if not _is_enough(x, min_wear, dt):
            return np.nan
        return x.sum() * dt / 3600

    daily = data.resample('D')\
                .agg(_total_hrs, min_wear=min_wear_per_day, dt=dt)\
                .rename(columns={label: f"{label.capitalize()}(hours)" for label in labels})
    daily_adj = data_adjusted.resample('D')\
                             .agg(_total_hrs, min_wear=min_wear_per_day, dt=dt)\
                             .rename(columns={label: f"{label.capitalize()} Adjusted(hours)"
                                                for label in labels})

    summary = pd.concat([daily, daily_adj], axis=1)

    return summary


def calculate_daily_wear_stats(data: pd.DataFrame):
    """
    Calculate daily wear time statistics from raw accelerometer data.

    Parameters:
    - data (pd.DataFrame): A pandas DataFrame of raw accelerometer data with columns 'x', 'y', 'z' and a DatetimeIndex.

    Returns:
    - pd.DataFrame: A DataFrame with dates as index and daily wear statistics as columns:
        - 'WearTime(hours)': Total wear time in hours

    Example:
        daily_wear_stats = calculate_daily_wear_stats(data)
    """

    if len(data) == 0:
        return pd.DataFrame()

    # Identify non-wear periods (NaN in any of x,y,z columns)
    na = data.isna().any(axis=1)
    dt = infer_freq(data.index).total_seconds()

    # Group by date
    date_groups = data.groupby(data.index.date)

    results = []

    for date, day_data in date_groups:
        day_na = na.loc[day_data.index]
        n_samples = len(day_data)

        if n_samples == 0:
            # Skip empty days
            continue

        # Calculate wear time
        nonwear_samples = day_na.sum()
        wear_samples = n_samples - nonwear_samples

        # Convert to hours
        wear_hours = wear_samples * dt / 3600

        results.append({
            'Date': pd.to_datetime(date),
            'WearTime(hours)': round(wear_hours, 2)
        })

    if not results:
        return pd.DataFrame()

    # Create DataFrame and set date as index
    daily_stats = pd.DataFrame(results)
    daily_stats.set_index('Date', inplace=True)
    daily_stats.index.name = 'Date'

    return daily_stats