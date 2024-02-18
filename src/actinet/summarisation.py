"""Module to generate overall activity summary from epoch data."""

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import scipy.stats as stats

from actinet.utils.utils import date_parser, toScreen
from actinet import circadian


def getActivitySummary(
    data,
    labels,
    intensityDistribution=False,
    circadianMetrics=False,
    verbose=True,
):
    """
    Calculate overall activity summary from predicted activity label data.
    This is achieved by:
    1) calculate imputation values to replace nan PA metric values
    2) calculate empirical cumulative distribution function of vector magnitudes
    3) derive main movement summaries (overall, weekday/weekend, and hour)

    :param str data: Input csv.gz file or pandas dataframe of processed epoch data
    :param list(str) labels: Activity state labels
    :param bool intensityDistribution: Add intensity outputs to dict <summary>
    :param bool circadianMetrics: Add circadian rhythm metrics to dict <summary>
    :param bool verbose: Print verbose output

    :return: A summary of the activity.
    :rtype: dict

    """

    toScreen("=== Summarizing ===", verbose)

    if isinstance(data, str):
        data = pd.read_csv(
            data,
            index_col=["time"],
            parse_dates=["time"],
            date_parser=date_parser,
        )

    # Main movement summaries
    summary = _summarise(
        data,
        labels,
        intensityDistribution,
        circadianMetrics,
        verbose,
    )

    # Return physical activity summary
    return summary


def _summarise(
    data,
    labels,
    intensityDistribution=False,
    circadianMetrics=False,
    verbose=False,
    summary={},
):
    """Overall summary stats for each activity type to summary dict

    :param pandas.DataFrame data: Pandas dataframe of epoch data
    :param list(str) labels: Activity state labels
    :param dict summary: Output dictionary containing all summary metrics
    :param bool intensityDistribution: Add intensity outputs to dict <summary>
    :param bool circadianMetrics: Add circadian rhythm metrics to dict <summary>
    :param bool verbose: Print verbose output
    :param dict summary: Output dictionary containing all summary metrics

    :return: Updated dict <summary> keys for each activity type 'overall-<avg/sd>',
        'week<day/end>-avg', '<day..>-avg', 'hourOfDay-<hr..>-avg',
        'hourOfWeek<day/end>-<hr..>-avg'
    :rtype: dict
    """

    data = data.copy()
    freq = to_offset(infer_freq(data.index))

    # Get start day
    startTime = data.index[0]
    summary["FirstDay(0=mon,6=sun)"] = startTime.weekday()

    # Hours of activity for each recorded day
    epochPeriod = int(pd.Timedelta(freq).total_seconds())
    cols = labels
    dailyStats = (
        data[cols].astype("float").groupby(data.index.date).sum() * epochPeriod / 3600
    ).reset_index(drop=True)

    for i, row in dailyStats.iterrows():
        for col in cols:
            summary[f"day{i}-recorded-{col}(hrs)"] = row.loc[col]

    # Calculate empirical cumulative distribution function of vector magnitudes
    if intensityDistribution:
        summary = calculateECDF(data["acc"], summary)

    # In the following, we resample, pad and impute the data so that we have a
    # multiple of 24h for the stats calculations
    tStart, tEnd = data.index[0], data.index[-1]
    cols = ["acc"] + labels
    if "MET" in data.columns:
        cols.append("MET")
    data = imputeMissing(data[cols].astype("float"))

    # Overall stats (no padding, i.e. only within recording period)
    toScreen("=== Calculating overall statistics ===", verbose)
    overallStats = data[tStart:tEnd].apply(["mean", "std"])
    for col in overallStats:
        summary[f"{col}-overall-avg"] = overallStats[col].loc["mean"]
        summary[f"{col}-overall-sd"] = overallStats[col].loc["std"]

    dayOfWeekStats = data.groupby([data.index.weekday, data.index.hour]).mean()
    dayOfWeekStats.index = dayOfWeekStats.index.set_levels(
        dayOfWeekStats.index.levels[0]
        .to_series()
        .replace({0: "mon", 1: "tue", 2: "wed", 3: "thu", 4: "fri", 5: "sat", 6: "sun"})
        .to_list(),
        level=0,
    )
    dayOfWeekStats.index.set_names(["DayOfWeek", "Hour"], inplace=True)

    # Week stats
    for col, value in dayOfWeekStats.mean().items():
        summary[f"{col}-week-avg"] = value

    # Stats by day of week (Mon, Tue, ...)
    for col, stats in dayOfWeekStats.groupby(level=0).mean().to_dict().items():
        for dayOfWeek, value in stats.items():
            summary[f"{col}-{dayOfWeek}-avg"] = value

    # Stats by hour of day
    for col, stats in dayOfWeekStats.groupby(level=1).mean().to_dict().items():
        for hour, value in stats.items():
            summary[f"{col}-hourOfDay-{hour}-avg"] = value

    weekdayOrWeekendStats = dayOfWeekStats.groupby(
        [
            dayOfWeekStats.index.get_level_values("DayOfWeek").str.contains("sat|sun"),
            dayOfWeekStats.index.get_level_values("Hour"),
        ]
    ).mean()
    weekdayOrWeekendStats.index = weekdayOrWeekendStats.index.set_levels(
        weekdayOrWeekendStats.index.levels[0]
        .to_series()
        .replace({True: "Weekend", False: "Weekday"})
        .to_list(),
        level=0,
    )
    weekdayOrWeekendStats.index.set_names(["WeekdayOrWeekend", "Hour"], inplace=True)

    # Weekday/weekend stats
    for col, stats in weekdayOrWeekendStats.groupby(level=0).mean().to_dict().items():
        for weekdayOrWeekend, value in stats.items():
            summary[f"{col}-{weekdayOrWeekend.lower()}-avg"] = value

    # Stats by hour of day AND by weekday/weekend
    for col, stats in weekdayOrWeekendStats.to_dict().items():
        for key, value in stats.items():
            weekdayOrWeekend, hour = key
            summary[f"{col}-hourOf{weekdayOrWeekend}-{hour}-avg"] = value

    # Calculate circadian metrics
    if circadianMetrics:
        toScreen("=== Calculating circadian metrics ===", verbose)
        summary = circadian.calculatePSD(data, epochPeriod, False, labels, summary)
        summary = circadian.calculatePSD(data, epochPeriod, True, labels, summary)
        summary = circadian.calculateFourierFreq(
            data, epochPeriod, False, labels, summary
        )
        summary = circadian.calculateFourierFreq(
            data, epochPeriod, False, labels, summary
        )
        summary = circadian.calculateM10L5(data, epochPeriod, summary)

    return summary


def imputeMissing(data, extrapolate=True):
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


def infer_freq(x):
    """Like pd.infer_freq but more forgiving"""
    freq, _ = stats.mode(np.diff(x), keepdims=False)
    freq = pd.Timedelta(freq)
    return freq


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
