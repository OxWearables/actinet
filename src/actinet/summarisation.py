"""Module to generate overall activity summary from epoch data."""

import pandas as pd
from pandas.tseries.frequencies import to_offset

from actinet.utils.utils import date_parser, to_screen
from actinet.utils.summary_utils import *
from actinet import circadian


def get_activity_summary(
    data,
    labels,
    exclude_daily_wear_below=None,
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
    4) derive daily summaries (daily enmo and daily activity)

    :param str data: Input csv.gz file or pandas dataframe of processed epoch data
    :param list(str) labels: Activity state labels
    :param str exclude_daily_wear_below: Exclude days with wear time below this threshold
    :param bool intensityDistribution: Add intensity outputs to dict <summary>
    :param bool circadianMetrics: Add circadian rhythm metrics to dict <summary>
    :param bool verbose: Print verbose output

    :return: A summary of the activity and daily wear statistics.
    :rtype: tuple(dict, pd.DataFrame)

    """
    to_screen("=== Summarizing ===", verbose)

    if isinstance(data, str):
        data = pd.read_csv(
            data,
            index_col=["time"],
            parse_dates=["time"],
            date_parser=date_parser,
        )

    # Impute missing values
    data_imputed = _impute_missing(data, labels, verbose)

    # Main movement summaries
    summary = _summarise(
        data,
        data_imputed,
        labels,
        intensityDistribution,
        circadianMetrics,
        verbose,
    )

    # Daily summaries
    daily_summary = _daily_summary(data, data_imputed, labels, exclude_daily_wear_below, verbose)

    # Return physical activity summaries
    return summary, daily_summary


def _impute_missing(data, labels, verbose=False):
    # In the following, we resample, pad and impute the data so that we have a
    # multiple of 24h for the stats calculations
    to_screen("=== Imputing missing values ===", verbose)
    
    cols = ["acc"] + labels
    if "MET" in data.columns:
        cols.append("MET")
    data_imputed = impute_missing(data[cols].astype("float"))

    return data_imputed


def _summarise(
    data,
    data_imputed,
    labels,
    intensityDistribution=False,
    circadianMetrics=False,
    verbose=False,
    summary={},
):
    """Overall summary stats for each activity type to summary dict

    :param pandas.DataFrame data: Pandas dataframe of epoch data
    :param pandas.DataFrame data_adjusted: Pandas dataframe of epoch data with imputed missing values
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

    tStart, tEnd = data.index[0], data.index[-1]

    # Overall stats (no padding, i.e. only within recording period)
    to_screen("=== Calculating overall statistics ===", verbose)
    overallStats = data_imputed[tStart:tEnd].apply(["mean", "std"])
    for col in overallStats:
        summary[f"{col}-overall-avg"] = overallStats[col].loc["mean"]
        summary[f"{col}-overall-sd"] = overallStats[col].loc["std"]

    dayOfWeekStats = data_imputed.groupby([data_imputed.index.weekday, 
                                           data_imputed.index.hour]).mean()
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
        to_screen("=== Calculating circadian metrics ===", verbose)
        summary = circadian.calculatePSD(data_imputed, epochPeriod, False, labels, summary)
        summary = circadian.calculatePSD(data_imputed, epochPeriod, True, labels, summary)
        summary = circadian.calculateFourierFreq(
            data_imputed, epochPeriod, False, labels, summary
        )
        summary = circadian.calculateFourierFreq(
            data_imputed, epochPeriod, True, labels, summary
        )
        summary = circadian.calculateM10L5(data_imputed, epochPeriod, summary)

    return summary


def _daily_summary(data, data_imputed, labels, exclude_daily_wear_below, verbose=False):
    to_screen("=== Daily summary ===", verbose)

    min_wear_per_day = 0 if exclude_daily_wear_below is None \
        else pd.Timedelta(exclude_daily_wear_below).total_seconds() / 60 # in minutes
    
    daily_enmo = summarize_daily_enmo(data["acc"], data_imputed["acc"], min_wear_per_day)
    daily_activity = summarize_daily_activity(data[labels], data_imputed[labels], labels, min_wear_per_day)
    daily_summary = pd.concat([daily_enmo, daily_activity], axis=1)
    daily_summary.index.name = "Date"

    return daily_summary
