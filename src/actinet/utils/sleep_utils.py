import numpy as np
import pandas as pd

SLEEP_GAP_TOLERANCE = 30 * 60  # 30 minutes
SLEEP_BLOCK_PERIOD = 24 * 60 * 60  # 24 hours

HMM_SEDENTARY_CODE = 2
HMM_SLEEP_CODE = 3


def removeSpuriousSleep(Y, labels, period, sleepTol='1H', removeNaps=False):
    """
    Remove spurious sleep epochs from activity classification.

    :param numpy.ndarray Y: Sequence of activity labels
    :param list labels: List of activity labels
    :param int period: Time period between successive labels in seconds
    :param str sleepTol: Minimum sleep duration, e.g. '1H'
    :param bool removeNaps: If True, restricts sleep to the longest sleep block each 24 hours.
    :return: Numpy array of revised model output
    :rtype: numpy.ndarray

    """
    label_map = {label: i for i, label in enumerate(labels)}

    if not sleepTol and not removeNaps:
        return Y

    try:
        sleep_code = label_map['sleep']
        sedentary_code = label_map['sedentary'] if 'sedentary' in labels else label_map['sit-stand']
    except KeyError:
        raise ValueError(f"'sleep' and 'sedentary' or 'sit-stand' must be output labels for spurious sleep correction.")

    if sleepTol:
        Y = convertSleepBelowThreshold(Y, period, sleep_code, sedentary_code, sleepTol)

    if removeNaps:
        Y = convertNaps(Y, period, sleep_code, sedentary_code)

    return Y


def convertSleepBelowThreshold(Y, period, sleep_code, sedentary_code, sleepTol='1H'):
    """
    Convert sleep labels to sedentary if sleep block is below an expected threshold.

    :param numpy.ndarray Y: Sequence of activity labels
    :param int period: Time period between successive labels in seconds
    :param int sleep_code: Code for sleep in the labels
    :param int sedentary_code: Code for sedentary in the labels
    :param str sleepTol: Minimum sleep duration, e.g. '1H'

    :return: Numpy array of revised model output
    :rtype: numpy.ndarray

    """
    Y_series = pd.Series(Y.copy())
    sleep_mask = Y_series == sleep_code

    if sleep_mask.sum() == 0:
        return Y

    sleepStreak = (
        sleep_mask.ne(sleep_mask.shift())
        .cumsum()
        .pipe(lambda x: x.groupby(x).transform('count') * sleep_mask)
    )

    short_sleep = sleep_mask & (sleepStreak < (pd.Timedelta(sleepTol).total_seconds() / period))
    Y_series.loc[short_sleep] = sedentary_code

    return Y_series.values


def convertNaps(Y, period, sleep_code, sedentary_code):
    """
    Convert sleep labels to sedentary, if they do not occur during the longest sleep block in each 24-hour period.

    :param numpy.ndarray Y: Sequence of activity labels
    :param int period: Time between labels in seconds
    :param int sleep_code: Label code indicating sleep
    :param int sedentary_code: Label code indicating sedentary

    :return: Revised numpy array with naps removed
    :rtype: numpy.ndarray
    """

    sleep_blocks = find_blocks(Y, gap_tol=int(SLEEP_GAP_TOLERANCE/period),
                               block_code=sleep_code)

    longest_blocks = select_longest_blocks_per_period(
        sleep_blocks, len(Y), block_period=int(SLEEP_BLOCK_PERIOD/period)
    )

    return convert_non_selected_block(Y, longest_blocks, sleep_code, sedentary_code)


def find_blocks(labels, gap_tol, block_code='s'):
    """Finds blocks of a specific code in a sequence of labels, allowing for gap tolerance of other labels."""
    blocks = []
    is_block = False
    gap_len = 0

    for i, elem in enumerate(labels):
        if not is_block and elem == block_code:
            is_block = True
            block_start = i
            gap_len = 0

        elif is_block:
            if elem == block_code:
                gap_len = 0

            else:
                gap_len += 1
                if gap_len == gap_tol:
                    blocks.append((block_start, i - gap_tol))
                    is_block = False

        if is_block and i == len(labels) - 1:
            blocks.append((block_start, i))

    return blocks


def extract_start_end_tuple(row):
    """Helper to extract a (start, end) tuple from a DataFrame row."""
    return (row['start'], row['end'])


def select_longest_blocks_per_period(blocks, sequence_len, block_period):
    """
    Selects the longest blocks within an expected period.
    Also includes edge blocks that start at 0 or end at sequence end.
    """
    
    if not blocks:
        return []

    blocks_df = pd.DataFrame(blocks, columns=['start', 'end'])
    blocks_df['length'] = blocks_df['end'] - blocks_df['start'] + 1

    selected = []
    ref_start = 0

    if (start_blocks := blocks_df[blocks_df['start'] == 0]).any().any():
        selected.append(extract_start_end_tuple(start_blocks.iloc[0]))
        ref_start = start_blocks.iloc[0]['end']
    if (end_blocks := blocks_df[blocks_df['end'] == sequence_len - 1]).any().any():
        selected.append(extract_start_end_tuple(end_blocks.iloc[0]))

    blocks_df = blocks_df[(blocks_df['start'] != 0) & (blocks_df['end'] != sequence_len - 1)]

    while ref_start + block_period < sequence_len:
        window_blocks = blocks_df[(blocks_df['start'] > ref_start) & (blocks_df['start'] <= ref_start + block_period)]
        if window_blocks.empty:
            ref_start += block_period
            continue
        
        best_block = window_blocks.loc[window_blocks['length'].idxmax()]
        
        selected.append(extract_start_end_tuple(best_block))
        ref_start = best_block['end']

    return selected


def convert_non_selected_block(labels, selected_blocks, block_code='s', conv_code='d'):
    """
    Converts all block code labels found outside longest blocks to conv code.
    """
    mask = np.zeros(len(labels), dtype=bool)

    for start, end in selected_blocks:
        mask[start:end+1] = True

    mask_to_change = (labels == block_code) & (~mask)
    labels[mask_to_change] = conv_code

    return labels


def add_sleep_sedentary_transitions(df):
    """ 
    Adds a single transition from sleep to sedentary, and vice-versa, for each participant, if it does not exist.
    """
    df_copy = df.copy()

    rows_to_append = []

    for group in df_copy["group"].unique():
        group_df = df_copy[df_copy["group"] == group]

        if not ((group_df["label"] == HMM_SLEEP_CODE) & (group_df["shift"] == HMM_SEDENTARY_CODE)).any():
            rows_to_append.append({"label": HMM_SLEEP_CODE, "shift": HMM_SEDENTARY_CODE, "group": group})

        if not ((group_df["label"] == HMM_SEDENTARY_CODE) & (group_df["shift"] == HMM_SLEEP_CODE)).any():
            rows_to_append.append({"label": HMM_SEDENTARY_CODE, "shift": HMM_SLEEP_CODE, "group": group})

    if rows_to_append:
        new_rows = pd.DataFrame(rows_to_append)
        df = pd.concat([df, new_rows], ignore_index=True)

    return df
