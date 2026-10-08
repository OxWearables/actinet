import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_acceleration():
    index = pd.date_range("2024-01-01", periods=48, freq="1h", name="time")
    data = pd.DataFrame(
        {
            "x": np.zeros(len(index), dtype="float32"),
            "y": np.zeros(len(index), dtype="float32"),
            "z": np.ones(len(index), dtype="float32"),
        },
        index=index,
    )
    data.iloc[[2, 26]] = np.nan
    return data


@pytest.fixture
def epoch_predictions():
    index = pd.date_range("2024-01-01", periods=48, freq="1h", name="time")
    activity = np.arange(len(index)) % 4
    frame = pd.DataFrame(
        {
            "acc": np.arange(len(index), dtype=float) * 10,
            "sleep": activity == 0,
            "sedentary": activity == 1,
            "light": activity == 2,
            "moderate-vigorous": activity == 3,
        },
        index=index,
        dtype=float,
    )
    return frame
