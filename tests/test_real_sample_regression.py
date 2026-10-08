import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from actinet.actinet import read

FIXTURE = Path(__file__).parent / "data" / "tiny-sample.cwa.gz"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.mark.slow
def test_real_axivity_ingestion_matches_numerical_baseline():
    assert _sha256(FIXTURE) == "b49ace733e10b606c9957b67098fbb7628061276f519ee83a8e13a0e2e9b9405"

    data, info = read(str(FIXTURE), verbose=False)

    assert data.shape == (1_070_944, 5)
    assert data.index[0] == pd.Timestamp("2023-06-08 12:21:04.510")
    assert data.index[-1] == pd.Timestamp("2023-06-08 15:19:33.940")
    assert info["Device"] == "Axivity"
    assert info["DeviceID"] == 43923
    assert info["SampleRate"] == 100.0
    assert info["ResampleRate"] == 100.0
    assert info["NumTicks"] == 1_021_800
    assert info["NumTicksAfterResample"] == len(data)
    assert info["ReadErrors"] == 0
    assert info["ReadOK"] == 1
    assert info["WearTime(days)"] == pytest.approx(0.1211432638888889, abs=1e-6)
    assert data[["x", "y", "z"]].isna().sum().tolist() == [24_263] * 3
    xyz = data[["x", "y", "z"]]
    np.testing.assert_allclose(
        xyz.mean(),
        [-0.5436567664, -0.1926568002, 0.0764095485],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        xyz.std(),
        [0.3417401314, 0.5224549174, 0.5272349715],
        rtol=0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        xyz.quantile([0.01, 0.1, 0.5, 0.9, 0.99]),
        [
            [-1.078125, -1.015625, -1.015625],
            [-0.906250, -0.968750, -0.609375],
            [-0.609375, -0.171875, 0.062500],
            [-0.046875, 0.500000, 0.734375],
            [0.343750, 1.171875, 0.921875],
        ],
        rtol=0,
        atol=1e-4,
    )
