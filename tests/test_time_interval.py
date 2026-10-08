import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from actinet.actinet import main, read
from actinet.models import ActivityClassifier


class ReadTimeIntervalTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2024-01-01 09:59:58", periods=5, freq="1s")
        self.data = pd.DataFrame(
            {
                "x": np.arange(5, dtype="float32"),
                "y": np.arange(5, dtype="float32"),
                "z": np.arange(5, dtype="float32"),
            },
            index=self.index,
        )
        self.data.index.name = "time"
        self.info = {"SampleRate": 1, "ResampleRate": 1}

    @patch("actinet.actinet.actipy.process")
    def test_read_csv_trims_to_inclusive_interval(self, process_mock):
        process_mock.side_effect = lambda data, *args, **kwargs: (data, self.info)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "sample.csv")
            self.data.to_csv(filepath, date_format="%Y-%m-%d %H:%M:%S.%f")

            result, _ = read(
                filepath,
                usecols="time,x,y,z",
                dateFormat="%Y-%m-%d %H:%M:%S.%f",
                sample_rate=1,
                start_time="2024-01-01 09:59:59",
                end_time="2024-01-01 10:00:01",
                verbose=False,
            )

        pd.testing.assert_frame_equal(result, self.data.iloc[1:4], check_freq=False)
        pd.testing.assert_frame_equal(
            process_mock.call_args.args[0], self.data, check_freq=False
        )

    @patch("actinet.actinet.actipy.read_device")
    def test_read_device_supports_independent_start_and_end_bounds(
        self, read_device_mock
    ):
        read_device_mock.return_value = (self.data, self.info)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "sample.cwa")
            with open(filepath, "wb") as f:
                f.write(b"test")

            start_only, _ = read(
                filepath,
                start_time="2024-01-01 10:00:00",
                verbose=False,
            )
            end_only, _ = read(
                filepath,
                end_time="2024-01-01 10:00:00",
                verbose=False,
            )

        pd.testing.assert_index_equal(start_only.index, self.index[2:])
        pd.testing.assert_index_equal(end_only.index, self.index[:3])

    @patch("actinet.actinet.actipy.read_device")
    def test_read_rejects_invalid_bounds_before_preprocessing(self, read_device_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "sample.cwa")
            with open(filepath, "wb") as f:
                f.write(b"test")

            cases = [
                {
                    "start_time": "2024-01-02",
                    "end_time": "2024-01-01",
                },
                {"start_time": "not-a-timestamp"},
                {
                    "start_time": "2024-01-01 10:00:00+00:00",
                    "end_time": "2024-01-01 11:00:00",
                },
            ]
            for bounds in cases:
                with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                    read(filepath, verbose=False, **bounds)

        read_device_mock.assert_not_called()

    @patch("actinet.actinet.actipy.read_device")
    def test_read_rejects_non_overlapping_and_timezone_incompatible_bounds(
        self, read_device_mock
    ):
        read_device_mock.return_value = (self.data, self.info)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "sample.cwa")
            with open(filepath, "wb") as f:
                f.write(b"test")

            with self.assertRaisesRegex(ValueError, "does not overlap"):
                read(
                    filepath,
                    start_time="2024-01-02 00:00:00",
                    verbose=False,
                )
            with self.assertRaisesRegex(ValueError, "compatible timezones"):
                read(
                    filepath,
                    start_time="2024-01-01 10:00:00+00:00",
                    verbose=False,
                )

    def test_classifier_rejects_data_without_a_valid_prediction_window(self):
        classifier = ActivityClassifier(window_sec=30, labels=["sleep"])

        with self.assertRaisesRegex(ValueError, "at least two timestamps"):
            classifier.predict_from_frame(
                self.data.iloc[:1], sample_freq=None, hmm_smothing=False
            )

        short_data = self.data.iloc[:2]
        with self.assertRaisesRegex(ValueError, "valid samples"):
            classifier.predict_from_frame(
                short_data, sample_freq=1, hmm_smothing=False
            )

    @patch("actinet.actinet.save_and_print_summary")
    @patch("actinet.actinet.calculate_daily_wear_stats")
    @patch("actinet.actinet.calculate_wear_stats")
    @patch("actinet.actinet.read")
    def test_cli_forwards_start_and_end_to_reader(
        self,
        read_mock,
        calculate_wear_stats_mock,
        calculate_daily_wear_stats_mock,
        save_summary_mock,
    ):
        empty_data = pd.DataFrame(
            columns=["x", "y", "z"], index=pd.DatetimeIndex([], name="time")
        )
        read_mock.return_value = (empty_data, {"ReadOK": 1})
        calculate_wear_stats_mock.return_value = {}
        calculate_daily_wear_stats_mock.return_value = pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "actinet",
                "sample.cwa",
                "--outdir",
                tmpdir,
                "--start",
                "2024-01-01 10:00:00",
                "--end",
                "2024-01-08 09:59:59",
                "--quiet",
            ]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                main()

        self.assertEqual(read_mock.call_args.kwargs["start_time"], argv[5])
        self.assertEqual(read_mock.call_args.kwargs["end_time"], argv[7])
        save_summary_mock.assert_called_once()

    @patch("actinet.actinet.get_activity_summary")
    @patch("actinet.actinet.save_and_print_summary")
    @patch("actinet.actinet.calculate_daily_wear_stats")
    @patch("actinet.actinet.calculate_wear_stats")
    @patch("actinet.actinet.load_classifier")
    @patch("actinet.actinet.read")
    def test_cli_rejects_fewer_than_three_prediction_epochs_before_writing(
        self,
        read_mock,
        load_classifier_mock,
        calculate_wear_stats_mock,
        calculate_daily_wear_stats_mock,
        save_summary_mock,
        get_activity_summary_mock,
    ):
        read_mock.return_value = (self.data, {"ReadOK": 1})
        calculate_wear_stats_mock.return_value = {}
        calculate_daily_wear_stats_mock.return_value = pd.DataFrame()
        classifier = MagicMock()
        classifier.labels = ["sleep"]
        classifier.window_sec = 30
        load_classifier_mock.return_value = classifier

        for epoch_count in (1, 2):
            with self.subTest(epoch_count=epoch_count):
                classifier.predict_from_frame.return_value = pd.DataFrame(
                    {"sleep": np.ones(epoch_count)},
                    index=pd.date_range(
                        "2024-01-01", periods=epoch_count, freq="30s"
                    ),
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    argv = [
                        "actinet",
                        "sample.cwa",
                        "--outdir",
                        tmpdir,
                        "--pytorch-device",
                        "cpu",
                        "--quiet",
                    ]
                    with patch.object(sys, "argv", argv), self.assertRaisesRegex(
                        ValueError, "at least three"
                    ):
                        main()
                    self.assertFalse(
                        os.path.exists(
                            os.path.join(
                                tmpdir, "sample", "sample-timeSeries.csv.gz"
                            )
                        )
                    )

        self.assertEqual(save_summary_mock.call_count, 2)
        get_activity_summary_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
