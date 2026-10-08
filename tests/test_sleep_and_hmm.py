import numpy as np
import pandas as pd
import pytest

from actinet import hmm
from actinet.utils import sleep_utils


def test_short_sleep_and_nap_correction():
    labels = ["light", "sedentary", "sleep"]
    sequence = np.array([1, 2, 2, 1, 1, 2, 2, 2, 2, 1])
    corrected = sleep_utils.removeSpuriousSleep(
        sequence.copy(), labels, period=30 * 60, sleepTol="90min", removeNaps=False
    )
    np.testing.assert_array_equal(corrected, [1, 1, 1, 1, 1, 2, 2, 2, 2, 1])
    assert sleep_utils.removeSpuriousSleep(sequence, labels, 1800, None, False) is sequence
    with pytest.raises(ValueError, match="must be output labels"):
        sleep_utils.removeSpuriousSleep(sequence, ["sleep", "light"], 1800, "1h")


def test_find_select_and_convert_sleep_blocks():
    sequence = np.array(list("ssddssdsssssddss"))
    assert sleep_utils.find_blocks(sequence, gap_tol=2) == [(0, 1), (4, 11), (14, 15)]
    blocks = [(0, 2), (5, 8), (12, 17), (23, 24)]
    selected = sleep_utils.select_longest_blocks_per_period(blocks, 25, 10)
    assert (0, 2) in selected
    assert (23, 24) in selected
    assert sleep_utils.select_longest_blocks_per_period([], 10, 5) == []
    converted = sleep_utils.convert_non_selected_block(
        np.array(list("ssddss")), [(0, 1)], "s", "d"
    )
    assert "".join(converted) == "ssdddd"
    assert sleep_utils.extract_start_end_tuple(pd.Series({"start": 2, "end": 5})) == (2, 5)


def test_convert_naps_keeps_longest_daily_block(monkeypatch):
    sequence = np.array([3, 3, 2, 3, 2, 2, 3, 3, 3, 2, 2, 2])
    monkeypatch.setattr(sleep_utils, "find_blocks", lambda *args, **kwargs: [(0, 1), (3, 3), (6, 8)])
    monkeypatch.setattr(
        sleep_utils,
        "select_longest_blocks_per_period",
        lambda *args, **kwargs: [(0, 1), (6, 8)],
    )
    converted = sleep_utils.convertNaps(sequence.copy(), period=1, sleep_code=3, sedentary_code=2)
    np.testing.assert_array_equal(converted, [3, 3, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2])


def test_sleep_transition_rows_are_added_per_group():
    transitions = pd.DataFrame(
        {"label": [0, 3, 2], "shift": [1, 2, 3], "group": ["a", "a", "b"]}
    )
    result = sleep_utils.add_sleep_sedentary_transitions(transitions)
    for group in ["a", "b"]:
        pairs = set(map(tuple, result.loc[result.group == group, ["label", "shift"]].values))
        assert (3, 2) in pairs
        assert (2, 3) in pairs


def test_transition_matrix_respects_time_gaps_and_groups():
    labels = np.array([0, 0, 1, 1, 0])
    times = [0, 30, 60, 120, 150]
    matrix = hmm.calculate_transition_matrix(labels, t=times, interval=30)
    np.testing.assert_allclose(matrix, [[0.5, 0.5], [1.0, 0.0]])
    ignored = hmm.calculate_transition_matrix(labels, ignore_transition_gaps=True)
    np.testing.assert_allclose(ignored, [[0.5, 0.5], [0.5, 0.5]])
    dtimes = pd.date_range("2024-01-01", periods=5, freq="30s")
    dated = hmm.calculate_transition_matrix(labels, t=dtimes, interval=30)
    np.testing.assert_allclose(dated.sum(axis=1), [1, 1])


def test_hmm_fit_viterbi_predict_and_persistence(tmp_path):
    probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2]]
    )
    truth = np.array([0, 0, 1, 1, 0])
    times = [0, 30, 60, 90, 150]
    model = hmm.HMM(uniform_prior=False)
    model.fit(probabilities, truth, T=times, interval=30)
    np.testing.assert_allclose(model.prior, [0.6, 0.4])
    assert model.emission.shape == (2, 2)
    assert model.transition.shape == (2, 2)
    smoothed = model.predict(np.array([0, 0, 1, 1, 1]), times, 30)
    assert smoothed.shape == truth.shape
    assert smoothed[-1] == 1  # restored after the final timestamp gap
    assert "Hidden Markov Model" in str(model)

    path = tmp_path / "nested" / "hmm.npz"
    model.save(path)
    loaded = hmm.HMM()
    loaded.load(path)
    np.testing.assert_allclose(loaded.emission, model.emission)
    np.testing.assert_array_equal(loaded.labels, model.labels)


def test_hmm_validation_and_matrix_helpers(capsys):
    with pytest.raises(Exception, match="same length"):
        hmm.check_for_input_errors([1], [], 30)
    with pytest.raises(Exception, match="window length"):
        hmm.check_for_input_errors([1], [0], None)
    with pytest.raises(Exception, match="group labels"):
        hmm.check_for_input_errors([1], [0], 30, groups=[], handle_sleep_transitions=True)
    with pytest.raises(Exception, match="No transitions"):
        hmm.calculate_transition_matrix([1], t=[0], interval=30)
    assert hmm.get_activity_label_code("sleep", ["sleep", "light"]) == 1
    with pytest.raises(ValueError, match="not recognised"):
        hmm.get_activity_label_code("walk", ["sleep"])
    np.testing.assert_array_equal(hmm.reorder_matrix([1, 2], [1, 0]), [2, 1])
    np.testing.assert_array_equal(
        hmm.reorder_matrix([[1, 2], [3, 4]], [1, 0]), [[4, 3], [2, 1]]
    )
    with pytest.raises(ValueError, match="square"):
        hmm.reorder_matrix([[1, 2, 3], [4, 5, 6]], [0, 1])
    with pytest.raises(ValueError, match="1D or 2D"):
        hmm.reorder_matrix(np.zeros((2, 2, 2)), [0, 1])
    hmm.print_array([1 / 3], precision=2)
    assert "0.33" in capsys.readouterr().out


def test_pretty_hmm_params_and_display(capsys):
    model = hmm.HMM(
        prior=np.array([0.4, 0.6]),
        emission=np.eye(2),
        transition=np.eye(2),
        labels=np.array([0, 1]),
    )
    model.display({"light": 0, "sleep": 1}, precision=2)
    assert "HMM Parameters ordered light, sleep" in capsys.readouterr().out
    hmm.pretty_hmm_params(model, ("light", "sleep"))
    assert "HMM Parameters ordered light, sleep" in capsys.readouterr().out
    with pytest.raises(ValueError, match="list or dict"):
        hmm.pretty_hmm_params(model, "light")
