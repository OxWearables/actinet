import json
import sys

import numpy as np
import pandas as pd
import pytest

from actinet.utils import eval_utils
from actinet.utils.collate_outputs import collate_outputs
from actinet.utils.collate_outputs import main as collate_main
from actinet.utils.generate_commands import generate_commands
from actinet.utils.generate_commands import main as generate_main


def _evaluation_results():
    rows = []
    for participant, age, sex in [("P1", "18-40", "F"), ("P2", "41-60", "M")]:
        for model, offset in [("accelerometer", 1), ("actinet", 0)]:
            rows.append(
                {
                    "Participant": participant,
                    "Age Band": age,
                    "Sex": sex,
                    "Model": model,
                    "Pred_dict": {"sleep": 4 + offset, "light": 2},
                    "True_dict": {"sleep": 4, "light": 2},
                }
            )
    return pd.DataFrame(rows)


def test_eval_utils_metrics_extraction_and_tables():
    divided = eval_utils.DivDict({"a": 4, "b": 2}) / 2
    assert divided == {"a": 2, "b": 1}
    with pytest.raises(TypeError):
        eval_utils.DivDict({"a": 1}) / "two"
    metrics = eval_utils.calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    np.testing.assert_allclose(metrics, [0.75, 0.7333333333, 0.5, 0.75])

    results = _evaluation_results()
    baseline, actinet, population = eval_utils.extract_activity_predictions(results, "sleep")
    np.testing.assert_array_equal(baseline, [5, 5])
    np.testing.assert_array_equal(actinet, [4, 4])
    assert population == 2
    extracted = eval_utils.extract_activity_predictions(
        results, "sleep", age_band="18-40", sex="F", return_true_labels=True
    )
    assert extracted[-1] == 1
    assert all(values.tolist() == [expected] for values, expected in zip(extracted[:-1], [5, 4, 4, 4]))

    assert eval_utils.build_mae_cell(np.array([1, 2]), np.array([2, 4])) == "1.500 ± 0.500"
    assert eval_utils.build_pvalue_cell(np.arange(10), np.arange(10) + 100) == "<0.001"
    table = eval_utils.build_mae_table(results, ["sleep"])
    assert table.loc["Baseline", "sleep"] == "1.000 ± 0.000"
    assert table.loc["ActiNet", "sleep"] == "0.000 ± 0.000"
    assert eval_utils.convert_version("1.2.3+45") == "v1-2-3"
    with pytest.raises(ValueError, match="expected format"):
        eval_utils.convert_version("release")


def test_collate_outputs_function_and_cli(tmp_path, monkeypatch, capsys):
    outputs = tmp_path / "outputs"
    (outputs / "a").mkdir(parents=True)
    (outputs / "b").mkdir()
    (outputs / "a" / "a-outputSummary.json").write_text(json.dumps({"Filename": "a", "score": 1}))
    (outputs / "b" / "b-outputSummary.json").write_text(json.dumps({"Filename": "b", "score": 2}))
    outfile = tmp_path / "summary.csv"
    collate_outputs(str(outputs), str(outfile))
    frame = pd.read_csv(outfile).sort_values("Filename")
    assert frame.to_dict("records") == [{"Filename": "a", "score": 1}, {"Filename": "b", "score": 2}]
    assert "Found 2 summary files" in capsys.readouterr().out

    cli_out = tmp_path / "cli.csv"
    monkeypatch.setattr(sys, "argv", ["collate", str(outputs), "--outfile", str(cli_out)])
    collate_main()
    assert cli_out.exists()


def test_generate_commands_function_and_cli(tmp_path, monkeypatch, capsys):
    inputs = tmp_path / "inputs"
    (inputs / "group").mkdir(parents=True)
    for name in ["one.CWA", "two.cwa.gz", "skip.csv"]:
        (inputs / "group" / name).touch()
    commands = tmp_path / "commands.txt"
    generate_commands(
        str(inputs), str(tmp_path / "outputs"), str(commands), fext="cwa", cmdopts="--quiet"
    )
    lines = commands.read_text().splitlines()
    assert len(lines) == 2
    assert all(line.startswith("actinet '") and line.endswith("--quiet") for line in lines)
    assert "Found 2 accelerometer files" in capsys.readouterr().out

    cli_commands = tmp_path / "cli-commands.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            str(inputs),
            "--output_dir",
            str(tmp_path / "cli-output"),
            "--cmdsfile",
            str(cli_commands),
            "--fext",
            "cwa",
        ],
    )
    generate_main()
    assert len(cli_commands.read_text().splitlines()) == 2
