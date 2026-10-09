import csv
import errno
import json
import multiprocessing
import os
import stat
import sys
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from actinet.utils import eval_utils
from actinet.utils.collate_outputs import collate_outputs
from actinet.utils.collate_outputs import main as collate_main
from actinet.utils.generate_commands import generate_commands
from actinet.utils.generate_commands import main as generate_main

collate_module = import_module("actinet.utils.collate_outputs")


def _hold_collation_lock(outfile, acquired, release):
    with collate_module._output_lock(collate_module.Path(outfile)):
        acquired.set()
        release.wait(10)


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


def test_collate_outputs_unions_schemas_by_name_in_source_order(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "z-outputSummary.json").write_text(json.dumps({"Filename": "z", "shared": 2, "last": 3}))
    (outputs / "a-outputSummary.json").write_text(json.dumps({"first": 1, "Filename": "a", "shared": 4}))

    outfile = tmp_path / "summary.csv"
    collate_outputs(outputs, outfile)

    frame = pd.read_csv(outfile)
    assert frame.columns.tolist() == ["first", "Filename", "shared", "last"]
    assert frame["Filename"].tolist() == ["a", "z"]
    assert pd.isna(frame.loc[0, "last"])
    assert pd.isna(frame.loc[1, "first"])


def test_collate_outputs_preserves_scalar_and_nested_values(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(
        json.dumps(
            {
                "integer": 3,
                "decimal": 1.25,
                "missing": None,
                "flag": True,
                "nested": {"sleep": 4},
                "values": [1, 2],
            }
        )
    )

    outfile = tmp_path / "summary.csv"
    collate_outputs(outputs, outfile)

    assert pd.read_csv(outfile, keep_default_na=False).to_dict("records") == [
        {
            "integer": 3,
            "decimal": 1.25,
            "missing": "",
            "flag": True,
            "nested": "{'sleep': 4}",
            "values": "[1, 2]",
        }
    ]


def test_collate_outputs_writes_nan_as_blank_cell(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(
        json.dumps({"missing": float("nan"), "literal": "NaN"})
    )

    outfile = tmp_path / "summary.csv"
    collate_outputs(outputs, outfile)

    with outfile.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"missing": "", "literal": "NaN"}]


def test_collate_outputs_escapes_spreadsheet_formulas_in_text(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(
        json.dumps(
            {
                "=header": "=formula",
                "plus": "+formula",
                "minus": "-formula",
                "at": "@formula",
                "full_width_equals": "＝formula",
                "full_width_plus": "＋formula",
                "full_width_minus": "－formula",
                "full_width_at": "＠formula",
                "tab": "\tformula",
                "return": "\rformula",
                "newline": "\nformula",
                "negative_integer": -2,
                "negative_decimal": -1.5,
            }
        )
    )

    outfile = tmp_path / "summary.csv"
    collate_outputs(outputs, outfile)

    with outfile.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        [
            "'=header",
            "plus",
            "minus",
            "at",
            "full_width_equals",
            "full_width_plus",
            "full_width_minus",
            "full_width_at",
            "tab",
            "return",
            "newline",
            "negative_integer",
            "negative_decimal",
        ],
        [
            "'=formula",
            "'+formula",
            "'-formula",
            "'@formula",
            "'＝formula",
            "'＋formula",
            "'－formula",
            "'＠formula",
            "'\tformula",
            "'\rformula",
            "'\nformula",
            "-2",
            "-1.5",
        ],
    ]


def test_collate_outputs_rejects_escaped_header_collisions(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(
        json.dumps({"=metric": 1, "'=metric": 2})
    )
    outfile = tmp_path / "summary.csv"
    outfile.write_text("old output\n")

    with pytest.raises(ValueError, match="collide after spreadsheet-safe escaping"):
        collate_outputs(outputs, outfile)

    assert outfile.read_text() == "old output\n"


def test_collate_outputs_strict_accepts_reordered_keys(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(json.dumps({"Filename": "a", "score": 1}))
    (outputs / "b-outputSummary.json").write_text(json.dumps({"score": 2, "Filename": "b"}))

    outfile = tmp_path / "summary.csv"
    collate_outputs(outputs, outfile, schema_policy="strict")

    assert pd.read_csv(outfile).to_dict("records") == [
        {"Filename": "a", "score": 1},
        {"Filename": "b", "score": 2},
    ]


def test_collate_outputs_strict_mismatch_preserves_existing_output(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(json.dumps({"Filename": "a", "score": 1}))
    (outputs / "b-outputSummary.json").write_text(json.dumps({"Filename": "b", "extra": 2}))
    outfile = tmp_path / "summary.csv"
    outfile.write_text("old output\n")

    with pytest.raises(ValueError, match="schema differs under strict policy"):
        collate_outputs(outputs, outfile, schema_policy="strict")

    assert outfile.read_text() == "old output\n"


@pytest.mark.parametrize("contents", ["{broken", "[]", '"not an object"'])
def test_collate_outputs_invalid_json_preserves_existing_output(tmp_path, contents):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "bad-outputSummary.json").write_text(contents)
    outfile = tmp_path / "summary.csv"
    outfile.write_text("old output\n")

    with pytest.raises(ValueError, match="JSON"):
        collate_outputs(outputs, outfile)

    assert outfile.read_text() == "old output\n"


def test_collate_outputs_rejects_missing_file_and_empty_directory(tmp_path):
    outfile = tmp_path / "summary.csv"
    outfile.write_text("old output\n")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        collate_outputs(tmp_path / "missing", outfile)

    input_file = tmp_path / "input.txt"
    input_file.write_text("not a directory")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        collate_outputs(input_file, outfile)

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    with pytest.raises(FileNotFoundError, match="No output summary files"):
        collate_outputs(outputs, outfile)

    assert outfile.read_text() == "old output\n"


def test_collate_outputs_preserves_existing_output_permissions(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(json.dumps({"Filename": "a"}))
    outfile = tmp_path / "summary.csv"
    outfile.write_text("old output\n")
    outfile.chmod(0o440)

    collate_outputs(outputs, outfile)

    assert stat.S_IMODE(outfile.stat().st_mode) == 0o440
    assert pd.read_csv(outfile).to_dict("records") == [{"Filename": "a"}]


def test_collate_outputs_uses_source_snapshot_during_publication(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    source = outputs / "a-outputSummary.json"
    source.write_text(json.dumps({"Filename": "a", "score": 1}))
    outfile = tmp_path / "summary.csv"
    outfile.write_text("old output\n")
    original_write = collate_module._write_json_collation

    def change_source_then_write(plan, staged_path):
        source.write_text(json.dumps({"Filename": "a", "score": 2}))
        original_write(plan, staged_path)

    monkeypatch.setattr(
        collate_module,
        "_write_json_collation",
        change_source_then_write,
    )

    collate_outputs(outputs, outfile)

    assert pd.read_csv(outfile).to_dict("records") == [{"Filename": "a", "score": 1}]
    assert not list(tmp_path.glob(".summary.csv.*.tmp"))


def test_collate_outputs_skips_symlinked_sources(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(json.dumps({"Filename": "a"}))
    external = tmp_path / "external.json"
    external.write_text(json.dumps({"secret": "outside"}))
    (outputs / "b-outputSummary.json").symlink_to(external)

    outfile = tmp_path / "summary.csv"
    collate_outputs(outputs, outfile)

    assert pd.read_csv(outfile).to_dict("records") == [{"Filename": "a"}]


def test_collate_outputs_skips_symlinked_source_directories(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(json.dumps({"Filename": "a"}))
    external = tmp_path / "external"
    external.mkdir()
    (external / "b-outputSummary.json").write_text(json.dumps({"secret": "outside"}))
    (outputs / "linked").symlink_to(external, target_is_directory=True)

    outfile = tmp_path / "summary.csv"
    collate_outputs(outputs, outfile)

    assert pd.read_csv(outfile).to_dict("records") == [{"Filename": "a"}]


@pytest.mark.skipif(not hasattr(os, "fwalk"), reason="descriptor traversal is unavailable")
def test_collate_outputs_rejects_source_replaced_during_open(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    source = outputs / "a-outputSummary.json"
    source.write_text(json.dumps({"Filename": "original"}))
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps({"Filename": "replacement"}))
    outfile = tmp_path / "summary.csv"
    outfile.write_text("old output\n")
    original_open = collate_module.os.open
    replaced = False

    def replace_then_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        if path == source.name and dir_fd is not None and not replaced:
            replacement.replace(source)
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(collate_module.os, "open", replace_then_open)

    with pytest.raises(ValueError, match="changed while opening"):
        collate_outputs(outputs, outfile)

    assert outfile.read_text() == "old output\n"


def test_collate_outputs_preserves_global_source_path_order(tmp_path):
    outputs = tmp_path / "outputs"
    nested = outputs / "nested"
    nested.mkdir(parents=True)
    (outputs / "z-outputSummary.json").write_text(
        json.dumps({"root": 1, "Filename": "z"})
    )
    (nested / "a-outputSummary.json").write_text(
        json.dumps({"nested": 2, "Filename": "a"})
    )

    outfile = tmp_path / "summary.csv"
    collate_outputs(outputs, outfile)

    frame = pd.read_csv(outfile)
    assert frame.columns.tolist() == ["nested", "Filename", "root"]
    assert frame["Filename"].tolist() == ["a", "z"]


def test_collate_outputs_write_failure_cleans_stage_and_preserves_output(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(json.dumps({"Filename": "a"}))
    outfile = tmp_path / "summary.csv"
    outfile.write_text("old output\n")

    def fail_write(plan, staged_path):
        staged_path.write_text("partial output\n")
        raise OSError("disk full")

    monkeypatch.setattr(collate_module, "_write_json_collation", fail_write)

    with pytest.raises(OSError, match="disk full"):
        collate_outputs(outputs, outfile)

    assert outfile.read_text() == "old output\n"
    assert not list(tmp_path.glob(".summary.csv.*.tmp"))


def test_collate_outputs_locks_before_discovery(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(json.dumps({"Filename": "a"}))
    lock_held = False
    original_lock = collate_module._lock_descriptor
    original_snapshot = collate_module._snapshot_summary_files

    def record_lock(descriptor):
        nonlocal lock_held
        original_lock(descriptor)
        lock_held = True

    def assert_locked(outputs_path, outfile_path, snapshot_directory):
        assert lock_held
        return original_snapshot(outputs_path, outfile_path, snapshot_directory)

    monkeypatch.setattr(collate_module, "_lock_descriptor", record_lock)
    monkeypatch.setattr(collate_module, "_snapshot_summary_files", assert_locked)

    collate_outputs(outputs, tmp_path / "summary.csv")


def test_output_lock_path_is_stable_per_destination_entry(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    destination = real_parent / "summary.csv"
    alias = alias_parent / "summary.csv"
    other = real_parent / "other.csv"

    before = collate_module._output_lock_path(alias)
    alias.symlink_to(tmp_path / "target.csv")
    alias.unlink()
    alias.write_text("replacement")

    assert before == collate_module._output_lock_path(alias)
    assert before == collate_module._output_lock_path(destination)
    assert before != collate_module._output_lock_path(other)
    assert before.parent == real_parent / ".actinet-collate-locks"
    assert before.name == destination.name


def test_collate_outputs_keeps_canonical_destination_when_parent_alias_changes(
    tmp_path, monkeypatch
):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(json.dumps({"Filename": "a"}))
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    first_parent.mkdir()
    second_parent.mkdir()
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(first_parent, target_is_directory=True)
    original_snapshot = collate_module._snapshot_summary_files

    def retarget_parent_then_snapshot(outputs_path, outfile_path, snapshot_directory):
        alias_parent.unlink()
        alias_parent.symlink_to(second_parent, target_is_directory=True)
        return original_snapshot(outputs_path, outfile_path, snapshot_directory)

    monkeypatch.setattr(
        collate_module,
        "_snapshot_summary_files",
        retarget_parent_then_snapshot,
    )

    collate_outputs(outputs, alias_parent / "summary.csv")

    assert (first_parent / "summary.csv").is_file()
    assert not (second_parent / "summary.csv").exists()


def test_output_lock_rejects_symlinked_lock_directory(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    lock_directory = tmp_path / ".actinet-collate-locks"
    lock_directory.symlink_to(external, target_is_directory=True)

    with pytest.raises(RuntimeError, match="lock directory is not a directory"):
        collate_module._output_lock_path(tmp_path / "summary.csv")


def test_collate_outputs_rejects_symlinked_output_without_replacing_target(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(json.dumps({"Filename": "a"}))
    target = tmp_path / "target.csv"
    target.write_text("target contents\n")
    outfile = tmp_path / "summary.csv"
    outfile.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symlink"):
        collate_outputs(outputs, outfile)

    assert outfile.is_symlink()
    assert target.read_text() == "target contents\n"


def test_output_lock_serializes_only_same_destination(tmp_path):
    context = multiprocessing.get_context("spawn")
    first_acquired = context.Event()
    same_acquired = context.Event()
    other_acquired = context.Event()
    release_first = context.Event()
    release_same = context.Event()
    release_other = context.Event()
    outfile = tmp_path / "summary.csv"
    other_outfile = tmp_path / "other.csv"
    processes = [
        context.Process(
            target=_hold_collation_lock,
            args=(str(outfile), first_acquired, release_first),
        ),
        context.Process(
            target=_hold_collation_lock,
            args=(str(outfile), same_acquired, release_same),
        ),
        context.Process(
            target=_hold_collation_lock,
            args=(str(other_outfile), other_acquired, release_other),
        ),
    ]

    processes[0].start()
    try:
        assert first_acquired.wait(5)
        processes[1].start()
        processes[2].start()
        assert other_acquired.wait(5)
        assert not same_acquired.wait(0.5)
        release_first.set()
        assert same_acquired.wait(5)
    finally:
        release_first.set()
        release_same.set()
        release_other.set()
        for process in processes:
            if process.pid is not None:
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join(5)

    assert all(process.exitcode == 0 for process in processes)


def test_windows_output_lock_retries_only_contention(monkeypatch):
    attempts = 0
    sleeps = []

    def locking(descriptor, mode, length):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError(errno.EACCES, "locked")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=1, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(collate_module.os, "name", "nt")
    monkeypatch.setattr(collate_module.os, "lseek", lambda *args: None)
    monkeypatch.setattr(collate_module.time, "sleep", sleeps.append)

    collate_module._lock_descriptor(12)

    assert attempts == 3
    assert sleeps == [0.1, 0.1]


def test_windows_output_lock_propagates_non_contention_errors(monkeypatch):
    def locking(descriptor, mode, length):
        raise OSError(errno.EIO, "I/O failure")

    fake_msvcrt = SimpleNamespace(LK_NBLCK=1, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(collate_module.os, "name", "nt")
    monkeypatch.setattr(collate_module.os, "lseek", lambda *args: None)

    with pytest.raises(OSError, match="I/O failure"):
        collate_module._lock_descriptor(12)


def test_collate_outputs_rejects_unknown_policy_before_mutation(tmp_path):
    output_directory = tmp_path / "new-output-directory"

    with pytest.raises(ValueError, match="Unknown schema policy"):
        collate_outputs(tmp_path / "missing", output_directory / "summary.csv", "invalid")

    assert not output_directory.exists()


def test_collate_main_forwards_strict_schema_policy(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "a-outputSummary.json").write_text(json.dumps({"Filename": "a"}))
    (outputs / "b-outputSummary.json").write_text(json.dumps({"Filename": "b", "score": 2}))

    with pytest.raises(ValueError, match="schema differs under strict policy"):
        collate_main(
            [
                str(outputs),
                "--outfile",
                str(tmp_path / "summary.csv"),
                "--schema-policy",
                "strict",
            ]
        )


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
