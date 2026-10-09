from __future__ import annotations

import argparse
import csv
import errno
import gzip
import json
import logging
import math
import os
import stat
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

_LOGGER = logging.getLogger(__name__)
_LOCK_DIRECTORY = ".actinet-collate-locks"
_SPREADSHEET_FORMULA_PREFIXES = (
    "=",
    "+",
    "-",
    "@",
    "＝",
    "＋",
    "－",
    "＠",
    "\t",
    "\r",
    "\n",
)


@dataclass(frozen=True)
class _JsonSourceSnapshot:
    source: Path
    snapshot: Path


@dataclass(frozen=True)
class _JsonCollationPlan:
    columns: Tuple[str, ...]
    schemas: Tuple[Tuple[str, ...], ...]
    files: Tuple[Tuple[_JsonSourceSnapshot, int], ...]


def collate_outputs(
    outputs: Union[str, PathLike[str]],
    outfile: Union[str, PathLike[str]] = "collated-outputs/outputs.csv",
    schema_policy: str = "union",
) -> None:
    """Collate summary JSON files into the primary CSV and daily archives.

    JSON object keys are aligned by name. Under the default ``union`` policy,
    every key found across the inputs is retained and missing values are left
    blank. The ``strict`` policy requires every input to have the same keys.
    The completed primary CSV replaces *outfile* atomically. When daily inputs
    are present, they are merged into a sibling ``Daily.csv.gz`` file.
    """

    _validate_schema_policy(schema_policy)
    outputs_path = _resolve_outputs_directory(Path(outputs))
    outfile_path = Path(outfile)
    outfile_path.parent.mkdir(parents=True, exist_ok=True)
    outfile_path = outfile_path.parent.resolve() / outfile_path.name
    _validate_output_path(outfile_path)
    if outfile_path == outfile_path.parent / "Daily.csv.gz":
        raise ValueError("--outfile cannot be Daily.csv.gz; it is reserved for daily collation")

    with _output_lock(outfile_path):
        snapshot_directory = Path(
            tempfile.mkdtemp(
                dir=outfile_path.parent,
                prefix=f".{outfile_path.name}.sources.",
                suffix=".tmp",
            )
        )
        try:
            infofiles = _snapshot_summary_files(
                outputs_path,
                outfile_path,
                snapshot_directory,
            )
            if not infofiles:
                raise FileNotFoundError(
                    f"No output summary files found under {outputs_path}"
                )

            print(f"Found {len(infofiles)} summary files...")
            plan = _plan_json_collation(infofiles, schema_policy)
            has_daily_output = _collate_daily_outputs(outputs_path, outfile_path.parent)
            staged_path, final_mode = _create_staged_output(outfile_path)
            try:
                _write_json_collation(plan, staged_path)
                _sync_file(staged_path)
                os.chmod(staged_path, final_mode)
                if _path_exists(outfile_path):
                    print(f"Overwriting existing file: {outfile_path}")
                os.replace(staged_path, outfile_path)
                _fsync_directory(outfile_path.parent.resolve())
            finally:
                _cleanup_staged_output(staged_path)
        finally:
            _cleanup_snapshot_directory(snapshot_directory)

    print("Summary CSV written to", outfile_path)
    if has_daily_output:
        print("Daily CSV written to", outfile_path.parent / "Daily.csv.gz")


def _collate_daily_outputs(outputs: Path, destination: Path) -> bool:
    """Keep per-run daily archives available as one destination-level table."""
    daily_outfile = destination / "Daily.csv.gz"
    with _output_lock(daily_outfile):
        files = _find_daily_outputs(outputs, destination)
        if not files:
            return False
        staging_directory = Path(tempfile.mkdtemp(dir=destination, prefix=".Daily.csv.gz.", suffix=".tmp"))
        try:
            snapshots = _snapshot_daily_files(files, staging_directory)
            columns = _daily_columns(snapshots)
            staged = staging_directory / "Daily.csv.gz"
            with gzip.open(staged, "wt", encoding="utf-8", newline="") as stream:
                safe_columns = _spreadsheet_safe_columns(columns)
                writer = csv.DictWriter(stream, fieldnames=safe_columns, extrasaction="raise")
                writer.writeheader()
                for file in snapshots:
                    with gzip.open(file, "rt", encoding="utf-8-sig", newline="") as source:
                        reader = csv.DictReader(source, strict=True)
                        for row in reader:
                            if None in row or any(value is None for value in row.values()):
                                raise ValueError(f"Daily CSV has malformed rows: {file}")
                            writer.writerow({
                                safe_column: _normalize_daily_cell(row.get(column))
                                for column, safe_column in zip(columns, safe_columns)
                            })
            _sync_file(staged)
            if daily_outfile.exists():
                os.chmod(staged, stat.S_IMODE(daily_outfile.stat().st_mode))
            os.replace(staged, daily_outfile)
            _fsync_directory(destination.resolve())
        finally:
            if staging_directory.exists():
                for child in staging_directory.iterdir():
                    child.unlink()
                staging_directory.rmdir()
    return True


def _snapshot_daily_files(files: Sequence[Path], directory: Path) -> list[Path]:
    snapshots: list[Path] = []
    for index, source in enumerate(files):
        snapshot = directory / f"source-{index:08d}.csv.gz"
        try:
            descriptor = os.open(source, _source_open_flags())
        except OSError as error:
            raise ValueError(f"Could not safely open Daily CSV {source}: {error}") from error
        try:
            source_stat = os.fstat(descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError(f"Daily CSV source is not a regular file: {source}")
            _copy_stable_regular_file(descriptor, source, snapshot)
        finally:
            os.close(descriptor)
        snapshots.append(snapshot)
    return snapshots


def _daily_columns(files: Sequence[Path]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for file in files:
        with gzip.open(file, "rt", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if not reader.fieldnames:
                raise ValueError(f"Daily CSV has no header: {file}")
            if any(not column or not column.strip() for column in reader.fieldnames):
                raise ValueError(f"Daily CSV has blank column names: {file}")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f"Daily CSV has duplicate column names: {file}")
            for column in reader.fieldnames:
                if column not in seen:
                    seen.add(column)
                    columns.append(column)
            for row in reader:
                if None in row:
                    raise ValueError(f"Daily CSV has rows with extra fields: {file}")
    return columns


def _normalize_daily_cell(value: Any) -> Any:
    if isinstance(value, str) and _looks_numeric(value):
        return value
    return _normalize_csv_cell(value)


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return bool(value.strip())


def _find_daily_outputs(outputs: Path, destination: Path) -> list[Path]:
    resolved_destination = destination.resolve()
    generated = (resolved_destination / "Daily.csv.gz").resolve()
    return sorted(
        path
        for path in outputs.rglob("*-Daily.csv.gz")
        if path.is_file()
        and not path.is_symlink()
        and path.resolve() != generated
    )


def _resolve_outputs_directory(outputs: Path) -> Path:
    resolved_outputs = outputs.resolve()
    try:
        outputs_stat = resolved_outputs.stat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Outputs directory does not exist: {resolved_outputs}") from error
    if not stat.S_ISDIR(outputs_stat.st_mode):
        raise NotADirectoryError(f"Outputs path is not a directory: {resolved_outputs}")
    return resolved_outputs


def _snapshot_summary_files(
    outputs: Path,
    outfile: Path,
    snapshot_directory: Path,
) -> list[_JsonSourceSnapshot]:
    if hasattr(os, "fwalk") and os.name != "nt":
        return _snapshot_summary_files_with_descriptors(
            outputs,
            outfile,
            snapshot_directory,
        )
    return _snapshot_summary_files_with_paths(outputs, outfile, snapshot_directory)


def _snapshot_summary_files_with_descriptors(
    outputs: Path,
    outfile: Path,
    snapshot_directory: Path,
) -> list[_JsonSourceSnapshot]:
    resolved_outfile = outfile.resolve()
    snapshots: list[_JsonSourceSnapshot] = []
    for root_value, directory_names, file_names, root_descriptor in os.fwalk(
        outputs,
        topdown=True,
        onerror=_raise_walk_error,
        follow_symlinks=False,
    ):
        directory_names.sort()
        file_names.sort()
        root = Path(root_value)
        for file_name in file_names:
            if not file_name.endswith("-outputSummary.json"):
                continue
            source = root / file_name
            if source == resolved_outfile:
                continue
            try:
                source_stat = os.stat(
                    file_name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(source_stat.st_mode):
                continue
            snapshots.append(
                _snapshot_descriptor_source(
                    source,
                    file_name,
                    root_descriptor,
                    source_stat,
                    snapshot_directory / f"{len(snapshots):08d}.json",
                )
            )
    snapshots.sort(key=lambda item: item.source.as_posix())
    return snapshots


def _snapshot_descriptor_source(
    source: Path,
    file_name: str,
    root_descriptor: int,
    expected_stat: os.stat_result,
    snapshot: Path,
) -> _JsonSourceSnapshot:
    flags = _source_open_flags()
    try:
        descriptor = os.open(file_name, flags, dir_fd=root_descriptor)
    except OSError as error:
        raise ValueError(f"Could not safely open JSON file {source}: {error}") from error
    try:
        descriptor_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or not _same_file_identity(descriptor_stat, expected_stat)
        ):
            raise ValueError(f"JSON source changed while opening: {source}")
        _copy_stable_regular_file(descriptor, source, snapshot)
    finally:
        os.close(descriptor)
    return _JsonSourceSnapshot(source, snapshot)


def _snapshot_summary_files_with_paths(
    outputs: Path,
    outfile: Path,
    snapshot_directory: Path,
) -> list[_JsonSourceSnapshot]:
    resolved_outfile = outfile.resolve()
    snapshots: list[_JsonSourceSnapshot] = []
    for root_value, directory_names, file_names in os.walk(
        outputs,
        topdown=True,
        onerror=_raise_walk_error,
        followlinks=False,
    ):
        root = Path(root_value)
        directory_names[:] = sorted(
            name for name in directory_names if not (root / name).is_symlink()
        )
        file_names.sort()
        for file_name in file_names:
            if not file_name.endswith("-outputSummary.json"):
                continue
            source = root / file_name
            try:
                source_stat = source.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(source_stat.st_mode):
                continue
            resolved_source = source.resolve()
            if resolved_source == resolved_outfile:
                continue
            if not _is_relative_to(resolved_source, outputs):
                raise ValueError(f"JSON source resolves outside outputs directory: {source}")
            snapshots.append(
                _snapshot_path_source(
                    source,
                    resolved_source,
                    outputs,
                    snapshot_directory / f"{len(snapshots):08d}.json",
                )
            )
    snapshots.sort(key=lambda item: item.source.as_posix())
    return snapshots


def _snapshot_path_source(
    source: Path,
    resolved_source: Path,
    outputs: Path,
    snapshot: Path,
) -> _JsonSourceSnapshot:
    try:
        descriptor = os.open(source, _source_open_flags())
    except OSError as error:
        raise ValueError(f"Could not safely open JSON file {source}: {error}") from error
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = source.lstat()
        current_resolved_source = source.resolve()
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or stat.S_ISLNK(path_stat.st_mode)
            or not _same_file_identity(descriptor_stat, path_stat)
            or current_resolved_source != resolved_source
            or not _is_relative_to(current_resolved_source, outputs)
        ):
            raise ValueError(f"JSON source changed while opening: {source}")
        _copy_stable_regular_file(descriptor, source, snapshot)
    finally:
        os.close(descriptor)
    return _JsonSourceSnapshot(source, snapshot)


def _source_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _copy_stable_regular_file(descriptor: int, source: Path, snapshot: Path) -> None:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"JSON source is not a regular file: {source}")

    output_descriptor = os.open(
        snapshot,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(output_descriptor, "wb") as output_stream:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                output_stream.write(chunk)
    except BaseException:
        snapshot.unlink(missing_ok=True)
        raise

    after = os.fstat(descriptor)
    if _file_version(before) != _file_version(after):
        snapshot.unlink(missing_ok=True)
        raise ValueError(f"JSON source changed while snapshotting: {source}")


def _file_version(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _raise_walk_error(error: OSError) -> None:
    raise error


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _plan_json_collation(
    file_list: Sequence[_JsonSourceSnapshot], schema_policy: str
) -> _JsonCollationPlan:
    columns: list[str] = []
    seen_columns: set[str] = set()
    schemas: list[Tuple[str, ...]] = []
    schema_ids: Dict[Tuple[str, ...], int] = {}
    planned_files: list[Tuple[_JsonSourceSnapshot, int]] = []
    reference_columns: Optional[set[str]] = None

    for file in file_list:
        record = _read_json_record(file.snapshot, source=file.source)
        schema = tuple(record)
        schema_id = schema_ids.get(schema)
        if schema_id is None:
            schema_id = len(schemas)
            schemas.append(schema)
            schema_ids[schema] = schema_id
            if schema_policy == "strict":
                record_columns = set(schema)
                if reference_columns is None:
                    reference_columns = record_columns
                elif record_columns != reference_columns:
                    missing = sorted(reference_columns - record_columns)
                    unexpected = sorted(record_columns - reference_columns)
                    raise ValueError(
                        f"JSON schema differs under strict policy for {file.source}: "
                        f"missing={missing}, unexpected={unexpected}"
                    )
        planned_files.append((file, schema_id))
        for column in schema:
            if column not in seen_columns:
                seen_columns.add(column)
                columns.append(column)

    return _JsonCollationPlan(tuple(columns), tuple(schemas), tuple(planned_files))


def _read_json_record(
    file: Path,
    source: Optional[Path] = None,
) -> OrderedDict[str, Any]:
    display_file = source if source is not None else file
    try:
        with open(file, "r", encoding="utf-8") as stream:
            record = json.load(stream, object_pairs_hook=OrderedDict)
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"Could not read JSON file {display_file}: {error}") from error
    if not isinstance(record, OrderedDict):
        raise ValueError(f"JSON output summary must be an object: {display_file}")
    _validate_json_keys(record, display_file)
    return record


def _validate_json_keys(record: OrderedDict[str, Any], file: Path) -> None:
    invalid_keys = [key for key in record if not isinstance(key, str) or not key.strip()]
    if invalid_keys:
        raise ValueError(f"JSON output summary has blank column names: {file}")


def _write_json_collation(plan: _JsonCollationPlan, outfile: Path) -> None:
    with open(outfile, "w", encoding="utf-8", newline="") as output_stream:
        writer = csv.writer(output_stream, lineterminator="\r\n")
        writer.writerow(_spreadsheet_safe_columns(plan.columns))
        for file, schema_id in tqdm(plan.files):
            record = _read_json_record(file.snapshot, source=file.source)
            if tuple(record) != plan.schemas[schema_id]:
                raise ValueError(f"JSON snapshot schema changed during collation: {file.source}")
            writer.writerow(
                [_normalize_csv_cell(record.get(column)) for column in plan.columns]
            )
        output_stream.flush()


def _spreadsheet_safe_columns(columns: Sequence[str]) -> tuple[str, ...]:
    safe_columns = tuple(_escape_spreadsheet_text(column) for column in columns)
    if len(set(safe_columns)) != len(safe_columns):
        collisions = sorted(
            safe_column
            for safe_column in set(safe_columns)
            if safe_columns.count(safe_column) > 1
        )
        raise ValueError(
            f"JSON column names collide after spreadsheet-safe escaping: {collisions}"
        )
    return safe_columns


def _normalize_csv_cell(value: Any) -> Any:
    value = convert_ordereddict(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        return _escape_spreadsheet_text(value)
    return value


def _escape_spreadsheet_text(value: str) -> str:
    if value.startswith(_SPREADSHEET_FORMULA_PREFIXES):
        return f"'{value}"
    return value


def _create_staged_output(outfile: Path) -> Tuple[Path, int]:
    temporary_directory = Path(
        tempfile.mkdtemp(
            dir=outfile.parent,
            prefix=f".{outfile.name}.",
            suffix=".tmp",
        )
    )
    temporary_path = temporary_directory / outfile.name
    try:
        temporary_path.touch(mode=0o666, exist_ok=False)
        if outfile.exists():
            final_mode = stat.S_IMODE(outfile.stat().st_mode)
        else:
            final_mode = stat.S_IMODE(temporary_path.stat().st_mode)
        os.chmod(temporary_path, final_mode | stat.S_IWUSR)
    except BaseException:
        _cleanup_staged_output(temporary_path)
        raise
    return temporary_path, final_mode


def _validate_output_path(outfile: Path) -> None:
    try:
        output_mode = outfile.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(output_mode):
        raise ValueError(f"Output path must not be a symlink: {outfile}")
    if not stat.S_ISREG(output_mode):
        raise ValueError(f"Output path is not a regular file: {outfile}")


def _cleanup_staged_output(temporary_path: Path) -> None:
    try:
        temporary_path.unlink(missing_ok=True)
    except OSError as error:
        _LOGGER.warning("Could not remove staged output %s: %s", temporary_path, error)
    try:
        temporary_path.parent.rmdir()
    except OSError as error:
        _LOGGER.warning(
            "Could not remove staging directory %s: %s",
            temporary_path.parent,
            error,
        )


def _cleanup_snapshot_directory(snapshot_directory: Path) -> None:
    try:
        children = list(snapshot_directory.iterdir())
    except FileNotFoundError:
        return
    except OSError as error:
        _LOGGER.warning(
            "Could not inspect source snapshot directory %s: %s",
            snapshot_directory,
            error,
        )
        return

    for child in children:
        try:
            child.unlink()
        except OSError as error:
            _LOGGER.warning("Could not remove source snapshot %s: %s", child, error)
    try:
        snapshot_directory.rmdir()
    except OSError as error:
        _LOGGER.warning(
            "Could not remove source snapshot directory %s: %s",
            snapshot_directory,
            error,
        )


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _sync_file(file: Path) -> None:
    descriptor = os.open(file, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _output_lock(outfile: Path) -> Iterator[None]:
    lock_path = _output_lock_path(outfile)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o666)
    try:
        _validate_lock_file(lock_path, descriptor)
        _lock_descriptor(descriptor)
        try:
            yield
        finally:
            _unlock_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _output_lock_path(outfile: Path) -> Path:
    canonical_parent = outfile.parent.resolve()
    lock_directory = canonical_parent / _LOCK_DIRECTORY
    lock_directory.mkdir(mode=0o700, exist_ok=True)
    lock_directory_stat = lock_directory.lstat()
    if not stat.S_ISDIR(lock_directory_stat.st_mode) or stat.S_ISLNK(
        lock_directory_stat.st_mode
    ):
        raise RuntimeError(
            f"Collation lock directory is not a directory: {lock_directory}"
        )
    return lock_directory / outfile.name


def _validate_lock_file(lock_path: Path, descriptor: int) -> None:
    descriptor_stat = os.fstat(descriptor)
    path_stat = lock_path.lstat()
    if not stat.S_ISREG(descriptor_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise RuntimeError(f"Collation lock is not a regular file: {lock_path}")
    if descriptor_stat.st_dev != path_stat.st_dev or descriptor_stat.st_ino != path_stat.st_ino:
        raise RuntimeError(f"Collation lock changed while opening: {lock_path}")


def _lock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        while True:
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                vars(msvcrt)["locking"](
                    descriptor,
                    vars(msvcrt)["LK_NBLCK"],
                    1,
                )
                return
            except OSError as error:
                if error.errno != errno.EACCES:
                    raise
                time.sleep(0.1)

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX)


def _unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        vars(msvcrt)["locking"](descriptor, vars(msvcrt)["LK_UNLCK"], 1)
        return

    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _validate_schema_policy(schema_policy: str) -> None:
    if schema_policy not in {"union", "strict"}:
        raise ValueError(f"Unknown schema policy {schema_policy!r}; expected 'union' or 'strict'")


def convert_ordereddict(value: Any) -> Any:
    if isinstance(value, OrderedDict):
        return dict(value)
    return value


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", help="Directory containing JSON files.")
    parser.add_argument(
        "--outfile",
        "-o",
        default="collated-outputs/outputs.csv",
        help="Output CSV filename (default: collated-outputs/outputs.csv).",
    )
    parser.add_argument(
        "--schema-policy",
        choices=["union", "strict"],
        default="union",
        help="How to handle differing output-summary keys.",
    )
    args = parser.parse_args(argv)

    collate_outputs(
        outputs=args.outputs,
        outfile=args.outfile,
        schema_policy=args.schema_policy,
    )


if __name__ == "__main__":
    main()
