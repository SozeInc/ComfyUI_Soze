import csv
import hashlib
import logging
import os
import random
from pathlib import Path

import folder_paths

from .status_utils import EventLog, push_node_status, finalize_status

logger = logging.getLogger(__name__)


def _row_preview(row: list[str], max_len: int = 80) -> str:
    """Compact preview of a row for status display."""
    joined = " | ".join(str(c).strip() for c in row[:5] if str(c).strip())
    if len(joined) > max_len:
        joined = joined[: max_len - 1] + "…"
    return joined

_CSV_DIR = (Path(__file__).parent / "csv_files").resolve()


def _looks_absolute(raw: str) -> bool:
    """True if the string looks like an absolute path on any platform.

    `Path.is_absolute()` is platform-specific (e.g. WindowsPath rejects
    forward-slash POSIX paths). We accept POSIX absolutes, Windows drive-letter
    paths, and UNC shares too.
    """
    if not raw:
        return False
    if raw.startswith("/") or raw.startswith("\\\\"):
        return True
    if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        return True
    try:
        return Path(raw).is_absolute()
    except Exception:
        return False


def _resolve_csv_path(csv_filename_path: str) -> Path:
    """Resolve a user-supplied CSV path.

    - Absolute paths (POSIX, Windows drive-letter, UNC) are accepted as-is.
    - Anything that already exists as a file on disk is accepted as-is too.
    - Otherwise the path resolves under the bundled csv_files dir, and `..`
      escape attempts are rejected.
    """
    raw = csv_filename_path.strip()

    # 1) Looks absolute? Take it as-is.
    if _looks_absolute(raw):
        candidate = Path(raw).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"CSV file not found: {candidate}")
        return candidate

    # 2) Exists as-given (e.g. a relative path from CWD)? Take it.
    direct = Path(raw)
    if direct.is_file():
        return direct.resolve()

    # 3) Fall back to the bundled csv_files dir.
    candidate = (_CSV_DIR / raw).resolve()
    try:
        candidate.relative_to(_CSV_DIR)
    except ValueError:
        raise ValueError(
            f"CSV path escapes csv_files directory: {csv_filename_path!r}"
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"CSV file not found: {candidate}")
    return candidate


def _read_csv_lines(csv_path: Path) -> list[str]:
    """Read CSV with utf-8-sig, falling back to windows-1252."""
    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            return f.readlines()
    except UnicodeDecodeError:
        with open(csv_path, "r", newline="", encoding="windows-1252") as f:
            return f.readlines()


def _file_signature(csv_path: Path) -> str:
    """Cheap signature for IS_CHANGED that invalidates only when the file changes."""
    st = csv_path.stat()
    return f"{csv_path}:{st.st_mtime_ns}:{st.st_size}"


def _load_rows(csv_filename_path: str, csv_text: str = "") -> list[list[str]]:
    if csv_text.strip():
        csv_data = csv_text.splitlines()
    elif csv_filename_path.strip():
        csv_path = _resolve_csv_path(csv_filename_path)
        csv_data = _read_csv_lines(csv_path)
    else:
        return []
    return list(csv.reader(csv_data))


def _row_to_outputs(row: list[str]) -> tuple[list[str], str]:
    output = list(row[:10]) + [""] * (10 - len(row))
    entire_line = ",".join(row)
    return output, entire_line


class Soze_CSVReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_filename_path": ("STRING", {"default": "", "multiline": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1}),
            },
            "optional": {
                "csv_text": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_NAMES = ('Column_1', 'Column_2', 'Column_3', 'Column_4', 'Column_5', 'Column_6', 'Column_7', 'Column_8', 'Column_9', 'Column_10', 'Entire_Line', 'Row_Count', 'status')
    RETURN_TYPES = ("STRING",) * 11 + ("INT", "STRING")
    FUNCTION = "read_csv"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, csv_filename_path, index, csv_text="", unique_id=None):
        if csv_text.strip():
            return f"text:{hashlib.sha1(csv_text.encode('utf-8')).hexdigest()}:{index}"
        if csv_filename_path.strip():
            try:
                return f"{_file_signature(_resolve_csv_path(csv_filename_path))}:{index}"
            except (FileNotFoundError, ValueError):
                return f"missing:{csv_filename_path}:{index}"
        return f"empty:{index}"

    def read_csv(self, csv_filename_path, index, csv_text="", unique_id=None):
        log = EventLog()
        source = "inline csv_text" if csv_text.strip() else (csv_filename_path or "(empty)")
        push_node_status(unique_id, f"Source: {source}", log)
        rows = _load_rows(csv_filename_path, csv_text)
        row_count = len(rows)
        if row_count == 0:
            headline = "Empty: 0 rows."
            push_node_status(unique_id, headline, log)
            return tuple([""] * 11 + [0, finalize_status(headline, log)])
        if index < 0:
            push_node_status(unique_id, f"ERROR: index must be >= 0, got {index}", log)
            raise ValueError(f"index must be >= 0, got {index}")
        if index >= row_count:
            push_node_status(unique_id, f"ERROR: index {index} >= row_count {row_count}", log)
            raise ValueError(f"There are no more rows in the CSV file ({row_count})")
        output, entire_line = _row_to_outputs(rows[index])
        headline = f"OK: row {index+1}/{row_count} — {_row_preview(rows[index])}"
        push_node_status(unique_id, headline, log)
        return tuple(output + [entire_line, row_count, finalize_status(headline, log)])


class Soze_CSVReaderXCheckpoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_filename_path": ("STRING", {"default": "", "multiline": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1, "tooltip": "The row number to read from the CSV file."}),
                "start_ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "The name of the starting checkpoint."}),
                "ckpt_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "The number of checkpoints to iterate."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_NAMES = ('Column_1', 'Column_2', 'Column_3', 'Column_4', 'Column_5', 'Column_6', 'Column_7', 'Column_8', 'Column_9', 'Column_10', 'Entire_Line', 'Row_Count', "Ckpt_Full_Path", "Ckpt_Name_Only", "Cktp_Index", "status")
    RETURN_TYPES = ("STRING",) * 11 + ("INT", "STRING", "STRING", "INT", "STRING")
    FUNCTION = "process"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, csv_filename_path, index, start_ckpt_name, ckpt_count, unique_id=None):
        try:
            sig = _file_signature(_resolve_csv_path(csv_filename_path))
        except (FileNotFoundError, ValueError):
            sig = f"missing:{csv_filename_path}"
        return f"{sig}:{index}:{start_ckpt_name}:{ckpt_count}"

    def process(self, csv_filename_path, index, start_ckpt_name, ckpt_count, unique_id=None):
        log = EventLog()
        if not csv_filename_path.strip():
            push_node_status(unique_id, "ERROR: CSV filename path is empty.", log)
            raise ValueError("CSV filename path cannot be empty.")

        ckpt_list = folder_paths.get_filename_list("checkpoints")
        try:
            start_ckpt_index = ckpt_list.index(start_ckpt_name)
        except ValueError:
            push_node_status(unique_id, f"ERROR: checkpoint '{start_ckpt_name}' not found.", log)
            raise ValueError(f"Checkpoint '{start_ckpt_name}' not found in checkpoint list.")

        rows = _load_rows(csv_filename_path)
        row_count = len(rows)
        if row_count == 0:
            push_node_status(unique_id, "ERROR: CSV is empty.", log)
            raise ValueError("CSV file contains no rows.")

        csv_index = index % row_count
        ckpt_index = start_ckpt_index + (index // row_count)

        if ckpt_index >= len(ckpt_list):
            push_node_status(unique_id, f"ERROR: ckpt_index {ckpt_index} >= ckpt_list size {len(ckpt_list)}", log)
            raise ValueError(f"There are no more checkpoints in the list ({len(ckpt_list)})")
        if ckpt_index >= (start_ckpt_index + ckpt_count):
            push_node_status(unique_id, f"ERROR: iteration complete after {row_count} rows × {ckpt_count} ckpts.", log)
            raise ValueError(f"Index {index} has completed the iteration of rows {row_count} against each checkpoint indicated {ckpt_count}.")

        output, entire_line = _row_to_outputs(rows[csv_index])
        ckpt_full_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_list[ckpt_index])
        ckpt_name_only = os.path.basename(ckpt_list[ckpt_index])
        headline = (
            f"OK: row {csv_index+1}/{row_count}, ckpt {ckpt_index+1}/{len(ckpt_list)} ({ckpt_name_only}) "
            f"— {_row_preview(rows[csv_index])}"
        )
        push_node_status(unique_id, headline, log)
        return tuple(output + [entire_line, row_count, ckpt_full_path, ckpt_name_only, ckpt_index + 1, finalize_status(headline, log)])


class Soze_CSVReaderXLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_filename_path": ("STRING", {"default": "", "multiline": True}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "control_after_generate": True, "step": 1, "tooltip": "The row number to read from the CSV file."}),
                "start_lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the starting LoRA."}),
                "lora_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1, "tooltip": "The number of LoRAs to iterate."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_NAMES = ('Column_1', 'Column_2', 'Column_3', 'Column_4', 'Column_5', 'Column_6', 'Column_7', 'Column_8', 'Column_9', 'Column_10', 'Entire_Line', 'Row_Count', "Lora_Full_Path", "Lora_Name_Only", "Lora_Index", "status")
    RETURN_TYPES = ("STRING",) * 11 + ("INT", "STRING", "STRING", "INT", "STRING")
    FUNCTION = "process"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, csv_filename_path, index, start_lora_name, lora_count, unique_id=None):
        try:
            sig = _file_signature(_resolve_csv_path(csv_filename_path))
        except (FileNotFoundError, ValueError):
            sig = f"missing:{csv_filename_path}"
        return f"{sig}:{index}:{start_lora_name}:{lora_count}"

    def process(self, csv_filename_path, index, start_lora_name, lora_count, unique_id=None):
        log = EventLog()
        if not csv_filename_path.strip():
            push_node_status(unique_id, "ERROR: CSV filename path is empty.", log)
            raise ValueError("CSV filename path cannot be empty.")

        lora_list = folder_paths.get_filename_list("loras")
        try:
            start_lora_index = lora_list.index(start_lora_name)
        except ValueError:
            push_node_status(unique_id, f"ERROR: lora '{start_lora_name}' not found.", log)
            raise ValueError(f"Lora '{start_lora_name}' not found in lora list.")

        rows = _load_rows(csv_filename_path)
        row_count = len(rows)
        if row_count == 0:
            push_node_status(unique_id, "ERROR: CSV is empty.", log)
            raise ValueError("CSV file contains no rows.")

        csv_index = index % row_count
        lora_index = start_lora_index + (index // row_count)

        if lora_index >= len(lora_list):
            push_node_status(unique_id, f"ERROR: lora_index {lora_index} >= lora_list size {len(lora_list)}", log)
            raise ValueError(f"There are no more lora in the list ({len(lora_list)})")
        if lora_index >= (start_lora_index + lora_count):
            push_node_status(unique_id, f"ERROR: iteration complete after {row_count} rows × {lora_count} loras.", log)
            raise ValueError(f"Index {index} has completed the iteration of rows {row_count} against each lora indicated {lora_count}.")

        output, entire_line = _row_to_outputs(rows[csv_index])
        lora_full_path = folder_paths.get_full_path_or_raise("loras", lora_list[lora_index])
        lora_name_only = os.path.basename(lora_list[lora_index])
        headline = (
            f"OK: row {csv_index+1}/{row_count}, lora {lora_index+1}/{len(lora_list)} ({lora_name_only}) "
            f"— {_row_preview(rows[csv_index])}"
        )
        push_node_status(unique_id, headline, log)
        return tuple(output + [entire_line, row_count, lora_full_path, lora_name_only, lora_index + 1, finalize_status(headline, log)])


class Soze_CSVRandomReader:
    """Pick N random rows from a CSV file (any path) or inline csv_text.

    - `csv_filename_path` accepts absolute paths anywhere on disk (POSIX,
      Windows drive-letter, or UNC). Relative paths fall back to the bundled
      py/csv_files directory.
    - If `csv_text` is non-empty, it takes precedence over the file path.
    - With `seed > 0` the picks are deterministic and the node is cacheable.
      With `seed == 0` the node re-runs every prompt for a fresh draw.
    - `allow_repeats=False` uses random.sample (unique picks, clamped to row
      count). `allow_repeats=True` uses random.choice each time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_filename_path": ("STRING", {"default": "", "multiline": True, "tooltip": "Absolute path or path relative to py/csv_files. Ignored if csv_text is non-empty."}),
                "num_rows": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1, "tooltip": "How many rows to randomly pick."}),
            },
            "optional": {
                "csv_text": ("STRING", {"default": "", "multiline": True, "tooltip": "Inline CSV. Takes precedence over csv_filename_path when non-empty."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True, "tooltip": "0 = nondeterministic; any other value seeds the RNG for reproducible picks."}),
                "allow_repeats": ("BOOLEAN", {"default": False, "tooltip": "If True, the same row may be picked more than once when num_rows exceeds available rows."}),
                "skip_header": ("BOOLEAN", {"default": False, "tooltip": "Skip the first row (treat it as a header)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_NAMES = (
        'Column_1', 'Column_2', 'Column_3', 'Column_4', 'Column_5',
        'Column_6', 'Column_7', 'Column_8', 'Column_9', 'Column_10',
        'First_Row_Line', 'Selected_Lines', 'Selected_Count', 'Total_Rows', 'status',
    )
    RETURN_TYPES = ("STRING",) * 11 + ("STRING", "INT", "INT", "STRING")
    FUNCTION = "read_random_rows"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, csv_filename_path, num_rows, csv_text="", seed=0, allow_repeats=False, skip_header=False, unique_id=None):
        # Deterministic when seed > 0 → cacheable on (source, knobs).
        if seed and seed != 0:
            if csv_text and csv_text.strip():
                base = "text:" + hashlib.sha1(csv_text.encode("utf-8")).hexdigest()
            elif csv_filename_path and csv_filename_path.strip():
                try:
                    base = _file_signature(_resolve_csv_path(csv_filename_path))
                except (FileNotFoundError, ValueError):
                    base = f"missing:{csv_filename_path}"
            else:
                base = "empty"
            return f"{base}|n={num_rows}|seed={seed}|repeats={allow_repeats}|skip_header={skip_header}"
        return float("NaN")

    def read_random_rows(self, csv_filename_path, num_rows, csv_text="", seed=0,
                         allow_repeats=False, skip_header=False, unique_id=None):
        log = EventLog()
        source = "inline csv_text" if csv_text and csv_text.strip() else (csv_filename_path or "(empty)")
        push_node_status(unique_id, f"Source: {source}", log)

        try:
            rows = _load_rows(csv_filename_path, csv_text)
        except Exception as e:
            push_node_status(unique_id, f"ERROR loading CSV: {e!r}", log)
            raise

        if skip_header and rows:
            push_node_status(unique_id, "Skipping header row.", log)
            rows = rows[1:]

        total = len(rows)
        push_node_status(unique_id, f"Loaded {total} row(s) (after header skip={skip_header}).", log)

        if total == 0:
            headline = "Empty: 0 rows after loading."
            push_node_status(unique_id, headline, log)
            return tuple([""] * 11 + ["", 0, 0, finalize_status(headline, log)])

        rng = random.Random(seed) if seed and seed != 0 else random.Random()

        if allow_repeats:
            picks = [rng.choice(rows) for _ in range(num_rows)]
        else:
            n = min(num_rows, total)
            if num_rows > total:
                push_node_status(unique_id, f"Note: requested {num_rows} but only {total} unique rows available; returning {n}.", log)
            picks = rng.sample(rows, n)

        first = picks[0]
        first_outputs, first_line = _row_to_outputs(first)

        # Selected_Lines: each picked row joined with commas, rows separated by newlines.
        selected_lines = "\n".join(",".join(r) for r in picks)

        headline = (
            f"OK: picked {len(picks)} of {total} row(s) "
            f"(seed={seed if seed else 'random'}, repeats={'on' if allow_repeats else 'off'}); "
            f"first={_row_preview(first)}"
        )
        push_node_status(unique_id, headline, log)
        return tuple(first_outputs + [first_line, selected_lines, len(picks), total, finalize_status(headline, log)])
