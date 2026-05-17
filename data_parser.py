"""Parse raw seed experiment .txt files into structured DataFrames."""

import os
import re
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Iterable


def variant_display_sort_key(name: str) -> Tuple[Tuple[int, object], ...]:
    """Sort key for variant labels (display only): numeric runs compare as integers."""
    parts = re.split(r"(\d+)", name)
    key: List[Tuple[int, object]] = []
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p.casefold()))
    return tuple(key)


def sorted_variants(names: Iterable[str]) -> List[str]:
    """Unique variant names in deterministic natural-sorted order (for tables/plots)."""
    return sorted(dict.fromkeys(names), key=variant_display_sort_key)


@dataclass
class FileData:
    """Parsed data from a single experiment file."""
    filepath: str
    filename: str
    variant: str
    replicate: int
    seeds: List[Dict]  # each dict: {"ver": int, "roots": [int, ...]}
    failed_count: int

    @property
    def label(self) -> str:
        return f"{self.variant}_{self.replicate}"


@dataclass(frozen=True)
class GerminationStats:
    """Alive / dead seed counts for one input file (display and Всхожесть).

    Alive: rows parsed as seed measurements. Dead: trailing ``0`` lines and/or
    ``0=N`` in the file (same rules as ``failed_count``).
    """

    alive: int
    dead: int
    total: int

    @property
    def rate(self) -> float:
        if self.total <= 0:
            return float("nan")
        return self.alive / self.total


def germination_stats(file_data: FileData) -> GerminationStats:
    """Counts for Всхожесть: alive / (alive + dead)."""
    alive = len(file_data.seeds)
    dead = file_data.failed_count
    return GerminationStats(alive=alive, dead=dead, total=alive + dead)


def _detect_variant_simple(base: str) -> Tuple[str, int]:
    """Short filenames: 3V.txt, 3V no2.txt, Control _no2.txt, Control in TANKno2.txt."""
    s = base.strip()
    # Spaced replicate markers: "3V no2", "Control _no2"
    m = re.search(r"\s+_?no(\d+)\s*$", s, re.IGNORECASE)
    if m:
        return s[: m.start()].strip(), int(m.group(1))
    # Glued "...TANKno2", "3Vno2"
    m = re.search(r"^(.+?)(no\d+)\s*$", s, re.IGNORECASE)
    if m:
        suf = m.group(2).lower()
        if suf.startswith("no") and m.group(1).strip():
            rep_str = suf[2:]
            if rep_str.isdigit():
                return m.group(1).strip(), int(rep_str)
    # Trailing _2 (variant_2)
    m = re.match(r"^(.+)_(\d+)$", s)
    if m and m.group(1):
        return m.group(1), int(m.group(2))
    return s, 1


def _detect_variant_and_replicate(filename: str) -> Tuple[str, int]:
    """Extract variant name and replicate number from filename."""
    base = os.path.splitext(filename)[0]
    parts = base.split("_")

    # Dated / ОЗП experiment naming (e.g. 24.11.25_ОЗП-21_5sm_10min_1)
    if "ОЗП" in base or "OZP" in base.upper():
        suffix_parts = []
        found_experiment = False
        for p in parts:
            if found_experiment:
                suffix_parts.append(p)
            elif "ОЗП" in p or "OZP" in p.upper():
                found_experiment = True

        if not suffix_parts:
            suffix_parts = parts[2:]

        suffix = "_".join(suffix_parts)

        k_match = re.match(r"^[КкKk](\d+)$", suffix)
        if k_match:
            return "К", int(k_match.group(1))

        num_match = re.match(r"^(.+?)_(\d+)$", suffix)
        if num_match:
            return num_match.group(1), int(num_match.group(2))

        return (suffix or base.strip()), 1

    return _detect_variant_simple(base)


def _split_numeric_tokens(line: str) -> Optional[List[int]]:
    """Split on whitespace or tabs; return ints if every token is an integer."""
    parts = re.split(r"[\t ]+", line.strip())
    parts = [p for p in parts if p]
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def parse_file(filepath: str, *, display_name: Optional[str] = None) -> FileData:
    """Parse a single raw data file."""
    filename = display_name if display_name else os.path.basename(filepath)
    variant, replicate = _detect_variant_and_replicate(os.path.basename(filepath))

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()

    seeds: List[Dict] = []
    explicit_failed: Optional[int] = None
    line_zero_failures = 0
    skipped_leading_title = False

    for line in lines:
        s = line.strip()
        if not s:
            continue

        zero_match = re.match(r"^0=(\d+)$", s)
        if zero_match:
            explicit_failed = int(zero_match.group(1))
            continue

        if s == "0":
            line_zero_failures += 1
            continue

        nums = _split_numeric_tokens(s)
        if nums is not None and len(nums) >= 1:
            seeds.append({
                "ver": nums[0],
                "roots": nums[1:] if len(nums) > 1 else [],
            })
            continue

        # Non-numeric line: skip at most one leading title row (e.g. "3V", "Control Rye...")
        if not skipped_leading_title and not seeds and explicit_failed is None and line_zero_failures == 0:
            skipped_leading_title = True
            continue

    failed_count = explicit_failed if explicit_failed is not None else line_zero_failures

    return FileData(
        filepath=filepath,
        filename=filename,
        variant=variant,
        replicate=replicate,
        seeds=seeds,
        failed_count=failed_count,
    )


def build_seed_dataframe(file_data: FileData, start_id: int = 1) -> pd.DataFrame:
    """Build a DataFrame for one file with all computed columns (report display).

    Root lengths in kor01..korNN are sorted longest→shortest per seed for readability only;
    sm.kor and mn.kor match the unsorted data because sum and mean are invariant.

    Returns DataFrame with columns: id, zar, ver, kor01..korNN, sm.kor, mn.kor, rs.rat
    Empty rows appended for failed seeds.
    """
    max_roots = max((len(s["roots"]) for s in file_data.seeds), default=0)

    rows = []
    for i, seed in enumerate(file_data.seeds):
        root_vals = sorted(int(r) for r in seed["roots"])
        root_vals.reverse()
        row = {
            "id": start_id + i,
            "zar": 0,
            "ver": seed["ver"],
        }
        for j in range(max_roots):
            col = f"kor{j + 1:02d}"
            if j < len(root_vals):
                row[col] = root_vals[j]
            else:
                row[col] = np.nan

        row["sm.kor"] = sum(root_vals) if root_vals else 0
        row["mn.kor"] = round(np.mean(root_vals)) if root_vals else 0
        row["rs.rat"] = round(row["sm.kor"] / seed["ver"], 2) if seed["ver"] != 0 else np.nan

        rows.append(row)

    # Append empty rows for failed seeds
    next_id = start_id + len(file_data.seeds)
    for i in range(file_data.failed_count):
        row = {"id": next_id + i}
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def discover_files(raw_dir: str) -> List[str]:
    """Find all .txt files under raw_dir (including subfolders)."""
    raw_dir = os.path.abspath(raw_dir)
    pattern = os.path.join(raw_dir, "**", "*.txt")
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def parse_all(raw_dir: str) -> List[FileData]:
    """Parse all files in the raw data directory tree."""
    raw_abs = os.path.abspath(raw_dir)
    filepaths = discover_files(raw_abs)
    file_datas = [
        parse_file(fp, display_name=os.path.relpath(fp, raw_abs))
        for fp in filepaths
    ]

    # Sort by display variant order then replicate (tables / listing only)
    file_datas.sort(key=lambda fd: (variant_display_sort_key(fd.variant), fd.replicate))
    return file_datas


def get_variant_groups(file_datas: List[FileData]) -> Dict[str, List[FileData]]:
    """Group FileData by variant name."""
    groups: Dict[str, List[FileData]] = {}
    for fd in file_datas:
        groups.setdefault(fd.variant, []).append(fd)
    for v in groups:
        groups[v].sort(key=lambda fd: fd.replicate)
    return groups


def build_analysis_dataframe(file_datas: List[FileData]) -> pd.DataFrame:
    """Build a flat DataFrame with all seeds from all files, suitable for analysis.

    Columns: variant, replicate, file_label, ver, sm.kor, mn.kor, rs.rat
    Only includes seeds that actually grew (excludes failed/empty rows).
    """
    rows = []
    for fd in file_datas:
        for seed in fd.seeds:
            root_vals = seed["roots"]
            sm_kor = sum(root_vals) if root_vals else 0
            mn_kor = round(np.mean(root_vals)) if root_vals else 0
            ver = seed["ver"]
            rs_rat = round(sm_kor / ver, 2) if ver != 0 else np.nan

            rows.append({
                "variant": fd.variant,
                "replicate": fd.replicate,
                "file_label": fd.label,
                "ver": ver,
                "sm.kor": sm_kor,
                "mn.kor": mn_kor,
                "rs.rat": rs_rat,
            })

    return pd.DataFrame(rows)
