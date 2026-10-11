# -*- coding: utf-8 -*-
"""Batch-prepare Napari cell CSVs for every animal/channel in the overview.

Run this file in Spyder or with: python data_prep.py
Requires pandas, numpy and openpyxl (conda install pandas numpy openpyxl).
Source CSVs and the spreadsheet are only read; all writes go to OUTPUT_DIR.
"""

from pathlib import Path
from datetime import datetime
import re

import numpy as np
import pandas as pd


# --------------------------- SETTINGS ---------------------------------
BASE_DIR = Path(r"L:\dmclab\Joana\Tracing")
OUTPUT_DIR = BASE_DIR / "Analysis" / "cells_prep"
# Put the spreadsheet beside this script, or replace this with its full path.
OVERVIEW_FILE = Path(r"L:\dmclab\Joana\Tracing\tracing_overview.xlsx")

# Optional folder-name corrections if the disk uses different strain names.
STRAIN_FOLDERS = {}  # Example: {"Fmr1-Tlx3": "Fmr1_Tlx3"}

# Only needed when a channel folder contains several candidate cell CSVs.
# Keys: (animal ID as a string, channel); values: exact source CSV paths.
SOURCE_OVERRIDES = {}
# Example:
# SOURCE_OVERRIDES = {
#     ("848599", "green"): BASE_DIR / "Tlx3" / "Cohort 1" / "848599"
#         / "results" / "cells" / "green" / "my_cells.csv",
# }

# Rebuild every prepared output from its original raw CSV with corrected layers.
# After this full rerun, set False to process only new injections on future runs.
# Original source CSVs are always read-only.
OVERWRITE_OUTPUTS = True

CELL_COLUMNS = [
    "acronym", "structure_id", "ap_mm", "dv_mm", "ml_mm",
    "ap_coords", "dv_coords", "ml_coords",
]
REQUIRED_COLUMNS = set(CELL_COLUMNS + ["name"])
OUTPUT_COLUMNS = [
    "animal_id", "acronym", "region", "part", "layer", "hemisphere",
    "structure_id", "ap_mm", "dv_mm", "ml_mm", "ap_coords", "dv_coords",
    "ml_coords", "strain", "cohort", "channel", "sex",
]


def clean(value):
    """Normalize Excel text and numeric IDs (848599.0 -> 848599)."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return re.sub(r"^(\d+)\.0+$", r"\1", text)


def path_component(value):
    if not value or value in {".", ".."} or re.search(r'[<>:"/\\|?*]', value):
        raise ValueError(f"Invalid folder/file component: {value!r}")
    return value


def read_jobs(workbook):
    """Read each side-by-side Animal...Channel table independently.

    Carry animal metadata onto continuation rows, but never carry channels
    or injection properties forward. Reset metadata at every new animal.
    """
    jobs = {}
    sheets = pd.read_excel(workbook, sheet_name=None, header=None, dtype=object)
    for sheet, table in sheets.items():
        headers = [i for i in table.index
                   if any(clean(v).lower() == "animal" for v in table.loc[i])]
        if not headers:
            continue
        if len(headers) != 1:
            raise ValueError(f"{sheet}: expected one header row containing Animal.")
        header_row = headers[0]
        header = [clean(v).lower() for v in table.loc[header_row]]
        starts = [i for i, v in enumerate(header) if v == "animal"]
        for start, end in zip(starts, starts[1:] + [len(header)]):
            columns = {header[i]: i for i in range(start, end) if header[i]}
            missing = {"animal", "strain", "cohort", "channel"} - columns.keys()
            if missing:
                raise ValueError(f"{sheet}: missing overview columns {sorted(missing)}")
            current = None
            for row_index in range(header_row + 1, len(table)):
                row = table.iloc[row_index]
                values = {k: clean(row.iloc[c]) for k, c in columns.items()}
                if not any(values.values()):
                    current = None
                    continue
                if values["animal"]:
                    current = {k: values.get(k, "")
                               for k in ("animal", "strain", "cohort", "sex")}
                elif current:
                    for k in ("strain", "cohort", "sex"):
                        if values.get(k) and values[k] != current[k]:
                            raise ValueError(f"{sheet} row {row_index + 1}: conflicting {k}.")
                channel = values["channel"].lower()
                if not channel:
                    if values["animal"]:
                        raise ValueError(f"{sheet} row {row_index + 1}: animal has no channel.")
                    continue
                if not current or not all(current[k] for k in ("animal", "strain", "cohort")):
                    raise ValueError(f"{sheet} row {row_index + 1}: incomplete animal metadata.")
                job = dict(current, channel=channel)
                for k in ("animal", "strain", "cohort", "channel"):
                    path_component(job[k])
                key = tuple(job[k] for k in ("strain", "cohort", "animal", "channel"))
                if key in jobs and jobs[key] != job:
                    raise ValueError(f"Conflicting metadata for {key}")
                jobs[key] = job
    if not jobs:
        raise ValueError("No animal/channel entries found in the overview.")
    return list(jobs.values())


def source_folder(job):
    strain = path_component(STRAIN_FOLDERS.get(job["strain"], job["strain"]))
    return (BASE_DIR / strain / f"Cohort {job['cohort']}" / job["animal"]
            / "results" / "cells" / job["channel"])


def find_source(job):
    override = SOURCE_OVERRIDES.get((job["animal"], job["channel"]))
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"Override CSV does not exist: {path}")
        return path
    folder = source_folder(job)
    if not folder.is_dir():
        raise FileNotFoundError(f"Channel folder does not exist: {folder}")
    csvs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {folder}")
    # Use the filename convention in the original script when it exists.
    expected = f"{job['animal']}_cells_{job['channel']}.csv".lower()
    exact = [p for p in csvs if p.name.lower() == expected]
    if len(exact) == 1:
        return exact[0]
    if len(csvs) == 1:
        return csvs[0]
    candidates = []
    for path in csvs:
        try:
            columns = pd.read_csv(path, nrows=0).columns
            if REQUIRED_COLUMNS.issubset(columns):
                candidates.append(path)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Cannot select one cell CSV in {folder}. Files: "
        + ", ".join(p.name for p in csvs)
        + ". Set SOURCE_OVERRIDES for this animal/channel."
    )


def extract_layer(name, acronym):
    """Return a canonical layer and whether it was recovered from an acronym.

    Read layer/Layer anywhere in the atlas name. If no layer is present,
    accept only explicit layer suffixes on the listed PFC acronyms.
    Parent structures without a layer suffix remain unassigned.
    """
    match = re.search(r"\blayers?\s+(2\s*/\s*3|6a|6b|[1-6])\b",
                      str(name), flags=re.IGNORECASE)
    if match:
        return "layer " + re.sub(r"\s+", "", match.group(1)).lower(), False
    match = re.fullmatch(
        r"(?:FRP|PL|ILA|MOs|ORB(?:m|l|vl)?|ACA[dv]?|AI[dvp]?)(2/3|6a|6b|[1-6])",
        str(acronym).strip(), flags=re.IGNORECASE)
    if match:
        return "layer " + match.group(1).lower(), True
    return None, False


def prepare_cells(raw, job):
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Cell CSV is missing columns: {sorted(missing)}")
    data = raw.copy(deep=True)
    root = data["acronym"].astype("string").str.strip().eq("root").fillna(False)
    data = data.loc[~root].copy()
    ml = pd.to_numeric(data["ml_mm"], errors="coerce")
    if not np.isfinite(ml.to_numpy(dtype=float)).all():
        raise ValueError("Non-root cells have missing/invalid ml_mm; hemisphere cannot be assigned.")
    data["ml_mm"] = ml
    midline = data["ml_mm"].eq(0)
    stats = {"input_cells": len(raw), "removed_root": int(root.sum()),
             "removed_midline": int(midline.sum())}
    data = data.loc[~midline].copy()
    if data["name"].isna().any():
        raise ValueError("Retained cells have missing atlas names.")
    data["hemisphere"] = np.where(data["ml_mm"] > 0, "right", "left")
    # Same region/part/layer meaning as the original, with whitespace removed
    # and case-insensitive layer matching (both 'layer' and 'Layer').
    parts = data["name"].astype(str).map(lambda name: [p.strip() for p in name.split(",")])
    data["region"] = parts.map(lambda p: p[0])
    data["part"] = parts.map(lambda p: next((v for v in p[1:] if v.lower().endswith("part")), None))
    layers = [extract_layer(name, acronym)
              for name, acronym in zip(data["name"], data["acronym"])]
    data["layer"] = [layer for layer, recovered in layers]
    stats["layers_recovered_from_acronym"] = sum(recovered for layer, recovered in layers)
    stats["cells_without_layer"] = int(data["layer"].isna().sum())
    data["animal_id"] = job["animal"]
    for k in ("strain", "cohort", "channel", "sex"):
        data[k] = job[k]
    stats.update(output_cells=len(data),
                 right_cells=int(data["hemisphere"].eq("right").sum()),
                 left_cells=int(data["hemisphere"].eq("left").sum()))
    return data.loc[:, OUTPUT_COLUMNS].reset_index(drop=True), stats


def main():
    if not BASE_DIR.is_dir():
        raise FileNotFoundError(f"Tracing folder unavailable: {BASE_DIR}")
    if not OVERVIEW_FILE.is_file():
        raise FileNotFoundError(f"Set OVERVIEW_FILE to your spreadsheet location: {OVERVIEW_FILE}")
    jobs = read_jobs(OVERVIEW_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_animals = len({(j["strain"], j["cohort"], j["animal"]) for j in jobs})
    print(f"Found {n_animals} animals and {len(jobs)} animal/channel datasets.")
    summary = []
    for i, job in enumerate(jobs, 1):
        filename = f"{job['strain']}_cohort{job['cohort']}_{job['animal']}_{job['channel']}_data.csv"
        destination = OUTPUT_DIR / filename
        record = dict(job, source_folder=str(source_folder(job)),
                      source_csv="", output_csv=str(destination), status="", message="")
        try:
            source = find_source(job)
            record["source_csv"] = str(source)
            # Never let a destination resolve to an original source file.
            if source.resolve() == destination.resolve() or (destination.exists() and source.samefile(destination)):
                raise ValueError("Output is the source file; refusing to write.")
            overwrite = OVERWRITE_OUTPUTS
            if destination.exists() and not overwrite:
                record.update(status="SKIPPED_EXISTS", message="Existing processed file preserved.")
            else:
                processed, stats = prepare_cells(pd.read_csv(source), job)
                # Exclusive creation by default; no extra CSV index column.
                processed.to_csv(destination, index=False, mode="w" if overwrite else "x")
                record.update(stats, status="OK")
        except Exception as exc:
            record.update(status="ERROR", message=f"{type(exc).__name__}: {exc}")
        summary.append(record)
        print(f"[{i}/{len(jobs)}] {job['animal']} {job['channel']}: {record['status']} {record['message']}")
    report = OUTPUT_DIR / f"processing_summary_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    pd.DataFrame(summary).to_csv(report, index=False)
    counts = pd.Series([r["status"] for r in summary]).value_counts().to_dict()
    print(f"DONE! {counts}\nSummary: {report}")
    return pd.DataFrame(summary)


if __name__ == "__main__":
    main()
