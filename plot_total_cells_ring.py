"""Donut, injection-count table, and cell counts per injection by mouse line.

Run in Spyder. Requires pandas, numpy, matplotlib and openpyxl. Sources are read-only.
Only injections with overview % Striatum >= MIN_STRIATUM_PERCENT are included.
Only cells from the seven regions in PFC_REGIONS enter counts and figures.
Each retained CSV row counts once; channels are NOT deduplicated.
One animal/channel CSV = one injection. Each plotted point is one injection.
"""
from pathlib import Path
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INPUT_DIR = Path(r"L:\dmclab\Joana\Tracing\Analysis\cells_prep")
OUTPUT_DIR = Path(r"L:\dmclab\Joana\Tracing\Analysis\plots\total_cells")
# Set this to the actual full path of your overview spreadsheet.
OVERVIEW_FILE = Path(r"L:\dmclab\Joana\Tracing\tracing_overview.xlsx")
MIN_STRIATUM_PERCENT = 80.0  # Inclusive; overview values are on a 0–100 scale.
SHOW_FIGURE = True
# Optional selection, e.g. {"green"}; None includes every prepared channel.
CHANNELS = None

# Region names before the first comma; subdivisions and layers are pooled.
PFC_REGIONS = [
    "Frontal pole", "Prelimbic area", "Infralimbic area",
    "Secondary motor area", "Orbital area", "Anterior cingulate area",
    "Agranular insular area",
]


def select_pfc_cells(data):
    """Use atlas region names, preserving all parts/layers of each named area.

    Prepared CSVs have 'region'; 'name' is supported as a fallback. Matching
    ignores case, whitespace and text after the first comma. Missing region
    labels are excluded. The source dataframe is never changed.
    """
    if "region" in data.columns:
        labels = data["region"].astype("string")
        if "name" in data.columns:
            labels = labels.mask(labels.fillna("").str.strip().eq(""), data["name"])
    elif "name" in data.columns:
        labels = data["name"].astype("string")
    else:
        raise ValueError("Cell CSV has neither 'region' nor 'name'; cannot select PFC cells.")
    labels = (labels.str.split(",", n=1).str[0].str.strip()
              .str.replace(r"\s+", " ", regex=True).str.casefold())
    keep = labels.isin({name.casefold() for name in PFC_REGIONS})
    return data.loc[keep].copy()

LINE_ORDER = ["Tlx3", "Fezf2", "Fmr1-Fezf2", "Fmr1-Tlx3"]
GROUP_MAP = dict(zip(LINE_ORDER, ["Control", "Control", "Fmr1", "Fmr1"]))
GROUP_COLORS = {"Control": "#A67DBB", "Fmr1": "#1D9999"}
LINE_COLORS = {"Tlx3": "#CBB4D9", "Fezf2": "#805797",
               "Fmr1-Fezf2": "#116E70", "Fmr1-Tlx3": "#80C7C5"}


def canonical_line(value):
    key = re.sub(r"[\s_\-–—]+", "", str(value)).lower()
    aliases = {"tlx3": "Tlx3", "fezf2": "Fezf2",
               "fmr1fezf2": "Fmr1-Fezf2", "fmr1tlx3": "Fmr1-Tlx3",
               "fmr1tlx": "Fmr1-Tlx3"}
    if key not in aliases:
        raise ValueError(f"Unrecognized strain: {value!r}")
    return aliases[key]


def clean_excel(value):
    if pd.isna(value):
        return ""
    return re.sub(r"^(\d+)\.0+$", r"\1", str(value).strip())


def read_injection_overview(path):
    """Read side-by-side tables; carry animal metadata to second-channel rows.

    Never carry % Striatum forward: each injection has its own measurement.
    Numeric values are percentage points (e.g. 80, not 0.8), as in the
    supplied overview. Text such as '80%' is also accepted.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Set OVERVIEW_FILE to the actual spreadsheet path: {path}")
    entries = []
    for sheet, table in pd.read_excel(path, sheet_name=None, header=None, dtype=object).items():
        header_rows = [i for i in table.index
                       if any(clean_excel(v).lower() == "animal" for v in table.loc[i])]
        if not header_rows:
            continue
        if len(header_rows) != 1:
            raise ValueError(f"{sheet}: expected one header row containing Animal.")
        header_row = header_rows[0]
        header = [clean_excel(v).lower() for v in table.loc[header_row]]
        starts = [i for i, v in enumerate(header) if v == "animal"]
        for start, end in zip(starts, starts[1:] + [len(header)]):
            cols = {header[i]: i for i in range(start, end) if header[i]}
            required = {"animal", "strain", "cohort", "channel", "% striatum"}
            if required - cols.keys():
                raise ValueError(f"{sheet}: missing columns {sorted(required - cols.keys())}")
            current = None
            for index in range(header_row + 1, len(table)):
                values = {k: clean_excel(table.iloc[index, c]) for k, c in cols.items()}
                if not any(values.values()):
                    current = None
                    continue
                if values["animal"]:
                    current = {k: values[k] for k in ("animal", "strain", "cohort")}
                elif current:
                    for k in ("strain", "cohort"):
                        if values[k] and values[k] != current[k]:
                            raise ValueError(f"{sheet} row {index + 1}: conflicting {k}.")
                if not values["channel"]:
                    continue
                if not current or not all(current.values()):
                    raise ValueError(f"{sheet} row {index + 1}: incomplete injection metadata.")
                raw_percent = values["% striatum"]
                percent = pd.to_numeric(raw_percent.rstrip("%").strip().replace(",", "."), errors="coerce")
                entries.append({"strain": canonical_line(current["strain"]),
                                "cohort": current["cohort"], "animal_id": current["animal"],
                                "channel": values["channel"].lower(),
                                "striatum_percent": percent,
                                "striatum_percent_raw": raw_percent,
                                "overview_location": f"{sheet}, row {index + 1}"})
    if not entries:
        raise ValueError("No injections found in the overview.")
    overview = pd.DataFrame(entries)
    keys = ["strain", "cohort", "animal_id", "channel"]
    duplicates = overview.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError("Duplicate injection entries in overview; resolve before plotting:\n"
                         + overview.loc[duplicates].to_string(index=False))
    return overview


def filter_injections(records, overview):
    """Exclude unmatched, missing/invalid percentages and values below cutoff."""
    if not 0 <= MIN_STRIATUM_PERCENT <= 100:
        raise ValueError("MIN_STRIATUM_PERCENT must be between 0 and 100.")
    keys = ["strain", "cohort", "animal_id", "channel"]
    records = records.copy()
    for key in ("cohort", "animal_id"):
        records[key] = records[key].map(clean_excel)
    audit = records.merge(overview, on=keys, how="left", validate="one_to_one", indicator=True)
    percent = audit["striatum_percent"]
    audit["included"] = (audit["_merge"].eq("both") & percent.between(0, 100)
                         & percent.ge(MIN_STRIATUM_PERCENT))
    audit["filter_reason"] = "included"
    audit.loc[percent.lt(MIN_STRIATUM_PERCENT), "filter_reason"] = "below_striatum_threshold"
    audit.loc[~percent.between(0, 100), "filter_reason"] = "missing_or_invalid_striatum_percent"
    audit.loc[audit["_merge"].eq("left_only"), "filter_reason"] = "not_found_in_overview"
    audit["minimum_striatum_percent"] = MIN_STRIATUM_PERCENT
    audit = audit.drop(columns="_merge")
    return audit.loc[audit.included].copy(), audit


def collect_counts(input_dir):
    files = sorted(input_dir.glob("*_data.csv"))
    if not files:
        raise FileNotFoundError(f"No prepared *_data.csv files in {input_dir}")
    records, seen = [], set()
    pattern = re.compile(r"(?P<strain>.+)_cohort(?P<cohort>[^_]+)_(?P<animal>[^_]+)_(?P<channel>[^_]+)_data\.csv$", re.I)
    for path in files:
        data = pd.read_csv(path, dtype=str)
        required = {"strain", "cohort", "animal_id", "channel"}
        if not required.issubset(data.columns):
            raise ValueError(f"{path.name}: missing metadata {sorted(required - set(data.columns))}. Use the batch data_prep.py outputs.")
        if data.empty:
            match = pattern.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Cannot recover metadata from empty CSV: {path.name}")
            meta = match.groupdict()
            meta["animal_id"] = meta.pop("animal")
        else:
            meta = {}
            for column in required:
                values = data[column].str.strip()
                if values.isna().any() or values.eq("").any() or values.nunique() != 1:
                    raise ValueError(f"{path.name}: expected one nonempty {column} value.")
                meta[column] = values.iloc[0]
        line = canonical_line(meta["strain"])
        channel = meta["channel"].lower()
        if CHANNELS is not None and channel not in {c.lower() for c in CHANNELS}:
            continue
        key = (line, meta["cohort"], meta["animal_id"], channel)
        if key in seen:
            raise ValueError(f"Duplicate animal/channel dataset: {key}. Remove the duplicate input copy before plotting.")
        seen.add(key)
        pfc_data = select_pfc_cells(data)
        records.append({"group": GROUP_MAP[line], "strain": line,
                        "injection_id": "_".join(key),
                        "cohort": meta["cohort"], "animal_id": meta["animal_id"],
                        "channel": channel, "input_cells": len(data),
                        "excluded_non_pfc_or_unlabelled_cells": len(data) - len(pfc_data),
                        "n_cells": len(pfc_data), "source_csv": str(path)})
    if not records:
        raise ValueError("No datasets remain after channel selection.")
    return pd.DataFrame(records)


def summarize(records):
    # Sum channels within an animal; count that animal once in n_animals.
    animals = records.groupby(["group", "strain", "cohort", "animal_id"], as_index=False)["n_cells"].sum()
    lines = (animals.groupby("strain").agg(n_cells=("n_cells", "sum"),
             n_animals=("animal_id", "size")).reindex(LINE_ORDER, fill_value=0))
    lines.insert(0, "group", [GROUP_MAP[line] for line in LINE_ORDER])
    lines["n_injections"] = records.groupby("strain").size().reindex(LINE_ORDER, fill_value=0)
    groups = lines.groupby("group")[["n_cells", "n_animals", "n_injections"]].sum().reindex(["Control", "Fmr1"])
    return animals, lines, groups


LAYER_ORDER = ["L1", "L2/3", "L2", "L3", "L4", "L5", "L6", "L6a", "L6b",
               "Other", "Unassigned"]
LAYER_COLORS = dict(zip(LAYER_ORDER, [
    "#F4D35E", "#EE964B", "#F6B988", "#CE6C32", "#D65A72", "#5986C1",
    "#7160A5", "#9A83C3", "#504074", "#8D9B88", "#D3D3D3",
]))


def normalize_layer(value):
    """Normalize existing layer labels; never infer layers from mouse line."""
    if pd.isna(value):
        return "Unassigned"
    text = str(value).strip().lower()
    if text in {"", "none", "nan", "na", "n/a", "unknown", "unassigned", "-"}:
        return "Unassigned"
    text = re.sub(r"^layers?\s*", "", text)
    text = re.sub(r"^l\s*", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^([1-6])\.0$", r"\1", text)
    if text in {"2-3", "2–3", "2/3"}:
        text = "2/3"
    candidate = "L" + text
    return candidate if candidate in LAYER_ORDER else "Other"


def count_layers(records):
    """Count layers only for injections passing the overview filter.

    Returns per-injection counts (including zeros) and pooled line counts.
    Blank labels and non-layered structures remain Unassigned, preserving
    exactly the same total cells as the outer ring.
    """
    rows = []
    for record in records.to_dict("records"):
        data = pd.read_csv(record["source_csv"], dtype=str)
        if "layer" not in data.columns:
            raise ValueError(f"{record['source_csv']}: missing layer column. Rerun data preparation.")
        if len(data) != record["input_cells"]:
            raise ValueError(f"Source changed during this run: {record['source_csv']}")
        data = select_pfc_cells(data)
        if len(data) != record["n_cells"]:
            raise ValueError(f"PFC cell selection changed during this run: {record['source_csv']}")
        counts = data["layer"].map(normalize_layer).value_counts()
        for layer in LAYER_ORDER:
            rows.append({k: record[k] for k in
                         ["injection_id", "strain", "cohort", "animal_id", "channel"]}
                        | {"layer": layer, "n_cells": int(counts.get(layer, 0))})
    by_injection = pd.DataFrame(rows)
    pooled = (by_injection.pivot_table(index="strain", columns="layer", values="n_cells",
                                      aggfunc="sum", fill_value=0)
              .reindex(index=LINE_ORDER, columns=LAYER_ORDER, fill_value=0).astype(int))
    expected = records.groupby("strain").n_cells.sum().reindex(LINE_ORDER, fill_value=0)
    if not np.array_equal(pooled.sum(axis=1).to_numpy(), expected.to_numpy()):
        raise ValueError("Layer counts do not match total cell counts.")
    return by_injection, pooled


def report_unassigned_layers(records, injection_layers, output_dir):
    """Identify included injection CSVs with unassigned layers among PFC cells.

    total_cells and unassigned_percent use PFC cells only.
    """
    counts = injection_layers.loc[
        injection_layers["layer"].eq("Unassigned"), ["injection_id", "n_cells"]
    ].rename(columns={"n_cells": "unassigned_cells"})
    report = records.merge(counts, on="injection_id", how="left", validate="one_to_one")
    if report["unassigned_cells"].isna().any():
        raise ValueError("Missing Unassigned layer counts for an included injection.")
    report = report.loc[report["unassigned_cells"] > 0].copy()
    report["unassigned_cells"] = report["unassigned_cells"].astype(int)
    report["unassigned_percent"] = 100 * report["unassigned_cells"] / report["n_cells"]
    report = report.rename(columns={"n_cells": "total_cells"})
    columns = ["injection_id", "strain", "cohort", "animal_id", "channel",
               "total_cells", "unassigned_cells", "unassigned_percent", "source_csv"]
    report = report.loc[:, columns].sort_values(
        ["unassigned_cells", "source_csv"], ascending=[False, True])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "files_with_unassigned_layers.csv"
    # Always write the header, even if no files are affected, clearing old results.
    report.to_csv(report_path, index=False)
    if report.empty:
        print("\nNo unassigned layers among PFC cells in the included injections.")
    else:
        print(f"\nUNASSIGNED LAYERS: {int(report.unassigned_cells.sum()):,} PFC cells "
              f"across {len(report)} included injection files:")
        for row in report.itertuples(index=False):
            print(f"  {row.animal_id} | {row.strain} | {row.channel}: "
                  f"{row.unassigned_cells:,}/{row.total_cells:,} cells "
                  f"({row.unassigned_percent:.2f}%)\n    {row.source_csv}")
    print(f"Unassigned-layer report: {report_path}")
    return report


def make_figure(lines, layer_counts):
    """Outer ring: mouse lines. Inner: layers repeated within each line arc."""
    total = int(lines.n_cells.sum())
    if not np.array_equal(layer_counts.sum(axis=1).reindex(LINE_ORDER).to_numpy(),
                          lines.n_cells.reindex(LINE_ORDER).to_numpy()):
        raise ValueError("Inner and outer ring totals differ by mouse line.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, (ax, line_ax, layer_ax) = plt.subplots(
        1, 3, figsize=(11.2, 5), gridspec_kw={"width_ratios": [1.5, 1.25, .7]})
    present_lines = lines.loc[lines.n_cells > 0]
    inner_values, inner_colors = [], []
    # Concatenating layers in the OUTER line order preserves parent boundaries.
    for line in present_lines.index:
        for layer in LAYER_ORDER:
            count = int(layer_counts.loc[line, layer])
            if count:
                inner_values.append(count)
                inner_colors.append(LAYER_COLORS[layer])
    if total > 0:
        ax.pie(present_lines.n_cells, radius=1, startangle=90, counterclock=False,
               colors=[LINE_COLORS[k] for k in present_lines.index],
               wedgeprops={"width": .25, "edgecolor": "white", "linewidth": 1})
        ax.pie(inner_values, radius=.74, startangle=90, counterclock=False,
               colors=inner_colors,
               wedgeprops={"width": .28, "edgecolor": "white", "linewidth": .5})
        # Draw parent boundaries across both rings to clarify layer ownership.
        cumulative = np.r_[0, np.cumsum(present_lines.n_cells.to_numpy())]
        for value in cumulative[:-1]:
            theta = np.deg2rad(90 - 360 * value / total)
            ax.plot(np.array([.46, 1]) * np.cos(theta),
                    np.array([.46, 1]) * np.sin(theta), color="white", lw=1.8)
    ax.text(0, .06, f"{total:,}", ha="center", va="center", fontsize=19, weight="bold")
    ax.text(0, -.13, "PFC cell records", ha="center", va="center", fontsize=10)
    ax.set(aspect="equal")
    for legend_ax in (line_ax, layer_ax):
        legend_ax.axis("off")
    line_ax.legend(
        [Patch(facecolor=LINE_COLORS[line]) for line in LINE_ORDER],
        [f"{line}\n{int(lines.loc[line, 'n_cells']):,} cells · "
         f"{int(lines.loc[line, 'n_injections'])} injections" for line in LINE_ORDER],
        title="Outer ring: mouse line", loc="center left", frameon=False,
        fontsize=10, labelspacing=1.4, handlelength=1.2)
    present_layers = [layer for layer in LAYER_ORDER if layer_counts[layer].sum() > 0]
    if present_layers:
        layer_ax.legend([Patch(facecolor=LAYER_COLORS[layer]) for layer in present_layers],
                        present_layers, title="Inner ring: layer", loc="center left",
                        frameon=False, fontsize=10, labelspacing=.8, handlelength=1.2)
    fig.suptitle("Labelled PFC cells by mouse line and layer", fontsize=15, weight="bold", y=.96)
    fig.text(.5, .11, "Each mouse-line sector is subdivided by layer in the inner ring.",
             ha="center", fontsize=10)
    fig.text(.5, .055, f"Striatum ≥ {MIN_STRIATUM_PERCENT:g}%; cells pooled across included injections. "
             "Missing layer labels: Unassigned.", ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.02, right=.99, top=.85, bottom=.18, wspace=.03)
    return fig


def make_injection_table(lines, groups):
    """Compact colored sample-count table; n is injections, not mice."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axis("off")
    table = ax.table(
        cellText=[[GROUP_MAP[line], line, str(int(lines.loc[line, "n_injections"]))]
                  for line in LINE_ORDER],
        colLabels=["Group", "Mouse line", "Injections (n)"],
        cellLoc="center", colWidths=[.25, .45, .3], bbox=[.04, .22, .92, .66])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(2)
        if r == 0:
            cell.set_facecolor("#EDEDF2")
            cell.set_text_props(weight="bold")
        else:
            color = LINE_COLORS[LINE_ORDER[r - 1]]
            cell.set_facecolor(color)
            cell.set_text_props(color="white" if LINE_ORDER[r - 1] in {"Fezf2", "Fmr1-Fezf2"} else "#202020")
    fig.suptitle("Number of injections", fontsize=15, weight="bold", y=.97)
    totals = "    |    ".join(f"{g}: {int(groups.loc[g, 'n_injections'])}" for g in ["Control", "Fmr1"])
    fig.text(.5, .15, f"{totals}    |    Total: {int(groups.n_injections.sum())}", ha="center", fontsize=10)
    fig.text(.5, .06, "One animal/channel file = one injection", ha="center", fontsize=9, color="#555555")
    return fig


def make_injection_plot(records):
    """Boxes: median/IQR; whiskers: min/max; each dot: one injection.

    Log boxes use positive counts only. Zero-cell injections are shown as
    crosses on a separate labeled strip, never changed to a pseudocount.
    """
    fig = plt.figure(figsize=(7.2, 5.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, .12], hspace=.05, wspace=.14)
    axes, strips = [], []
    positive = records.loc[records.n_cells > 0, "n_cells"]
    ymax = max(10, 10 ** np.ceil(np.log10(positive.max() * 1.4))) if len(positive) else 10
    rng = np.random.default_rng(42)
    for panel, group in enumerate(["Control", "Fmr1"]):
        ax = fig.add_subplot(grid[0, panel], sharey=axes[0] if axes else None)
        strip = fig.add_subplot(grid[1, panel], sharex=ax)
        axes.append(ax)
        strips.append(strip)
        line_names = [line for line in LINE_ORDER if GROUP_MAP[line] == group]
        for x, line in enumerate(line_names):
            counts = records.loc[records.strain == line, "n_cells"].to_numpy()
            nonzero = counts[counts > 0]
            if len(nonzero):
                ax.boxplot([nonzero], positions=[x], widths=.42, patch_artist=True,
                           manage_ticks=False, showfliers=False, whis=(0, 100),
                           boxprops={"facecolor": LINE_COLORS[line], "edgecolor": LINE_COLORS[line], "alpha": .75},
                           medianprops={"color": "#222222", "linewidth": 1.6},
                           whiskerprops={"color": "#777777"}, capprops={"color": "#777777"})
                ax.scatter(x + rng.uniform(-.12, .12, len(nonzero)), nonzero,
                           s=25, color="#555555", edgecolors="white", linewidths=.35, alpha=.85, zorder=3)
            zeros = int((counts == 0).sum())
            if zeros:
                strip.scatter(x + rng.uniform(-.12, .12, zeros), np.zeros(zeros),
                              marker="x", s=25, color="#555555")
            ax.text(x, 1.015, f"n = {len(counts)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(.8, ymax)
        ax.set_xlim(-.55, 1.55)
        ax.set_title(group, color=GROUP_COLORS[group], fontsize=13, weight="bold", pad=28)
        ax.grid(axis="y", which="major", linestyle=":", color="#CCCCCC")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for side in ["top", "right", "bottom"]:
            ax.spines[side].set_visible(False)
        strip.set_facecolor(GROUP_COLORS[group] + "22")
        strip.set_ylim(-.6, .6)
        strip.set_yticks([0], labels=["0"] if panel == 0 else [""])
        strip.tick_params(axis="y", length=0)
        strip.set_xticks(range(2), labels=line_names, rotation=25, ha="right")
        for spine in strip.spines.values():
            spine.set_visible(False)
        if panel:
            ax.tick_params(axis="y", labelleft=False)
            ax.spines["left"].set_visible(False)
    axes[0].set_ylabel("PFC cells per injection (log scale)")
    fig.suptitle("Labelled PFC cells per injection", fontsize=15, weight="bold", y=.98)
    fig.text(.5, .08, "One point = one injection; boxes = median and IQR; whiskers = min–max.", ha="center", fontsize=9)
    fig.text(.5, .035, "Zero-cell injections: crosses in the 0 strip; excluded from log-scale boxes.", ha="center", fontsize=8.5, color="#555555")
    fig.subplots_adjust(left=.12, right=.97, top=.78, bottom=.24)
    return fig


def main():
    overview = read_injection_overview(OVERVIEW_FILE)
    records = collect_counts(INPUT_DIR)
    records, audit = filter_injections(records, overview)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUTPUT_DIR / "injection_filter_audit.csv", index=False)
    print(f"Striatum >= {MIN_STRIATUM_PERCENT:g}%: included {len(records)} of {len(audit)} available injections.")
    print(audit.filter_reason.value_counts().to_string())
    if records.empty:
        raise ValueError("No injections pass the filter. See injection_filter_audit.csv. No figures were updated.")
    records[["injection_id", "strain", "cohort", "animal_id", "channel",
             "input_cells", "excluded_non_pfc_or_unlabelled_cells", "n_cells", "source_csv"]].to_csv(
                 OUTPUT_DIR / "pfc_cell_filter_audit.csv", index=False)
    print(f"PFC filter: retained {int(records.n_cells.sum()):,} of "
          f"{int(records.input_cells.sum()):,} cells in included injections.")
    animals, lines, groups = summarize(records)
    injection_layers, layer_counts = count_layers(records)
    report_unassigned_layers(records, injection_layers, OUTPUT_DIR)
    figures = {
        "total_cells_donut": make_figure(lines, layer_counts),
        "injection_counts_table": make_injection_table(lines, groups),
        "cells_per_injection": make_injection_plot(records),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records.to_csv(OUTPUT_DIR / "cell_counts_by_file.csv", index=False)
    records.to_csv(OUTPUT_DIR / "cell_counts_by_injection.csv", index=False)
    animals.to_csv(OUTPUT_DIR / "cell_counts_by_animal.csv", index=False)
    lines.to_csv(OUTPUT_DIR / "cell_counts_by_line.csv")
    groups.to_csv(OUTPUT_DIR / "cell_counts_by_group.csv")
    injection_layers.to_csv(OUTPUT_DIR / "cell_counts_by_injection_layer.csv", index=False)
    layer_counts.to_csv(OUTPUT_DIR / "cell_counts_by_line_layer.csv")
    layer_percent = layer_counts.div(layer_counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    layer_percent.to_csv(OUTPUT_DIR / "cell_percent_by_layer_within_line.csv")
    for name, fig in figures.items():
        for extension in ("png", "pdf", "svg"):
            fig.savefig(OUTPUT_DIR / f"{name}.{extension}", dpi=400, bbox_inches="tight")
    print(lines.to_string())
    print(f"\nDONE! Figures and count tables saved in: {OUTPUT_DIR}")
    if SHOW_FIGURE:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)


if __name__ == "__main__":
    main()
