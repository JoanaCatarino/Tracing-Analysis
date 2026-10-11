"""Striatum_all_cells: PFC counts and laterality per striatal injection.

Standalone script: run in Spyder. Requires pandas, numpy, matplotlib, openpyxl.
Set OVERVIEW_FILE to the updated workbook's actual path before running.
Each animal/channel file is ONE injection. Only % Striatum >=80 is included.
'injection side' must be right/left for each injection row (including second
channels). A blank side is excluded and reported, never inferred.
Only the seven user-defined PFC areas are counted; all their layers are pooled.
Source CSVs/workbook are read-only. Figures: PNG/PDF/SVG at 400 dpi.
The anatomical atlas drawings in the reference are not generated here.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

OUTPUT_DIR = Path(r"L:\dmclab\Joana\Tracing\Analysis\plots\Striatum_all_cells")
SHOW_FIGURES = True
# Reference subregions. Parent labels and additional PFC subregions present
# in the data are appended, so no PFC cells disappear from regional totals.
REGION_ORDER = ["MOs", "ACAd", "ACAv", "ILA", "PL", "ORBm", "ORBl", "ORBvl", "FRP", "AId", "AIv"]
SIDES = ["ipsilateral", "contralateral"]


INPUT_DIR = Path(r"L:\dmclab\Joana\Tracing\Analysis\cells_prep")


OVERVIEW_FILE = Path(r"L:\dmclab\Joana\Tracing\tracing_overview.xlsx")


MIN_STRIATUM_PERCENT = 80.0


CHANNELS = None


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
            required = {"animal", "strain", "cohort", "channel", "% striatum", "injection side"}
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
                                "injection_side": values["injection side"].strip().lower(),
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


def pfc_subregion(row):
    """Strip only explicit layer suffixes; preserve subdivision identity."""
    acronym = clean_excel(row.get("acronym", ""))
    match = re.fullmatch(
        r"(FRP|PL|ILA|MOs|ORB(?:m|l|vl)?|ACA[dv]?|AI[dvp]?)(?:2/3|6a|6b|[1-6])?",
        acronym, flags=re.IGNORECASE)
    canonical = {x.lower(): x for x in
                 ["FRP", "PL", "ILA", "MOs", "ORB", "ORBm", "ORBl", "ORBvl",
                  "ACA", "ACAd", "ACAv", "AI", "AId", "AIv", "AIp"]}
    if match:
        return canonical[match.group(1).lower()]
    # Use explicit region/part labels if an acronym is unavailable.
    region = clean_excel(row.get("region", row.get("name", ""))).split(",")[0].strip().lower()
    part = clean_excel(row.get("part", "")).lower()
    simple = {"frontal pole": "FRP", "prelimbic area": "PL", "infralimbic area": "ILA",
              "secondary motor area": "MOs"}
    if region in simple:
        return simple[region]
    subdivisions = {
        "orbital area": ("ORB", {"medial part": "ORBm", "lateral part": "ORBl", "ventrolateral part": "ORBvl"}),
        "anterior cingulate area": ("ACA", {"dorsal part": "ACAd", "ventral part": "ACAv"}),
        "agranular insular area": ("AI", {"dorsal part": "AId", "ventral part": "AIv", "posterior part": "AIp"}),
    }
    if region in subdivisions:
        parent, parts = subdivisions[region]
        return parts.get(part, parent)
    raise ValueError(f"Cannot map PFC subregion: {region!r}, acronym {acronym!r}")


def analyze_injections(records):
    """Keep each injection as one observation; explicitly zero-fill regions."""
    totals, region_rows, errors = [], [], []
    for record in records.to_dict("records"):
        try:
            data = pd.read_csv(record["source_csv"], dtype=str)
            data = select_pfc_cells(data)
            if len(data) != record["n_cells"]:
                raise ValueError("PFC count changed during the run.")
            if "hemisphere" not in data:
                raise ValueError("Missing cell hemisphere column.")
            hemisphere = data.hemisphere.astype("string").str.strip().str.lower()
            if not hemisphere.isin(["left", "right"]).all():
                raise ValueError("Missing or invalid PFC cell hemisphere; expected left/right.")
            side = record["injection_side"]
            if side not in {"left", "right"}:
                raise ValueError("Missing or invalid injection side.")
            laterality = pd.Series(np.where(hemisphere.eq(side), "ipsilateral", "contralateral"), index=data.index)
            regions = data.apply(pfc_subregion, axis=1) if len(data) else pd.Series(dtype=str)
            ipsi = int(laterality.eq("ipsilateral").sum())
            contra = int(laterality.eq("contralateral").sum())
            total = len(data)
            totals.append(dict(record, ipsilateral_cells=ipsi, contralateral_cells=contra,
                               ipsilateral_percent=100 * ipsi / total if total else np.nan,
                               contralateral_percent=100 * contra / total if total else np.nan))
            for (region, hemi), count in pd.DataFrame({"region": regions, "laterality": laterality}).value_counts().items():
                region_rows.append({"injection_id": record["injection_id"], "region": region,
                                    "laterality": hemi, "n_cells": int(count)})
        except Exception as exc:
            errors.append({"injection_id": record["injection_id"], "source_csv": record["source_csv"],
                           "error": str(exc)})
    total_df = pd.DataFrame(totals)
    error_df = pd.DataFrame(errors, columns=["injection_id", "source_csv", "error"])
    if total_df.empty:
        return total_df, pd.DataFrame(), error_df
    extras = sorted({row["region"] for row in region_rows} - set(REGION_ORDER))
    order = REGION_ORDER + extras
    grid = pd.MultiIndex.from_product([total_df.injection_id, order, SIDES],
                                     names=["injection_id", "region", "laterality"])
    observed = pd.DataFrame(region_rows, columns=["injection_id", "region", "laterality", "n_cells"])
    full = observed.set_index(["injection_id", "region", "laterality"]).reindex(grid, fill_value=0).reset_index()
    full = full.merge(total_df[["injection_id", "strain", "animal_id", "cohort", "channel", "injection_side"]],
                      on="injection_id", validate="many_to_one")
    check = full.groupby("injection_id").n_cells.sum().reindex(total_df.injection_id).to_numpy()
    if not np.array_equal(check, total_df.n_cells.to_numpy()):
        raise ValueError("Regional counts do not sum to each injection's total PFC cells.")
    return total_df, full, error_df


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="out")


def draw_total_counts(ax, totals):
    rng = np.random.default_rng(10)
    for x, line in enumerate(LINE_ORDER):
        values = totals.loc[totals.strain.eq(line), "n_cells"].to_numpy() / 1000
        if len(values):
            ax.boxplot([values], positions=[x], widths=.52, patch_artist=True,
                       manage_ticks=False, showfliers=False, whis=(0, 100),
                       boxprops={"facecolor": LINE_COLORS[line], "edgecolor": LINE_COLORS[line]},
                       medianprops={"color": "#333333"}, whiskerprops={"color": "#777777"},
                       capprops={"color": "#777777"})
            ax.scatter(x + rng.uniform(-.13, .13, len(values)), values, s=17,
                       color="#555555", alpha=.7, edgecolors="white", linewidths=.3, zorder=3)
        ax.text(x, 1.02, f"n={len(values)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
    ax.set_xticks(range(4), LINE_ORDER, rotation=30, ha="right")
    ax.set_ylabel("PFC cells (×1,000)")
    ax.set_title("B   Total PFC cells", loc="left", weight="bold", pad=26)
    ax.set_ylim(bottom=0)
    ax.set_xlim(-.6, 3.6)
    style_axis(ax)


def draw_percentages(ax, totals):
    rng = np.random.default_rng(20)
    offsets = np.linspace(-.27, .27, 4)
    for x, side in enumerate(SIDES):
        for offset, line in zip(offsets, LINE_ORDER):
            values = totals.loc[totals.strain.eq(line), side + "_percent"].dropna().to_numpy()
            mean, sem = mean_sem(values)
            if len(values):
                ax.bar(x + offset, mean, width=.16, color=LINE_COLORS[line])
                if np.isfinite(sem):
                    ax.errorbar(x + offset, mean, yerr=sem, color="#555555", capsize=2, lw=.8)
                ax.scatter(x + offset + rng.uniform(-.045, .045, len(values)), values,
                           s=13, color="#555555", alpha=.6, edgecolors="white", linewidths=.25, zorder=3)
    ax.set_xticks([0, 1], ["Ipsilateral", "Contralateral"])
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("PFC cells (%)")
    ax.set_title("C   Hemispheric distribution", loc="left", weight="bold", pad=12)
    style_axis(ax)


def draw_regions(ax, regional):
    rng = np.random.default_rng(30)
    regions = list(dict.fromkeys(regional.region))
    offsets = np.linspace(-.30, .30, 4)
    max_extent = 1
    for y, region in enumerate(regions):
        for offset, line in zip(offsets, LINE_ORDER):
            for side, sign in [("contralateral", -1), ("ipsilateral", 1)]:
                values = regional.loc[regional.region.eq(region) & regional.strain.eq(line)
                                      & regional.laterality.eq(side), "n_cells"].to_numpy()
                if not len(values):
                    continue
                mean, sem = mean_sem(values)
                ax.barh(y + offset, sign * mean, height=.17, color=LINE_COLORS[line], alpha=.9)
                if np.isfinite(sem):
                    ax.errorbar(sign * mean, y + offset, xerr=sem, color="#555555", capsize=1.5, lw=.7)
                ax.scatter(sign * values, y + offset + rng.uniform(-.045, .045, len(values)),
                           s=7, color="#555555", alpha=.4, linewidths=0, zorder=3)
                max_extent = max(max_extent, values.max(), mean + (sem if np.isfinite(sem) else 0))
    ax.axvline(0, color="#333333", lw=.8)
    ax.set_xlim(-max_extent * 1.08, max_extent * 1.08)
    ax.set_yticks(range(len(regions)), regions)
    ax.set_ylim(len(regions) - .45, -.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):,.0f}"))
    ax.set_xlabel("PFC cells per injection")
    ax.set_title("D   Regional distribution", loc="left", weight="bold", pad=30)
    ax.text(.25, 1.01, "Contralateral", transform=ax.transAxes, ha="center", fontsize=10)
    ax.text(.75, 1.01, "Ipsilateral", transform=ax.transAxes, ha="center", fontsize=10)
    style_axis(ax)


def make_figure(totals, regional):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    n_regions = regional.region.nunique()
    fig = plt.figure(figsize=(12, max(8, n_regions * .55 + 2)))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.9], hspace=.65, wspace=.4)
    draw_total_counts(fig.add_subplot(grid[0, 0]), totals)
    draw_percentages(fig.add_subplot(grid[1, 0]), totals)
    draw_regions(fig.add_subplot(grid[:, 1]), regional)
    fig.legend([Patch(facecolor=LINE_COLORS[line]) for line in LINE_ORDER], LINE_ORDER,
               loc="upper center", bbox_to_anchor=(.5, .955), ncol=4, frameon=False)
    fig.suptitle("PFC cells labelled from striatal injections", fontsize=16, weight="bold", y=.985)
    zero = int(totals.n_cells.eq(0).sum())
    fig.text(.5, .075, f"Each point = one injection · Striatum ≥ {MIN_STRIATUM_PERCENT:g}% · "
             f"{len(totals)} included injections", ha="center", fontsize=10)
    fig.text(.5, .045, "B: median, IQR and min–max. C–D: mean ± SEM across injections; SEM omitted when n=1.",
             ha="center", fontsize=9)
    fig.text(.5, .02, f"Percentages use total PFC cells within each injection. Zero-PFC injections omitted from C: {zero}.",
             ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.08, right=.98, bottom=.18, top=.85)
    return fig


def main():
    overview = read_injection_overview(OVERVIEW_FILE)
    records = collect_counts(INPUT_DIR)
    _, audit = filter_injections(records, overview)
    bad_side = audit.included & ~audit.injection_side.isin(["left", "right"])
    audit.loc[bad_side, "included"] = False
    audit.loc[bad_side, "filter_reason"] = "missing_or_invalid_injection_side"
    included = audit.loc[audit.included].copy()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if included.empty:
        audit.to_csv(OUTPUT_DIR / "injection_filter_audit.csv", index=False)
        raise ValueError("No eligible injections. Check injection_filter_audit.csv and injection side values.")
    totals, regional, errors = analyze_injections(included)
    if not errors.empty:
        failed = audit.injection_id.isin(errors.injection_id)
        audit.loc[failed, "included"] = False
        audit.loc[failed, "filter_reason"] = "cell_data_error"
        print(errors.to_string(index=False))
    errors.to_csv(OUTPUT_DIR / "injection_errors.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "injection_filter_audit.csv", index=False)
    if totals.empty:
        raise ValueError("No valid injections. See injection_errors.csv. No figures updated.")
    totals.to_csv(OUTPUT_DIR / "pfc_counts_per_injection.csv", index=False)
    regional.to_csv(OUTPUT_DIR / "pfc_region_counts_per_injection.csv", index=False)
    summary = regional.groupby(["strain", "region", "laterality"]).n_cells.agg(["size", "mean", "sem"])
    summary.rename(columns={"size": "n_injections"}).to_csv(OUTPUT_DIR / "pfc_region_summary.csv")
    fig = make_figure(totals, regional)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(OUTPUT_DIR / f"Striatum_all_cells.{extension}", dpi=400, bbox_inches="tight")
    print(audit.filter_reason.value_counts().to_string())
    print(f"DONE! {len(totals)} injections; {int(totals.n_cells.sum()):,} PFC cell records.\nSaved in: {OUTPUT_DIR}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
