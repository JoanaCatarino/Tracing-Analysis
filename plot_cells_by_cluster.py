"""PFC cell counts by spatial cluster and mouse line.
One point = one injection. Boxes = median/IQR, whiskers = min/max.
Run cluster_injection_centroids.py first using the current overview.
Dependencies: numpy, pandas, matplotlib. Original cell files are read only.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(r"L:\dmclab\Joana\Tracing\Analysis")
INPUT_DIR = BASE_DIR / "cells_prep"
CLUSTER_FILE = Path(r'L:/dmclab/Joana/Tracing/Analysis/custering_results_new/injection_cluster_candidates.csv')
OUTPUT_DIR = BASE_DIR / "plots" / "cells_by_injection_cluster"
CLUSTER_SOLUTIONS = [2, 3, 4, 5]
SHOW_FIGURES = False
CHANNELS = None
LINE_ORDER = ["Tlx3", "Fezf2", "Fmr1-Fezf2", "Fmr1-Tlx3"]
GROUP_MAP = dict(zip(LINE_ORDER, ["Control", "Control", "Fmr1", "Fmr1"]))
LINE_COLORS = {"Tlx3": "#CBB4D9", "Fezf2": "#805797", "Fmr1-Fezf2": "#116E70", "Fmr1-Tlx3": "#80C7C5"}
CLUSTER_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]
PFC_REGIONS = ["Frontal pole", "Prelimbic area", "Infralimbic area", "Secondary motor area",
               "Orbital area", "Anterior cingulate area", "Agranular insular area"]

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

def plot_solution(data, k, ymax):
    col = f"cluster_k{k}"
    fig = plt.figure(figsize=(max(9, k * 3.0), 5.5))
    grid = fig.add_gridspec(2, k, height_ratios=[1, .12], hspace=.04, wspace=.10)
    axes = []
    rng = np.random.default_rng(42)
    for panel in range(k):
        cluster = panel + 1
        ax = fig.add_subplot(grid[0, panel], sharey=axes[0] if axes else None)
        strip = fig.add_subplot(grid[1, panel], sharex=ax)
        axes.append(ax)
        subset = data[data[col] == cluster]
        for position, line in enumerate(LINE_ORDER):
            values = subset.loc[subset.strain == line, "n_cells"].to_numpy()
            positive = values[values > 0]
            if len(positive):
                ax.boxplot([positive], positions=[position], widths=.48, patch_artist=True,
                    manage_ticks=False, showfliers=False, whis=(0, 100),
                    boxprops={"facecolor": LINE_COLORS[line], "edgecolor": LINE_COLORS[line], "alpha": .85},
                    medianprops={"color": "#222222", "linewidth": 1.6},
                    whiskerprops={"color": "#555555"}, capprops={"color": "#555555"})
                ax.scatter(position+rng.uniform(-.12,.12,len(positive)), positive,
                    s=23, c="#555555", edgecolors="white", linewidths=.4, alpha=.85, zorder=3)
            zeros = int((values == 0).sum())
            if zeros:
                strip.scatter(position+rng.uniform(-.12,.12,zeros), np.zeros(zeros),
                              marker="x", color="#333333", s=25)
            ax.text(position, 1.02, f"n={len(values)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log"); ax.set_ylim(.8,ymax); ax.set_xlim(-.6,3.6)
        ax.set_title(f"Cluster {cluster}", pad=26, color=CLUSTER_COLORS[panel], weight="bold")
        ax.grid(axis="y", which="major", linestyle=":", alpha=.5); ax.set_axisbelow(True)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        for side in ["top","right","bottom"]: ax.spines[side].set_visible(False)
        if panel:
            ax.tick_params(axis="y", labelleft=False); ax.spines["left"].set_visible(False)
        strip.set_facecolor(CLUSTER_COLORS[panel]+"25")
        strip.set_ylim(-.6,.6)
        strip.set_yticks([0], labels=["0"] if panel == 0 else [""])
        strip.tick_params(axis="y", length=0)
        strip.set_xticks(range(4), labels=LINE_ORDER, rotation=45, ha="right", fontsize=9)
        for spine in strip.spines.values(): spine.set_visible(False)
    axes[0].set_ylabel("PFC cells per injection (log scale)")
    fig.suptitle(f"{k} spatial clusters — {len(data)} injections, {int(data.n_cells.sum()):,} PFC cells", fontsize=13, y=.98)
    fig.text(.5,.075,"Points: injections · Boxes: median/IQR · Whiskers: min–max · n: injections",ha="center",fontsize=9)
    fig.text(.5,.035,"Zero counts: crosses in the 0 strip, excluded from log boxes. Empty groups retain n=0.",ha="center",fontsize=8)
    fig.subplots_adjust(left=.08,right=.99,top=.80,bottom=.31)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clusters = pd.read_csv(CLUSTER_FILE, dtype={"animal_id":str,"cohort":str})
    required = ["injection_id", "strain", "animal_id", "cohort", "channel", "cells", "striatum_percent"]
    required += [f"cluster_k{k}" for k in CLUSTER_SOLUTIONS]
    missing = set(required)-set(clusters.columns)
    if missing: raise ValueError(f"Cluster table missing {sorted(missing)}. Rerun the current clustering script.")
    if clusters.injection_id.duplicated().any(): raise ValueError("Duplicated injection IDs in cluster table.")
    eligible = clusters.cells.fillna("").astype(str).str.strip().str.lower().eq("y") & pd.to_numeric(clusters.striatum_percent,errors="coerce").between(80,100)
    if not eligible.all(): raise ValueError("Cluster table contains ineligible injections. Rerun clustering against the current overview.")
    if clusters.empty: raise ValueError("No clustered injections.")
    for k in CLUSTER_SOLUTIONS:
        col=f"cluster_k{k}"
        values=pd.to_numeric(clusters[col],errors="coerce")
        if values.isna().any() or not values.isin(range(1,k+1)).all() or values.nunique()!=k:
            raise ValueError(f"Invalid/incomplete assignments for {col}.")
        clusters[col]=values.astype(int)
    counts=collect_counts(INPUT_DIR)
    merged=clusters.merge(counts[["injection_id","n_cells","input_cells","source_csv"]],
        on="injection_id",how="left",validate="one_to_one")
    merged.to_csv(OUTPUT_DIR/"injection_cell_counts_and_clusters.csv",index=False)
    absent=merged[merged.n_cells.isna()]
    if len(absent):
        raise FileNotFoundError("Prepared cell files missing for: "+", ".join(absent.injection_id))
    positive=merged.loc[merged.n_cells>0,"n_cells"]
    ymax=max(10,10**np.ceil(np.log10(positive.max()*1.4))) if len(positive) else 10
    summary=[]
    for k in CLUSTER_SOLUTIONS:
        for cluster in range(1,k+1):
            for line in LINE_ORDER:
                sub=merged[(merged[f"cluster_k{k}"]==cluster)&merged.strain.eq(line)]
                v=sub.n_cells
                summary.append(dict(k=k,cluster=cluster,strain=line,injections=len(sub),
                    animals=sub[["strain","cohort","animal_id"]].drop_duplicates().shape[0],
                    total_pfc_cells=int(v.sum()),zero_count_injections=int(v.eq(0).sum()),
                    median_all_counts=v.median() if len(v) else np.nan))
        fig=plot_solution(merged,k,ymax)
        for ext in ["png","pdf","svg"]:
            fig.savefig(OUTPUT_DIR/f"pfc_cells_{k}_clusters.{ext}",dpi=400)
        if SHOW_FIGURES: plt.show()
    pd.DataFrame(summary).to_csv(OUTPUT_DIR/"cluster_line_summary.csv",index=False)
    print(f"DONE! {len(merged)} injections; {int(merged.n_cells.sum()):,} PFC cells. Saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
