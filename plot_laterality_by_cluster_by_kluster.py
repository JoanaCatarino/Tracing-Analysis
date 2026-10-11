"""PFC laterality and regional counts by spatial cluster, pooling all mouse lines.
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
OUTPUT_DIR = BASE_DIR / "plots" / "laterality_by_injection_cluster_pooled"
CLUSTER_SOLUTIONS = [2, 3, 4, 5]
SHOW_FIGURES = False
CHANNELS = None
LINE_ORDER = ["Tlx3", "Fezf2", "Fmr1-Fezf2", "Fmr1-Tlx3"]
GROUP_MAP = dict(zip(LINE_ORDER, ["Control", "Control", "Fmr1", "Fmr1"]))
LINE_COLORS = {"Tlx3": "#CBB4D9", "Fezf2": "#805797", "Fmr1-Fezf2": "#116E70", "Fmr1-Tlx3": "#80C7C5"}
CLUSTER_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]
PFC_REGIONS = ["Frontal pole", "Prelimbic area", "Infralimbic area", "Secondary motor area",
               "Orbital area", "Anterior cingulate area", "Agranular insular area"]

from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FuncFormatter
REGION_ORDER = ["MOs", "ACAd", "ACAv", "ILA", "PL", "ORBm", "ORBl", "ORBvl", "FRP", "AId", "AIv"]
SIDES = ["ipsilateral", "contralateral"]

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

def make_cluster_figure(totals, regional, k):
    """Reference-style B/C/D panels; colors encode clusters, all lines pooled."""
    fig=plt.figure(figsize=(13,9))
    grid=fig.add_gridspec(2,2,width_ratios=[1,2],hspace=.65,wspace=.42)
    ax_total=fig.add_subplot(grid[0,0]);ax_pct=fig.add_subplot(grid[1,0]);ax_region=fig.add_subplot(grid[:,1])
    col=f'cluster_k{k}'
    labels=[f'Cluster {c}' for c in range(1,k+1)]
    offsets=np.linspace(-.32,.32,k)
    width=min(.18,.7/k)
    regions=list(dict.fromkeys(regional.region))
    rng=np.random.default_rng(42)
    extent=1
    for j,cluster in enumerate(range(1,k+1)):
        color=CLUSTER_COLORS[j]
        t=totals[totals[col]==cluster]
        r=regional[regional.injection_id.isin(t.injection_id)]
        counts=t.n_cells.to_numpy()/1000
        if len(counts):
            ax_total.boxplot([counts],positions=[j],widths=.5,patch_artist=True,manage_ticks=False,
                showfliers=False,whis=(0,100),boxprops={'facecolor':color,'edgecolor':color},
                medianprops={'color':'#333333'},whiskerprops={'color':'#777777'},capprops={'color':'#777777'})
            ax_total.scatter(j+rng.uniform(-.12,.12,len(counts)),counts,s=18,color='#555555',alpha=.7,zorder=3)
        ax_total.text(j,1.02,f'n={len(t)}',transform=ax_total.get_xaxis_transform(),ha='center',fontsize=8)
        for x,side in enumerate(SIDES):
            v=t[side+'_percent'].dropna().to_numpy();mean,sem=mean_sem(v)
            if len(v):
                ax_pct.bar(x+offsets[j],mean,width=width,color=color)
                if np.isfinite(sem):ax_pct.errorbar(x+offsets[j],mean,yerr=sem,color='#555555',capsize=2,lw=.8)
                ax_pct.scatter(x+offsets[j]+rng.uniform(-width*.2,width*.2,len(v)),v,s=12,color='#555555',alpha=.6,zorder=3)
        for y,region in enumerate(regions):
            for side,sign in [('contralateral',-1),('ipsilateral',1)]:
                v=r.loc[r.region.eq(region)&r.laterality.eq(side),'n_cells'].to_numpy()
                if not len(v):continue
                mean,sem=mean_sem(v)
                ax_region.barh(y+offsets[j],sign*mean,height=width,color=color)
                if np.isfinite(sem):ax_region.errorbar(sign*mean,y+offsets[j],xerr=sem,color='#555555',capsize=1.5,lw=.7)
                ax_region.scatter(sign*v,y+offsets[j]+rng.uniform(-width*.2,width*.2,len(v)),s=7,color='#555555',alpha=.4,zorder=3)
                extent=max(extent,float(v.max()),mean+(sem if np.isfinite(sem) else 0))
    ax_total.set_xticks(range(k),labels,rotation=30,ha='right')
    ax_total.set_xlim(-.6,k-.4);ax_total.set_ylim(bottom=0)
    ax_total.set_ylabel('PFC cells (×1,000)');ax_total.set_title('B   Total PFC cells',loc='left',weight='bold',pad=25)
    ax_pct.set_xticks([0,1],['Ipsilateral','Contralateral']);ax_pct.set_xlim(-.55,1.55)
    ax_pct.set_ylim(0,110);ax_pct.set_yticks(range(0,101,20));ax_pct.set_ylabel('PFC cells (%)')
    ax_pct.set_title('C   Hemispheric distribution',loc='left',weight='bold',pad=12)
    ax_region.axvline(0,color='#333333',lw=.8)
    ax_region.set_xlim(-extent*1.08,extent*1.08)
    ax_region.set_yticks(range(len(regions)),regions);ax_region.set_ylim(len(regions)-.45,-.7)
    ax_region.xaxis.set_major_locator(MaxNLocator(nbins=7,integer=True))
    ax_region.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{abs(x):,.0f}'))
    ax_region.set_xlabel('PFC cells per injection')
    ax_region.set_title('D   Regional distribution',loc='left',weight='bold',pad=30)
    ax_region.text(.25,1.01,'Contralateral',transform=ax_region.transAxes,ha='center')
    ax_region.text(.75,1.01,'Ipsilateral',transform=ax_region.transAxes,ha='center')
    for ax in [ax_total,ax_pct,ax_region]:style_axis(ax)
    fig.legend([Patch(facecolor=CLUSTER_COLORS[j]) for j in range(k)],labels,
               loc='upper center',bbox_to_anchor=(.5,.955),ncol=k,frameon=False)
    fig.suptitle(f'PFC projection profiles — {k} spatial clusters, all mouse lines pooled',fontsize=15,y=.99)
    fig.text(.5,.085,f'{len(totals)} injections · cells=y and striatum ≥80% · Each dot = one injection',ha='center')
    fig.text(.5,.055,'B: median/IQR, min–max. C–D: mean ± SEM across injections; no SEM for n=1.',ha='center',fontsize=9)
    fig.text(.5,.025,f'Percentages use each injection’s PFC total. Zero-PFC injections omitted from C: {int(totals.n_cells.eq(0).sum())}.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.98,top=.84,bottom=.20)
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    clusters=pd.read_csv(CLUSTER_FILE,dtype={"animal_id":str,"cohort":str})
    required=['injection_id','strain','cohort','animal_id','channel','injection_side','cells','striatum_percent']+[f'cluster_k{k}' for k in CLUSTER_SOLUTIONS]
    if set(required)-set(clusters):raise ValueError(f'Missing cluster-table columns: {set(required)-set(clusters)}. Rerun clustering.')
    if clusters.empty or clusters.injection_id.duplicated().any():raise ValueError('Empty cluster table or duplicated injection IDs.')
    keep=clusters.cells.fillna('').astype(str).str.strip().str.lower().eq('y') & pd.to_numeric(clusters.striatum_percent,errors='coerce').between(80,100)
    if not keep.all():raise ValueError('Ineligible rows in cluster table; rerun clustering with the current overview.')
    clusters['injection_side']=clusters.injection_side.astype('string').str.strip().str.lower()
    if not clusters.injection_side.isin(['left','right']).all():raise ValueError('Invalid original injection side.')
    for k in CLUSTER_SOLUTIONS:
        col=f'cluster_k{k}';v=pd.to_numeric(clusters[col],errors='coerce')
        if not v.isin(range(1,k+1)).all() or v.nunique()!=k:raise ValueError(f'Invalid assignments: {col}')
        clusters[col]=v.astype(int)
    counts=collect_counts(INPUT_DIR)
    # Match the full injection key; do not substitute another channel from an animal.
    records=clusters.merge(counts[['injection_id','n_cells','source_csv']],on='injection_id',how='left',validate='one_to_one')
    records.to_csv(OUTPUT_DIR/'input_match_audit.csv',index=False)
    missing=records[records.source_csv.isna()]
    if len(missing):raise FileNotFoundError('Missing prepared cell files: '+', '.join(missing.injection_id))
    totals,regional,errors=analyze_injections(records)
    errors.to_csv(OUTPUT_DIR/'processing_errors.csv',index=False)
    if len(errors):raise ValueError('Cell hemisphere/region errors: see processing_errors.csv. No incomplete-cohort figures produced.')
    totals.to_csv(OUTPUT_DIR/'injection_laterality_counts.csv',index=False)
    regional=regional.merge(clusters[['injection_id']+[f'cluster_k{k}' for k in CLUSTER_SOLUTIONS]],on='injection_id',validate='many_to_one')
    regional.to_csv(OUTPUT_DIR/'injection_region_laterality_counts.csv',index=False)
    summaries=[]
    for k in CLUSTER_SOLUTIONS:
        for cluster in range(1,k+1):
            t=totals[totals[f'cluster_k{k}']==cluster]
            r=regional[regional[f'cluster_k{k}']==cluster]
            for metric in ['n_cells','ipsilateral_cells','contralateral_cells','ipsilateral_percent','contralateral_percent']:
                vals=t[metric].dropna().to_numpy();mean,sem=mean_sem(vals)
                summaries.append(dict(k=k,cluster=cluster,region='all_PFC',metric=metric,n_injections=len(t),
                    n_animals=t[['strain','cohort','animal_id']].drop_duplicates().shape[0],n_values=len(vals),mean=mean,sem=sem))
            for region in regional.region.unique():
                for side in SIDES:
                    vals=r.loc[r.region.eq(region)&r.laterality.eq(side),'n_cells'].to_numpy()
                    mean,sem=mean_sem(vals)
                    summaries.append(dict(k=k,cluster=cluster,region=region,metric=side+'_cells',n_injections=len(t),
                        n_animals=t[['strain','cohort','animal_id']].drop_duplicates().shape[0],n_values=len(vals),mean=mean,sem=sem))
        fig=make_cluster_figure(totals,regional,k)
        for ext in ['png','pdf','svg']:fig.savefig(OUTPUT_DIR/f'pfc_laterality_{k}_clusters.{ext}',dpi=300)
        if SHOW_FIGURES:plt.show()

    pd.DataFrame(summaries).to_csv(OUTPUT_DIR/'cluster_laterality_summary.csv',index=False)
    print(f'DONE! {len(totals)} injections. Saved in {OUTPUT_DIR}')

if __name__=='__main__':
    main()
