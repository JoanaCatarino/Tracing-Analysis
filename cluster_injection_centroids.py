"""Spatial Ward clustering of injection centroids. Run in Spyder or with Python.
Dependencies: numpy pandas scipy scikit-learn matplotlib.
3D HTML uses Plotly.js from its CDN (internet needed when opening).
Atlas meshes: reuse 997.obj, 672.obj and 56.obj from the reconstruction.
"""
from pathlib import Path
import json
import re
import html
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score

BASE = Path(r'L:\dmclab\Joana\Tracing\Analysis')
INPUT_FILE = BASE / 'injection_centroids' / 'centroids_for_clustering.csv'
OUTPUT_DIR = BASE / 'injection_centroids' / 'clustering'
ATLAS_MESH_DIR = BASE / 'allen_mesh'
OVERVIEW_FILE = Path(r"L:\dmclab\Joana\Tracing\tracing_overview.xlsx")
CANDIDATE_K = [2, 3, 4, 5]
FINAL_K = None  # Inspect candidates first; optionally set 2, 3, 4 or 5.
N_RESAMPLES = 300
ANIMAL_SUBSAMPLE_FRACTION = 0.8
SEED = 42
LINES = ['Tlx3', 'Fezf2', 'Fmr1-Fezf2', 'Fmr1-Tlx3']
COLORS = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#AA3377']
AXES = ['ap', 'ml', 'dv']
COORDS = [f'sampled_common_{a}_um' for a in AXES]  # Same definition for every injection.


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
                                "cells": values.get("cells", "").strip().lower(),
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


def labels_for(x, k):
    raw = cut_tree(linkage(x, method='ward'), n_clusters=k).ravel()
    # Number groups by increasing Allen AP coordinate within each candidate.
    order = sorted(np.unique(raw), key=lambda v: tuple(x[raw == v].mean(axis=0)))
    return np.array([order.index(v) + 1 for v in raw])


def read_obj(path):
    v, f = [], []
    for line in path.read_text().splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == 'v':
            v.append(list(map(float, t[1:4])))
        elif t[0] == 'f':
            ids = [int(s.split('/')[0]) for s in t[1:]]
            ids = [i-1 if i > 0 else len(v)+i for i in ids]
            f.extend([[ids[0], ids[j+1], ids[j]] for j in range(1, len(ids)-1)])
    v, f = np.asarray(v), np.asarray(f)
    if v.ndim != 2 or v.shape[1] != 3 or not len(f):
        raise ValueError(f'Invalid mesh {path}')
    return v[:, [0, 2, 1]] / 1000, f  # OBJ AP,DV,ML um -> AP,ML,DV mm


def viewer(d, x, assignments, out, mesh_dir, notes):
    atlas = []
    for ident, name, color, opacity in [(997, 'Brain', '#BBBBBB', .06),
            (672, 'CP', '#55A9D5', .22), (56, 'ACB', '#E6A14A', .35)]:
        path = mesh_dir / f'{ident}.obj'
        if not path.is_file():
            notes.append(f'3D overlay missing: {path}. Reuse meshes from the reconstruction and rerun.')
            continue
        v, f = read_obj(path)
        atlas.append(dict(type='mesh3d', name=name, showlegend=True, color=color, opacity=opacity,
            x=v[:,0].tolist(), y=v[:,1].tolist(), z=v[:,2].tolist(),
            i=f[:,0].tolist(), j=f[:,1].tolist(), k=f[:,2].tolist(), hoverinfo='name'))
    candidates = {}
    for k, labels in assignments.items():
        traces = []
        for c in range(1, k+1):
            mask = labels == c
            traces.append(dict(type='scatter3d', mode='markers', name=f'Cluster {c}',
                x=x[mask,0].tolist(), y=x[mask,1].tolist(), z=x[mask,2].tolist(),
                text=[html.escape(str(s)) for s in d.loc[mask, 'injection_id']],
                marker=dict(size=5, color=COLORS[(c-1)%len(COLORS)]),
                hovertemplate='%{text}<br>AP %{x:.3f} mm<br>ML %{y:.3f} mm<br>DV %{z:.3f} mm<extra></extra>'))
        candidates[str(k)] = traces
    payload = json.dumps(dict(atlas=atlas, candidates=candidates)).replace('</', '<\\/')
    options = ''.join(f'<option>{k}</option>' for k in assignments)
    page = '''<!doctype html><meta charset="utf-8"><title>Injection centroid clusters</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{font:16px Arial;margin:24px}#plot{height:82vh}</style>
<h2>Injection centroid clusters</h2><label>Number of clusters: <select id="k">OPTIONS</select></label>
<p>Common hemisphere, Allen coordinates in mm. Rotate/zoom and click legend to toggle structures.
Internet is needed to load Plotly. Missing atlas meshes are listed in README_results.txt.</p><div id="plot"></div>
<script>const data=PAYLOAD;const layout={uirevision:'keep',scene:{aspectmode:'data',
xaxis:{title:'Allen AP (mm)'},yaxis:{title:'Allen ML (mm)'},zaxis:{title:'Allen DV (mm)',autorange:'reversed'}}};
function draw(){Plotly.react('plot',[...data.atlas,...data.candidates[document.getElementById('k').value]],layout,{responsive:true});}
document.getElementById('k').onchange=draw;draw();</script>'''
    (out/'clusters_3d.html').write_text(page.replace('OPTIONS', options).replace('PAYLOAD', payload), encoding='utf-8')


def run(input_file=INPUT_FILE, output_dir=OUTPUT_DIR, mesh_dir=ATLAS_MESH_DIR, overview_file=OVERVIEW_FILE):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(input_file, dtype={'animal_id':str, 'cohort':str}).sort_values('injection_id').reset_index(drop=True)
    required = ['injection_id','strain','cohort','animal_id','channel','common_hemisphere'] + COORDS
    missing = set(required)-set(d.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if d.injection_id.duplicated().any():
        raise ValueError('Duplicate injection IDs in centroid input.')
    overview = read_injection_overview(Path(overview_file))
    overview['injection_id'] = overview[['strain','cohort','animal_id','channel']].agg('_'.join, axis=1)
    keep = overview.striatum_percent.between(80,100) & overview.cells.eq('y')
    overview.loc[~keep].to_csv(out/'excluded_injections.csv', index=False)
    eligible = overview.loc[keep].copy()
    # Left join from the overview prevents missing injections from disappearing.
    payload = [col for col in d if col not in eligible.columns or col == 'injection_id']
    d = eligible.merge(d[payload], on='injection_id', how='left', validate='one_to_one')
    d = d.sort_values('injection_id').reset_index(drop=True)
    invalid = d[required].isna().any(axis=1)
    d.assign(missing_centroid=invalid).to_csv(out/'eligibility_audit.csv', index=False)
    if invalid.any():
        raise ValueError('Eligible injections missing centroids: ' + ', '.join(d.loc[invalid,'injection_id']) +
                         '. Inspect eligibility_audit.csv and rerun the centroid script; no injections were silently dropped.')
    notes = [f'Eligibility verified against overview: cells=y and % Striatum >=80; {len(d)} injections.']
    hemis = d.common_hemisphere.astype(str).str.strip().str.lower().unique()
    if len(hemis)!=1 or hemis[0] not in ['left','right']:
        raise ValueError('All injections must be mapped to one common hemisphere.')
    x = d[COORDS].apply(pd.to_numeric, errors='raise').to_numpy(float)/1000
    if not np.isfinite(x).all() or len(x)<4:
        raise ValueError('Need >=4 injections with finite coordinates.')
    d['animal_key'] = d.strain.astype(str)+'|'+d.cohort.astype(str)+'|'+d.animal_id
    animals = d.animal_key.unique()
    distances = squareform(pdist(x)); pairs=[]
    for i in range(len(d)):
        for j in range(i+1,len(d)):
            if distances[i,j] <= .025:
                pairs.append(dict(injection_1=d.injection_id[i], injection_2=d.injection_id[j],
                    distance_um=distances[i,j]*1000, same_animal=d.animal_key[i]==d.animal_key[j]))
    pd.DataFrame(pairs, columns=['injection_1','injection_2','distance_um','same_animal']).to_csv(out/'near_duplicate_locations.csv', index=False)
    notes.append(f'{len(pairs)} pairs within 25 um. Kept as separate injections; inspect shared-mask cases.')
    counts=d.groupby('strain').agg(injections=('injection_id','size'), animals=('animal_key','nunique')).reindex(LINES,fill_value=0)
    counts.to_csv(out/'line_counts.csv')
    notes.append(f'{len(d)} injections, {len(animals)} animals; common hemisphere: {hemis[0]}.')
    notes.append('Missing expected lines: '+', '.join(counts.index[counts.injections==0]))
    assignments={}; metrics=[]; composition=[]; centers=[]
    rng=np.random.default_rng(SEED)
    z=linkage(x, method='ward', optimal_ordering=True)
    fig,ax=plt.subplots(figsize=(11,9))
    dendrogram(z, labels=d.injection_id.tolist(), orientation='right', leaf_font_size=7, color_threshold=0, ax=ax)
    ax.set_xlabel('Ward linkage height (coordinates in mm; not raw pairwise distance)')
    ax.set_title('Injection locations: Ward hierarchical clustering')
    fig.tight_layout(); fig.savefig(out/'dendrogram.png', dpi=200); fig.savefig(out/'dendrogram.pdf'); plt.close(fig)
    ks=[k for k in CANDIDATE_K if 2<=k<len(d)]
    if not ks:
        raise ValueError('No eligible candidate k.')
    fig, axs=plt.subplots(len(ks),3,figsize=(12,3.4*len(ks)),squeeze=False)
    for row,k in enumerate(ks):
        labels=labels_for(x,k); assignments[k]=labels; d[f'cluster_k{k}']=labels
        ari=[]
        for _ in range(N_RESAMPLES):
            chosen=rng.choice(animals, size=max(2,int(np.ceil(len(animals)*ANIMAL_SUBSAMPLE_FRACTION))),replace=False)
            idx=np.flatnonzero(d.animal_key.isin(chosen).to_numpy())
            if len(idx)>k:
                ari.append(adjusted_rand_score(labels[idx], labels_for(x[idx],k)))
        sizes=np.bincount(labels)[1:]
        metric=dict(k=k,silhouette=silhouette_score(x,labels),smallest_cluster=int(sizes.min()),
            largest_cluster=int(sizes.max()),animal_subsample_ARI_mean=np.mean(ari) if ari else np.nan,
            animal_subsample_ARI_p10=np.quantile(ari,.1) if ari else np.nan,
            valid_resamples=len(ari))
        sampled=[f'reconstructed_common_{a}_um' for a in AXES]
        if all(c in d for c in sampled):
            sx=d[sampled].to_numpy(float)/1000
            if np.isfinite(sx).all():
                metric['sampled_vs_reconstructed_ARI']=adjusted_rand_score(labels,labels_for(sx,k))
        metrics.append(metric)
        for c in range(1,k+1):
            sub=d.loc[labels==c]
            for line in list(dict.fromkeys(LINES+d.strain.tolist())):
                s=sub[sub.strain==line]
                composition.append(dict(k=k,cluster=c,strain=line,injections=len(s),animals=s.animal_key.nunique()))
            center=x[labels==c].mean(axis=0)
            centers.append(dict(k=k,cluster=c,injections=len(sub),animals=sub.animal_key.nunique(),
                **{f'mean_{a}_mm':center[i] for i,a in enumerate(AXES)}))
        for col,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
            ax=axs[row,col]
            for c in range(1,k+1):
                mask=labels==c
                ax.scatter(x[mask,a],x[mask,b],s=40,c=COLORS[c-1],label=f'C{c} (n={mask.sum()})',edgecolors='white',linewidths=.5)
            ax.set(xlabel=f'Allen {AXES[a].upper()} (mm)',ylabel=f'Allen {AXES[b].upper()} (mm)',title=f'k={k}')
            ax.set_aspect('equal',adjustable='box'); ax.grid(alpha=.15)
            if b==2: ax.invert_yaxis()
            if col==0: ax.legend(fontsize=7)
    fig.suptitle('Candidate spatial groups — one point per injection',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(out/'candidate_clusters.png',dpi=200); fig.savefig(out/'candidate_clusters.pdf');plt.close(fig)
    pd.DataFrame(metrics).to_csv(out/'candidate_metrics.csv',index=False)
    pd.DataFrame(composition).to_csv(out/'cluster_line_counts.csv',index=False)
    pd.DataFrame(centers).to_csv(out/'cluster_centers.csv',index=False)
    d.to_csv(out/'injection_cluster_candidates.csv',index=False)
    if FINAL_K is not None:
        if FINAL_K not in assignments: raise ValueError('FINAL_K must be in candidate list.')
        d.assign(cluster=assignments[FINAL_K]).to_csv(out/'final_cluster_assignments.csv',index=False)
    elif (out/'final_cluster_assignments.csv').exists():
        (out/'final_cluster_assignments.csv').unlink()  # Prevent stale final assignments.
    viewer(d,x,assignments,out,Path(mesh_dir),notes)
    notes += ['Coordinates: sampled common AP/ML/DV divided by 1000; no axis standardization or additional mirroring.',
        'Method: Ward linkage with Euclidean distance. All available lines pooled; line and counts are not clustering inputs.',
        'Stability: 300 random subsets of 80% of animals (all their injections kept). ARI compares re-clustering to full-data labels on that subset. It is a sensitivity diagnostic, not a confidence interval.',
        'Sampled-vs-reconstructed ARI is reported only when every injection has a reconstructed centroid.',
        'Cluster IDs are ordered by Allen AP within each k, and may change when data or k changes.',
        'No final k selected automatically. High silhouette/stability does not establish biological subregions.',
        'Animal counts within clusters cannot be summed to obtain distinct animals overall.',
        'Check injection extent/overlap separately. Downstream analyses must account for repeated animals.',
        'Rerunning refits all clusters because new injections can change the hierarchy.',
        'Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html']
    (out/'README_results.txt').write_text('\n'.join(notes),encoding='utf-8')
    print(counts.to_string());print(pd.DataFrame(metrics).to_string(index=False));print('\n'.join(notes[:4]));print(f'DONE! {out}')
    return d, pd.DataFrame(metrics)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=INPUT_FILE)
    parser.add_argument('--output',type=Path,default=OUTPUT_DIR)
    parser.add_argument('--meshes',type=Path,default=ATLAS_MESH_DIR)
    parser.add_argument('--overview',type=Path,default=OVERVIEW_FILE)
    args=parser.parse_args()
    run(args.input,args.output,args.meshes,args.overview)
