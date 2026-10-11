"""Exploratory injection comparability; run in Spyder after reconstruction.

Required: current overview + injection_centroids.csv from calculate_injection_centroids.py.
Reconstructed NPZs enable overlap; absent/failed masks are reported, never treated as zero overlap.
Optional: cluster assignments, source segmentation CSVs and flat-map regional exports.
Dependencies: numpy pandas matplotlib scipy openpyxl. No internet access required.
All coordinates are Allen AP/ML/DV, micrometres, aligned to the recorded common hemisphere.
This is descriptive exploration: candidate neighbours are NOT validated biological matches.
"""
from pathlib import Path
from itertools import combinations, product
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

BASE_DIR = Path(r'L:\dmclab\Joana\Tracing\Analysis')
OVERVIEW_FILE = Path(r"L:\dmclab\Joana\Tracing\tracing_overview.xlsx")
CENTROIDS_FILE = BASE_DIR / 'injection_centroids' / 'injection_centroids.csv'
RECONSTRUCTION_DIR = BASE_DIR / 'injection_centroids' / 'reconstruction_3d'
CLUSTER_FILE = Path(r'L:/dmclab/Joana/Tracing/Analysis/injection_centroids/clustering/injection_cluster_candidates.csv')
PFC_REGIONAL_FILE = BASE_DIR / 'plots' / 'pfc_flatmaps_by_cluster_percent' / 'presentation' / 'regional_percentages_per_injection.csv'
OUTPUT_DIR = BASE_DIR / 'inj_quantif_exploratory'
MIN_STRIATUM_PERCENT = 80.
# Shared, atlas-aligned voxel grid. Repeat at 25 and 50 um to assess discretisation.
OVERLAP_GRID_UM = 25.
SENSITIVITY_GRID_UM = [50.]
MAX_GRID_POINTS_PER_INJECTION = 30_000_000
CHUNK_SIZE = 250_000
TOP_CANDIDATES_PER_OTHER_LINE = 3
LINE_ORDER = ['Tlx3', 'Fezf2', 'Fmr1-Fezf2', 'Fmr1-Tlx3']
LINE_COLORS = {'Tlx3':'#DC8D00','Fezf2':'#A968A9','Fmr1-Fezf2':'#1D9999','Fmr1-Tlx3':'#80C7C5'}
AXES = ['ap','ml','dv']
COORDS = [f'sampled_common_{a}_um' for a in AXES]
FORMATS = ['png','pdf']
# Leave None unless you deliberately choose anatomical criteria after inspection.
# All criteria must be set to export a threshold-filtered candidate table.
MATCH_MIN_DICE = None
MATCH_MAX_DISTANCE_UM = None
MATCH_MIN_VOLUME_RATIO = None

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


def savefig(fig, name):
    for ext in FORMATS:
        fig.savefig(OUTPUT_DIR / f'{name}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def load_jobs():
    overview=read_injection_overview(OVERVIEW_FILE)
    overview['injection_id']=overview[['strain','cohort','animal_id','channel']].agg('_'.join,axis=1)
    overview['eligible']=overview.cells.eq('y') & overview.striatum_percent.between(MIN_STRIATUM_PERCENT,100)
    overview.to_csv(OUTPUT_DIR/'overview_eligibility.csv',index=False)
    eligible=overview.loc[overview.eligible].copy()
    if eligible.empty:raise ValueError('No overview injections pass cells=y and % Striatum inclusion.')
    d=pd.read_csv(CENTROIDS_FILE,dtype={'animal_id':str,'cohort':str})
    if d.injection_id.duplicated().any():raise ValueError('Duplicate centroid injection_id.')
    # Revalidate current overview side: an old reconstruction must not silently be reused.
    geometry_cols=[c for c in d if c not in overview.columns or c in ['injection_id','injection_side']]
    d=d[geometry_cols].rename(columns={'injection_side':'reconstructed_injection_side'})
    jobs=eligible.merge(d,on='injection_id',how='left',validate='one_to_one')
    for col in COORDS:jobs[col]=pd.to_numeric(jobs.get(col,np.nan),errors='coerce')
    good=np.isfinite(jobs[COORDS]).all(axis=1)
    good &= jobs.reconstructed_injection_side.eq(jobs.injection_side)
    jobs['centroid_usable']=good
    jobs.to_csv(OUTPUT_DIR/'input_audit.csv',index=False)
    if not good.all():
        raise ValueError('Missing/invalid centroids or changed injection side. See input_audit.csv; rerun calculate_injection_centroids.py against this overview.')
    if 'common_hemisphere' not in jobs or jobs.common_hemisphere.isna().any() or jobs.common_hemisphere.nunique()!=1:
        raise ValueError('All centroids must use one recorded common hemisphere.')
    for col in ['atlas_voxel_um','ml_midline_atlas_coord']:
        if col not in jobs or jobs[col].isna().any() or jobs[col].nunique()!=1:
            raise ValueError(f'Missing/inconsistent {col}; rerun current reconstruction script.')
    jobs['animal_key']=jobs[['strain','cohort','animal_id']].astype(str).agg('|'.join,axis=1)
    jobs['line_order']=jobs.strain.map({v:i for i,v in enumerate(LINE_ORDER)})
    jobs=jobs.sort_values(['line_order','injection_id']).reset_index(drop=True)
    if CLUSTER_FILE.is_file():
        c=pd.read_csv(CLUSTER_FILE)
        cols=[x for x in c if re.fullmatch(r'cluster_k\d+',x)]
        if c.injection_id.duplicated().any():raise ValueError('Duplicate cluster IDs.')
        jobs=jobs.merge(c[['injection_id']+cols],on='injection_id',how='left',validate='one_to_one')
    return jobs


def load_geometry(row):
    path=RECONSTRUCTION_DIR/f'{row.injection_id}.npz'
    if row.get('reconstruction_status','')!='OK':raise ValueError('Centroid export reports unsuccessful reconstruction')
    with np.load(path,allow_pickle=False) as f:
        required=['occupied_grid_indices','grid_origin_local_um','grid_step_um','local_to_atlas_basis']
        if any(k not in f for k in required):raise ValueError('NPZ lacks occupied volume; rerun reconstruction')
        occ=np.asarray(f[required[0]],dtype=np.int64)
        origin=np.asarray(f[required[1]],float);step=float(f[required[2]])
        basis=np.asarray(f[required[3]],float)
    if occ.ndim!=2 or occ.shape[1]!=3 or not len(occ) or (occ<0).any():raise ValueError('Invalid occupancy indices')
    if step<=0 or not np.isfinite(step) or origin.shape!=(3,) or not np.isfinite(origin).all():raise ValueError('Invalid grid')
    if basis.shape!=(3,3) or not np.allclose(basis.T@basis,np.eye(3),atol=1e-6):raise ValueError('Invalid atlas basis')
    expected=(occ.mean(axis=0)*step+origin)@basis.T
    saved=np.array([row.get(f'reconstructed_common_{a}_um',np.nan) for a in AXES],float)
    if not np.allclose(expected,saved,atol=.01):raise ValueError('NPZ and centroid export disagree; regenerate together')
    lower=occ.min(axis=0);upper=occ.max(axis=0)
    corners=np.array(list(product(*zip(lower-.5,upper+.5))))
    atlas_corners=(corners*step+origin)@basis.T
    return dict(occ=occ,origin=origin,step=step,basis=basis,
                bounds=np.array([atlas_corners.min(axis=0),atlas_corners.max(axis=0)]))


def common_grid_mask(g, spacing):
    """Inverse sample each oriented binary mask at shared atlas voxel centres.

    Do NOT round occupied source centres into a new grid: that creates holes
    and makes overlap depend on each injection's local section orientation.
    """
    lower=np.floor(g['bounds'][0]/spacing).astype(np.int64)
    upper=np.ceil(g['bounds'][1]/spacing).astype(np.int64)
    shape=upper-lower+1;total=int(np.prod(shape))
    if total>MAX_GRID_POINTS_PER_INJECTION:raise ValueError('Common grid too large; inspect extent or increase OVERLAP_GRID_UM')
    occ=g['occ'];src_shape=occ.max(axis=0)+1
    codes=np.unique(np.ravel_multi_index(occ.T,tuple(src_shape)))
    chunks=[]
    for start in range(0,total,CHUNK_SIZE):
        ids=np.arange(start,min(total,start+CHUNK_SIZE))
        ijk=np.column_stack(np.unravel_index(ids,tuple(shape)))+lower
        local=(ijk*spacing)@g['basis']
        src=np.floor((local-g['origin'])/g['step']+.5).astype(np.int64)
        valid=((src>=0)&(src<src_shape)).all(axis=1)
        hit=np.zeros(len(ijk),bool)
        hit[valid]=np.isin(np.ravel_multi_index(src[valid].T,tuple(src_shape)),codes,assume_unique=False)
        chunks.append(ijk[hit])
    result=np.concatenate(chunks)
    if not len(result):raise ValueError('Reconstruction disappears at common-grid resolution')
    return result


def row_codes(ijk):
    return np.ascontiguousarray(ijk,dtype=np.int64).view(np.dtype((np.void,24))).ravel()


def pairwise(jobs,masks,spacing):
    out=[]
    for i,j in combinations(range(len(jobs)),2):
        a,b=jobs.iloc[i],jobs.iloc[j]
        diff=b[COORDS].to_numpy(float)-a[COORDS].to_numpy(float)
        r=dict(injection_a=a.injection_id,injection_b=b.injection_id,line_a=a.strain,line_b=b.strain,
               animal_a=a.animal_key,animal_b=b.animal_key,same_animal=a.animal_key==b.animal_key,
               centroid_distance_um=float(np.linalg.norm(diff)),grid_um=spacing,
               overlap_available=i in masks and j in masks)
        r.update({f'abs_{axis}_difference_um':abs(diff[k]) for k,axis in enumerate(AXES)})
        for suffix,job in [('a',a),('b',b)]:
            for key in COORDS+['estimated_reconstructed_volume_mm3','sampled_cp_percent','sampled_acb_percent','striatum_percent']:
                r[key+'_'+suffix]=job.get(key,np.nan)
        for col in jobs:
            if re.fullmatch(r'cluster_k\d+',col):
                r[col+'_a']=a[col];r[col+'_b']=b[col]
                r['same_'+col]=a[col]==b[col] if pd.notna(a[col]) and pd.notna(b[col]) else np.nan
        for key in ['dice','jaccard','intersection_mm3','fraction_a_covered','fraction_b_covered','volume_ratio_small_large']:
            r[key]=np.nan
        if r['overlap_available']:
            va,vb=len(masks[i]),len(masks[j]);inter=len(np.intersect1d(row_codes(masks[i]),row_codes(masks[j]),assume_unique=True))
            r.update(dice=2*inter/(va+vb),jaccard=inter/(va+vb-inter),intersection_mm3=inter*spacing**3/1e9,
                     fraction_a_covered=inter/va,fraction_b_covered=inter/vb,volume_ratio_small_large=min(va,vb)/max(va,vb))
        out.append(r)
    return pd.DataFrame(out)


def sampled_composition(row):
    """Area-sample composition, not an atlas-labelled reconstructed-volume estimate."""
    path=Path(row.get('source_csv',''))
    raw=pd.read_csv(path)
    ml=pd.to_numeric(raw.ml_mm,errors='raise')
    d=raw.loc[ml.lt(0) if row.injection_side=='left' else ml.gt(0)].copy()
    if d.empty:raise ValueError('No mask points on injection side')
    if 'acronym' not in d:raise ValueError('Source mask lacks acronym; cannot quantify CP/ACB')
    weight_col=str(row.get('centroid_weight_basis',''))
    if weight_col.startswith('equal_area_points'):
        w=np.ones(len(d))
    elif weight_col.endswith('_x_spacing') and weight_col[:-10] in d:
        w=pd.to_numeric(d[weight_col[:-10]],errors='raise').to_numpy(float)
    else:raise ValueError('Unrecognised area weighting; CP/ACB composition not estimated')
    if not np.isfinite(w).all() or (w<=0).any():raise ValueError('Invalid area weights')
    ac=d.acronym.fillna('').astype(str).str.strip()
    cp=ac.eq('CP').to_numpy();acb=ac.eq('ACB').to_numpy()
    return dict(sampled_cp_percent=100*w[cp].sum()/w.sum(),sampled_acb_percent=100*w[acb].sum()/w.sum(),
                sampled_other_percent=100*w[~(cp|acb)].sum()/w.sum(),composition_basis='selected-side mask pixels weighted by area; uniform section spacing')


def spatial_plots(jobs,geometry):
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for ax,(u,v) in zip(axes,[(0,1),(0,2),(1,2)]):
        for i,row in jobs.iterrows():
            center=row[COORDS].to_numpy(float)/1000;color=LINE_COLORS[row.strain]
            if i in geometry:
                g=geometry[i];xyz=(g['occ']*g['step']+g['origin'])@g['basis'].T/1000
                take=np.linspace(0,len(xyz)-1,min(len(xyz),1500),dtype=int)
                ax.scatter(xyz[take,u],xyz[take,v],s=1,color=color,alpha=.035,rasterized=True)
                bounds=np.array([xyz.min(axis=0),xyz.max(axis=0)])
                ax.plot(bounds[:,u],[center[v]]*2,color=color,alpha=.4,lw=.6)
                ax.plot([center[u]]*2,bounds[:,v],color=color,alpha=.4,lw=.6)
            ax.scatter(center[u],center[v],color=color,s=28,edgecolor='black',lw=.3)
            ax.annotate(str(i+1),(center[u],center[v]),xytext=(3,3),textcoords='offset points',fontsize=6)
        ax.set_xlabel('Allen '+AXES[u].upper()+' (mm)');ax.set_ylabel('Allen '+AXES[v].upper()+' (mm)');ax.set_aspect('equal',adjustable='datalim')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=LINE_COLORS[l],label=l) for l in LINE_ORDER],loc='outside lower center',ncol=4)
    fig.suptitle('Injection centroids and reconstructed extent · common '+str(jobs.common_hemisphere.iloc[0])+' hemisphere\nNumbers identify injections in injection_summary.csv; faint points show reconstructed volume samples')
    savefig(fig,'injection_spatial_extent')


def heatmaps(jobs,pairs):
    n=len(jobs);lookup={x:i for i,x in enumerate(jobs.injection_id)}
    for metric,limit in [('dice',1),('jaccard',1),('centroid_distance_um',None)]:
        matrix=np.full((n,n),np.nan)
        for _,r in pairs.iterrows():
            i,j=lookup[r.injection_a],lookup[r.injection_b];matrix[i,j]=matrix[j,i]=r[metric]
        # Diagonal is intentionally omitted; grey means unavailable or self-comparison.
        pd.DataFrame(matrix,index=jobs.injection_id,columns=jobs.injection_id).to_csv(OUTPUT_DIR/f'{metric}_matrix.csv')
        fig,ax=plt.subplots(figsize=(max(8,n*.23),max(7,n*.23)))
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dddddd')
        image=ax.imshow(matrix,cmap=cmap,vmin=0,vmax=limit)
        labels=[f'{i+1} {r.strain}' for i,r in jobs.iterrows()]
        ax.set_xticks(range(n),labels,rotation=90,fontsize=6);ax.set_yticks(range(n),labels,fontsize=6)
        for tick,(_,r) in zip(ax.get_yticklabels(),jobs.iterrows()):tick.set_color(LINE_COLORS[r.strain])
        fig.colorbar(image,ax=ax,label=metric.replace('_',' '));ax.set_title(metric.replace('_',' ')+' · grey = unavailable / diagonal')
        savefig(fig,metric+'_heatmap')


def size_composition_plots(jobs):
    metrics=[('estimated_reconstructed_volume_mm3','Estimated reconstructed volume (mm³)'),
             ('striatum_percent','Overview striatum (%)'),('sampled_cp_percent','Sampled CP (%)'),('sampled_acb_percent','Sampled ACB (%)')]
    fig,axes=plt.subplots(1,4,figsize=(16,4),layout='constrained');rng=np.random.default_rng(42)
    for ax,(metric,label) in zip(axes,metrics):
        for j,line in enumerate(LINE_ORDER):
            vals=pd.to_numeric(jobs.loc[jobs.strain.eq(line),metric],errors='coerce').dropna()
            ax.scatter(j+rng.uniform(-.12,.12,len(vals)),vals,c=LINE_COLORS[line],s=24)
        ax.set_xticks(range(4),LINE_ORDER,rotation=30,ha='right');ax.set_ylabel(label);ax.set_ylim(bottom=0)
        if 'percent' in metric:ax.set_ylim(0,100)
        ax.spines[['top','right']].set_visible(False)
        if jobs[metric].isna().all():ax.text(.5,.5,'Unavailable: see audit',transform=ax.transAxes,ha='center')
    fig.suptitle('Each dot is one injection; CP/ACB percentages describe sampled masks, not reconstructed volumes')
    savefig(fig,'injection_size_and_composition')


def candidate_tables(pairs):
    directed=[]
    for _,r in pairs.loc[pairs.line_a.ne(pairs.line_b)].iterrows():
        for reverse in [False,True]:
            d=r.to_dict()
            if reverse:
                for key in list(d):
                    if key.endswith('_a') and key[:-2]+'_b' in d:
                        other=key[:-2]+'_b';d[key],d[other]=d[other],d[key]
                d['fraction_a_covered'],d['fraction_b_covered']=d['fraction_b_covered'],d['fraction_a_covered']
            directed.append(d)
    if not directed:return pd.DataFrame()
    d=pd.DataFrame(directed)
    d=d.sort_values(['injection_a','line_b','dice','centroid_distance_um'],ascending=[True,True,False,True],na_position='last')
    d['rank_by_overlap_then_distance']=d.groupby(['injection_a','line_b']).cumcount()+1
    d['rank_by_distance']=d.groupby(['injection_a','line_b']).centroid_distance_um.rank(method='min')
    d.to_csv(OUTPUT_DIR/'all_cross_line_candidates.csv',index=False)
    d.loc[d.rank_by_overlap_then_distance.le(TOP_CANDIDATES_PER_OTHER_LINE)].to_csv(OUTPUT_DIR/'top_cross_line_candidates.csv',index=False)
    settings=[MATCH_MIN_DICE,MATCH_MAX_DISTANCE_UM,MATCH_MIN_VOLUME_RATIO]
    if any(v is not None for v in settings):
        if any(v is None for v in settings):raise ValueError('Set all three MATCH criteria or leave all None')
        keep=d.dice.ge(MATCH_MIN_DICE)&d.centroid_distance_um.le(MATCH_MAX_DISTANCE_UM)&d.volume_ratio_small_large.ge(MATCH_MIN_VOLUME_RATIO)
        d.loc[keep].to_csv(OUTPUT_DIR/'threshold_filtered_candidates.csv',index=False)
    return d


def cluster_audit(jobs,pairs):
    records=[]
    for col in jobs:
        if not re.fullmatch(r'cluster_k\d+',col):continue
        for cluster,group in jobs.dropna(subset=[col]).groupby(col):
            within=pairs.loc[pairs.injection_a.isin(group.injection_id)&pairs.injection_b.isin(group.injection_id)]
            records.append(dict(solution=col,cluster=cluster,n_injections=len(group),n_animals=group.animal_key.nunique(),
                n_pairs=len(within),n_pairs_with_overlap=int(within.overlap_available.sum()),
                distance_median_um=within.centroid_distance_um.median(),distance_max_um=within.centroid_distance_um.max(),
                dice_median=within.dice.dropna().median() if within.dice.notna().any() else np.nan,
                dice_min=within.dice.min(),
                zero_overlap_pairs=int(within.dice.eq(0).sum()),
                **{f'n_{l}':int(group.strain.eq(l).sum()) for l in LINE_ORDER}))
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,metric in zip(axes,['centroid_distance_um','dice']):
            for c,group in jobs.dropna(subset=[col]).groupby(col):
                values=pairs.loc[pairs.injection_a.isin(group.injection_id)&pairs.injection_b.isin(group.injection_id),metric].dropna().to_numpy()
                ax.scatter(np.full(len(values),c)+np.linspace(-.12,.12,len(values)),values,s=12,alpha=.65)
            ax.set_xlabel('Cluster');ax.set_ylabel(metric.replace('_',' '));ax.set_ylim(bottom=0)
        fig.suptitle(col+' · each point is a pair, pairs are not independent')
        savefig(fig,'heterogeneity_'+col)
    pd.DataFrame(records).to_csv(OUTPUT_DIR/'within_cluster_heterogeneity.csv',index=False)


def pfc_exploration(jobs,candidates,notes):
    if not PFC_REGIONAL_FILE.is_file():
        notes.append('PFC exploration skipped: regional_percentages_per_injection.csv missing. Run the current flat-map script first.');return
    p=pd.read_csv(PFC_REGIONAL_FILE)
    required={'injection_id','region','laterality','percent'}
    if required-set(p):raise ValueError('PFC regional export lacks required columns')
    if p.duplicated(['injection_id','region','laterality']).any():raise ValueError('Duplicate PFC region/side rows')
    cols=['injection_id','region','laterality','percent']+[c for c in ['pfc_denominator','n_cells'] if c in p]
    d=p[cols].merge(jobs[['injection_id','strain','animal_key']+COORDS],on='injection_id',validate='many_to_one')
    if 'pfc_denominator' in d:
        d['percent']=d.percent.where(pd.to_numeric(d.pfc_denominator,errors='coerce').gt(0))
    d.to_csv(OUTPUT_DIR/'pfc_regional_percentages_with_position.csv',index=False)
    missing=set(jobs.injection_id)-set(d.injection_id)
    if missing:notes.append('Missing PFC exports for: '+', '.join(sorted(missing)))
    for side,subset in d.groupby('laterality'):
        names=list(subset.region.unique());fig,axes=plt.subplots(len(names),3,figsize=(13,2.5*len(names)),squeeze=False,layout='constrained')
        for i,region in enumerate(names):
            for j,axis in enumerate(AXES):
                ax=axes[i,j]
                for line,g in subset.loc[subset.region.eq(region)].groupby('strain'):
                    ax.scatter(g[COORDS[j]]/1000,g.percent,color=LINE_COLORS[line],label=line,s=22)
                ax.set_xlabel('Allen '+axis.upper()+' (mm)');ax.set_ylabel(region+' (% bilateral PFC)');ax.set_ylim(bottom=0)
        axes[0,0].legend(fontsize=7);fig.suptitle(side+' · PFC distribution versus injection position (individual injections)')
        savefig(fig,'pfc_vs_position_'+side)
    if not candidates.empty:
        # Descriptive differences for every directed candidate; no pseudo-replicated tests.
        differences=candidates.merge(d[['injection_id','region','laterality','percent']].rename(columns={'injection_id':'injection_a','percent':'percent_a'}),on='injection_a')
        differences=differences.merge(d[['injection_id','region','laterality','percent']].rename(columns={'injection_id':'injection_b','percent':'percent_b'}),on=['injection_b','region','laterality'])
        differences['difference_b_minus_a_percentage_points']=differences.percent_b-differences.percent_a
        differences.to_csv(OUTPUT_DIR/'candidate_pfc_differences.csv',index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    grids=list(dict.fromkeys([float(OVERLAP_GRID_UM)]+list(map(float,SENSITIVITY_GRID_UM))))
    if any(not np.isfinite(g) or g<=0 for g in grids):raise ValueError('Grid spacings must be positive')
    jobs=load_jobs();jobs['plot_number']=np.arange(1,len(jobs)+1)
    if len(jobs)<2:raise ValueError('At least two eligible injections required')
    notes=[];geometry={};audit=[]
    for i,row in jobs.iterrows():
        r=dict(injection_id=row.injection_id,reconstruction='unavailable',composition='unavailable')
        try:geometry[i]=load_geometry(row);r['reconstruction']='OK'
        except (OSError,KeyError,ValueError) as exc:r['reconstruction_error']=str(exc)
        try:
            composition=sampled_composition(row)
            for k,v in composition.items():jobs.loc[i,k]=v
            r['composition']='OK'
        except (OSError,KeyError,ValueError,AttributeError) as exc:r['composition_error']=str(exc)
        audit.append(r)
    for col in ['estimated_reconstructed_volume_mm3','sampled_cp_percent','sampled_acb_percent','sampled_other_percent']:
        if col not in jobs:jobs[col]=np.nan
    pd.DataFrame(audit).to_csv(OUTPUT_DIR/'mask_and_composition_audit.csv',index=False)
    spatial_plots(jobs,geometry);size_composition_plots(jobs)
    all_pairs=[];primary=None
    for spacing in grids:
        masks={};grid_audit=[]
        for i,g in geometry.items():
            try:
                masks[i]=common_grid_mask(g,spacing)
                grid_audit.append(dict(injection_id=jobs.iloc[i].injection_id,status='OK',grid_um=spacing,
                    common_grid_volume_mm3=len(masks[i])*spacing**3/1e9))
            except ValueError as exc:grid_audit.append(dict(injection_id=jobs.iloc[i].injection_id,status='ERROR',message=str(exc),grid_um=spacing))
        pd.DataFrame(grid_audit).to_csv(OUTPUT_DIR/f'grid_audit_{spacing:g}um.csv',index=False)
        pairs=pairwise(jobs,masks,spacing);all_pairs.append(pairs)
        pairs.to_csv(OUTPUT_DIR/f'pairwise_{spacing:g}um.csv',index=False)
        if spacing==OVERLAP_GRID_UM:primary=pairs
        print(f'{spacing:g} um grid: {len(masks)}/{len(jobs)} reconstructed masks available')
    pd.concat(all_pairs,ignore_index=True).to_csv(OUTPUT_DIR/'overlap_resolution_sensitivity.csv',index=False)
    jobs.to_csv(OUTPUT_DIR/'injection_summary.csv',index=False)
    heatmaps(jobs,primary);candidates=candidate_tables(primary);cluster_audit(jobs,primary)
    pfc_exploration(jobs,candidates,notes)
    if len(geometry)<len(jobs):notes.append(f'{len(jobs)-len(geometry)} injections lack valid reconstruction; overlap stays NaN, not zero.')
    config={k:str(v) if isinstance(v,Path) else v for k,v in globals().items() if k.isupper() and isinstance(v,(Path,str,int,float,list,dict,type(None)))}
    (OUTPUT_DIR/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (OUTPUT_DIR/'README_results.txt').write_text('''EXPLORATORY INJECTION COMPARABILITY
Start with injection_summary.csv and injection_spatial_extent.png; plot numbers link the figures and tables.
Inspect dice_heatmap.png and centroid_distance_um_heatmap.png together with injection_size_and_composition.png.
Dice = 2 intersection / (volume A + volume B). Jaccard = intersection / union.
Directional coverage reports what fraction of each injection is sampled by both masks.
Overlaps use all reconstructed selected-side tissue, including spillover; CP/ACB composition is separate.
Coordinates are Allen atlas coordinates, NOT bregma coordinates. No per-injection translation or size rescaling.
The masks are resampled by inverse lookup onto a fixed common atlas grid. Resolution sensitivity is exported.
Reconstructed masks estimate tissue between sampled sections; grid resolution is not histological precision.
CP/ACB values use source segmentation region labels and area weights with uniform section spacing, not reconstructed volume labels.
Missing reconstruction is not an exclusion from centroid analyses and is never interpreted as zero overlap.
Top candidates are independently ranked per injection and other line by Dice, then distance. Missing overlap ranks last.
These are suggestions for anatomical review; matches may be reused and are not one-to-one independent pairs.
Use distance rank, size ratio, directional coverage and CP/ACB distribution alongside Dice. No automatic universal threshold.
Threshold-filtered candidates are produced only if all three MATCH settings are explicitly set.
Within-cluster summaries expose heterogeneous clusters without redefining clusters using the PFC outcomes.
PFC position plots and candidate differences are descriptive; dots are injections, not independent animals.
No hypothesis tests or adjusted line-effect model are fitted automatically. First establish comparable spatial coverage.
Related/repeated injections and dependent pairs must not be treated as independent replicates.
Existing output filenames are overwritten; unrelated/older optional outputs are retained. See run_config.json for this run.

RUN NOTES
'''+ '\n'.join(notes),encoding='utf-8')
    print('Saved results to:',OUTPUT_DIR)
    for note in notes:print(note)

if __name__=='__main__':main()
