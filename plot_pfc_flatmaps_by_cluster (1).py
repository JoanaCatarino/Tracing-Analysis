"""Normalized PFC ROI flat maps by spatial cluster, pooling all mouse lines.
Each ROI = mean per-injection fraction of total bilateral PFC cells.
Run cluster_injection_centroids.py first using the current overview.
Dependencies: numpy, pandas, matplotlib, pynrrd, h5py. Original cell files are read only.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(r"L:\dmclab\Joana\Tracing\Analysis")
INPUT_DIR = BASE_DIR / "cells_prep"
CLUSTER_FILE = Path(r'L:/dmclab/Joana/Tracing/Analysis/injection_centroids/clustering/injection_cluster_candidates.csv')
OUTPUT_DIR = BASE_DIR / "plots" / "pfc_flatmaps_by_cluster"
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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
REGION_ORDER = ["MOs", "ACAd", "ACAv", "ILA", "PL", "ORBm", "ORBl", "ORBvl", "FRP", "AId", "AIv"]
SIDES = ["ipsilateral", "contralateral"]


# Point this to the folder containing the reference files used by your explorer.
FLATMAP_DIR = Path(r'L:/dmclab/Joana/Tracing/injection_explorer')
U_FILE = FLATMAP_DIR / "anno_U_filled.nrrd"
V_FILE = FLATMAP_DIR / "anno_V_filled.nrrd"
REGIONS_FILE = FLATMAP_DIR / "flatmap_PFCregions (1).h5"
ROIS_FILE = FLATMAP_DIR / "flatmap_PFCrois (1).h5"
# Continue with documented partial coverage. Unmapped selected PFC cells
# ALWAYS remain in the denominator. Tolerance units are U/V, not microns.
ALLOW_UNMAPPED_CELLS = True
ROI_TOLERANCE_UV = 2.0
SENSITIVITY_TOLERANCES_UV = [0.0, 2.0, 5.0]
COLOR_MAX_PERCENT = None  # None = one maximum across all k, clusters and sides.
from matplotlib.path import Path as MplPath
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import warnings

def select_pfc_cells(data):
    """Select the 11 template PFC regions, excluding AIp and other absent parts.

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
    if 'acronym' not in data:
        raise ValueError('Cell CSV needs acronym to select template-covered PFC regions.')
    # Match the original flatmap scripts: only the 11 template regions and layers.
    keep &= data.acronym.fillna('').astype(str).str.strip().str.match(
        r'^(?:MOs|ACAv|ACAd|PL|ILA|ORBm|ORBvl|ORBl|AId|AIv|FRP)(?:[1-6](?:[ab]|/3)?)?$')
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

def load_polygons(path):
    import h5py
    result={}
    with h5py.File(path,'r') as f:
        for key in sorted(f.keys(),key=lambda x:(0,int(x)) if x.isdigit() else (1,x)):
            v=np.asarray(f[key][()],dtype=float)
            if v.ndim!=2 or v.shape[1]!=2 or len(v)<3 or not np.isfinite(v).all():
                raise ValueError(f'Invalid polygon {key} in {path}')
            result[key]=v
    if not result:raise ValueError(f'No polygons in {path}')
    return result


def assign_rois(uv, polygons):
    assigned=np.full(len(uv),-1,dtype=int);hits=np.zeros(len(uv),dtype=int)
    # Shared edges get deterministic ownership by first ROI in sorted order.
    for j,v in enumerate(polygons.values()):
        if not np.array_equal(v[0],v[-1]):v=np.vstack([v,v[0]])
        inside=MplPath(v).contains_points(uv,radius=1e-8)
        assigned[inside & (assigned<0)]=j
        hits+=inside
    return assigned,hits


def nearest_roi_edges(uv, valid, strict_idx, polygons):
    """Exact point-to-segment distances; only otherwise-unassigned valid points.

    Original coordinates are never moved. Ties belong to the first sorted ROI.
    Invalid projections are never candidates, whatever the tolerance.
    """
    nearest=np.full(len(uv),-1,dtype=int)
    distance=np.full(len(uv),np.inf)
    pending=np.flatnonzero(valid & (strict_idx<0))
    for start in range(0,len(pending),2000):
        rows=pending[start:start+2000];points=uv[rows]
        best=np.full(len(rows),np.inf);owner=np.full(len(rows),-1,dtype=int)
        for j,verts in enumerate(polygons.values()):
            ring=np.vstack([verts,verts[0]]) if not np.array_equal(verts[0],verts[-1]) else verts
            d2=np.full(len(rows),np.inf)
            for a,b in zip(ring[:-1],ring[1:]):
                edge=b-a;length2=np.dot(edge,edge)
                t=np.clip((points-a)@edge/length2,0,1) if length2 else np.zeros(len(rows))
                delta=points-(a+t[:,None]*edge)
                d2=np.minimum(d2,np.sum(delta*delta,axis=1))
            better=d2<best
            best[better]=d2[better];owner[better]=j
        nearest[rows]=owner;distance[rows]=np.sqrt(best)
    return nearest,distance


def apply_tolerance(strict_idx, nearest, distance, tolerance):
    idx=strict_idx.copy()
    rescued=(idx<0)&(nearest>=0)&(distance<=tolerance)&(tolerance>0)
    idx[rescued]=nearest[rescued]
    return idx,rescued


def save_coverage_summaries(qc, jobs):
    meta=jobs[['injection_id','strain']+[f'cluster_k{k}' for k in CLUSTER_SOLUTIONS]].copy()
    q=qc.merge(meta,on='injection_id',validate='many_to_one')
    summaries=[]
    for k in CLUSTER_SOLUTIONS:
        q['cluster']=q[f'cluster_k{k}']
        for level,cols in [('cluster',['tolerance_uv','cluster']),
                           ('line',['tolerance_uv','strain']),
                           ('cluster_line',['tolerance_uv','cluster','strain'])]:
            for key,g in q.groupby(cols):
                row=dict(zip(cols,key if isinstance(key,tuple) else (key,)))
                total=g.pfc_cells.sum();unmapped=g.unmapped_cells.sum()
                summaries.append(dict(row,k=k,level=level,n_injections=len(g),
                    n_nonzero_pfc_injections=int(g.pfc_cells.gt(0).sum()),
                    pfc_cells=int(total),unmapped_cells=int(unmapped),
                    mean_injection_unmapped_percent=g.unmapped_percent.mean(),
                    max_injection_unmapped_percent=g.unmapped_percent.max(),
                    pooled_unmapped_percent=100*unmapped/total if total else np.nan))
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR/'mapping_coverage_by_cluster_and_line.csv',index=False)


def project_cells(data,u,v):
    coords=data[['ap_coords','dv_coords','ml_coords']].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    hemi=data.hemisphere.astype('string').str.strip().str.lower()
    if not hemi.isin(['left','right']).all():raise ValueError('Missing/invalid cell hemisphere')
    valid=np.isfinite(coords).all(axis=1)&(coords>=0).all(axis=1)
    # Same integer indexing and left-cell reflection as the supplied explorer.
    vox=np.zeros(coords.shape,dtype=np.int64);vox[valid]=coords[valid].astype(np.int64)
    left=hemi.eq('left').to_numpy()
    vox[left,2]=u.shape[2]-1-vox[left,2]
    valid&=(vox>=0).all(axis=1)&(vox<np.array(u.shape)).all(axis=1)
    uv=np.full((len(data),2),np.nan)
    ix=vox[valid]
    uv[valid,0]=u[ix[:,0],ix[:,1],ix[:,2]]
    uv[valid,1]=v[ix[:,0],ix[:,1],ix[:,2]]
    valid&=np.isfinite(uv).all(axis=1)&(uv>-1).all(axis=1)
    # Supplied filled U/V volumes use the pair (0, 0) for background.
    # Keep these cells in the denominator, but never assign them to an ROI.
    valid&=~(uv==0).all(axis=1)
    return uv,valid,hemi.to_numpy()


def mapping_diagnostics(data, job, uv, valid, hemi, roi_idx, regions, volume_shape):
    """Retain original CSV row references and distinguish projection failures."""
    d=data.copy()
    d['source_csv_row']=data.index.to_numpy()+2  # Header occupies row 1.
    d['source_csv']=str(job['source_csv'])
    d['injection_id']=job['injection_id']
    d['injection_side']=job['injection_side']
    for key,value in job.items():
        if key.startswith('cluster_k'):d[key]=value
    for col in ['region','part','acronym','hemisphere']:
        if col not in d:d[col]=''
    coords=data[['ap_coords','dv_coords','ml_coords']].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    finite=np.isfinite(coords).all(axis=1)
    nonnegative=finite&(coords>=0).all(axis=1)
    vox=np.zeros(coords.shape,dtype=np.int64)
    vox[nonnegative]=coords[nonnegative].astype(np.int64)
    left=hemi=='left';vox[left,2]=volume_shape[2]-1-vox[left,2]
    in_bounds=nonnegative&(vox>=0).all(axis=1)&(vox<np.array(volume_shape)).all(axis=1)
    reason=np.full(len(d),'mapped',dtype=object)
    reason[~finite]='missing_or_nonfinite_coordinates'
    reason[finite&~nonnegative]='negative_atlas_coordinates'
    reason[nonnegative&~in_bounds]='outside_lookup_volume'
    reason[in_bounds&~valid]='invalid_uv_lookup_value'
    reason[in_bounds&(uv==0).all(axis=1)]='missing_uv_projection_zero_pair'
    reason[valid&(roi_idx<0)]='outside_roi_polygons'
    d['failure_type']=reason
    d['u']=uv[:,0];d['v']=uv[:,1]
    for j,axis in enumerate(['ap','dv','ml']):
        d['lookup_'+axis+'_index']=np.where(nonnegative,vox[:,j],np.nan)
    d['laterality']=np.where(hemi==str(job['injection_side']).strip().lower(),'ipsilateral','contralateral')
    # This reports outline coverage; it does not change inclusion or normalization.
    inside=np.zeros(len(d),dtype=bool)
    inside[valid]=assign_rois(uv[valid],regions)[0]>=0
    d['inside_any_pfc_outline']=inside
    d['matches_explorer_pfc_prefix']=d.acronym.fillna('').astype(str).str.startswith(
        ('FRP','ORBl','ORBvl','ORBm','ACAd','ACAv','PL','ILA','AId','AIv','MOs'))
    group=['injection_id','region','part','acronym','hemisphere','laterality','failure_type',
           'inside_any_pfc_outline','matches_explorer_pfc_prefix']
    summary=d.groupby(group,dropna=False).size().reset_index(name='n_cells')
    return d.loc[d.failure_type.ne('mapped')].copy(),summary


def save_mapping_diagnostics(details, summaries, output_dir):
    detail=pd.concat(details,ignore_index=True)
    summary=pd.concat(summaries,ignore_index=True)
    detail.to_csv(output_dir/'unmapped_cells.csv',index=False)
    summary.to_csv(output_dir/'mapping_status_by_injection_region.csv',index=False)
    by_region=summary.groupby(['region','part','acronym','hemisphere','failure_type'],dropna=False).n_cells.sum().reset_index()
    by_region.to_csv(output_dir/'mapping_status_by_region.csv',index=False)
    failed=summary.loc[summary.failure_type.ne('mapped')]
    failed.to_csv(output_dir/'unmapped_summary.csv',index=False)
    print('Mapping diagnostics saved: unmapped_cells.csv, unmapped_summary.csv, mapping_status_by_region.csv')
    if len(failed):
        print(failed.groupby('failure_type').n_cells.sum().to_string())
        print('Most affected acronyms:')
        print(failed.groupby('acronym',dropna=False).n_cells.sum().sort_values(ascending=False).head(15).to_string())


def normalized_counts(roi_idx,side,n_total,n_rois):
    counts=np.zeros((2,n_rois),dtype=int)
    for i,name in enumerate(['ipsilateral','contralateral']):
        idx=roi_idx[(side==name)&(roi_idx>=0)]
        counts[i]=np.bincount(idx,minlength=n_rois)
    percent=counts*100/n_total if n_total else np.full(counts.shape,np.nan)
    return counts,percent


def main():
    import nrrd
    tolerances=sorted(set([0.0,float(ROI_TOLERANCE_UV)]+list(map(float,SENSITIVITY_TOLERANCES_UV))))
    if not all(np.isfinite(t) and t>=0 for t in tolerances):
        raise ValueError('Tolerances must be finite and nonnegative')
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    missing=[str(p) for p in [U_FILE,V_FILE,REGIONS_FILE,ROIS_FILE] if not p.is_file()]
    if missing:raise FileNotFoundError('Set FLATMAP_DIR to your explorer reference folder. Missing:\n'+'\n'.join(missing))
    clusters=pd.read_csv(CLUSTER_FILE,dtype={'animal_id':str,'cohort':str})
    required=['injection_id','injection_side','cells','striatum_percent']+[f'cluster_k{k}' for k in CLUSTER_SOLUTIONS]
    if set(required)-set(clusters):raise ValueError(f'Missing assignment columns: {set(required)-set(clusters)}')
    if clusters.empty or clusters.injection_id.duplicated().any():raise ValueError('Empty/duplicate injection assignments')
    eligible=clusters.cells.fillna('').astype(str).str.strip().str.lower().eq('y')&pd.to_numeric(clusters.striatum_percent,errors='coerce').between(80,100)
    if not eligible.all():raise ValueError('Rerun clustering against current overview: ineligible assignments found')
    for k in CLUSTER_SOLUTIONS:
        col=f'cluster_k{k}';x=pd.to_numeric(clusters[col],errors='coerce')
        if not x.isin(range(1,k+1)).all() or x.nunique()!=k:raise ValueError(f'Invalid assignments: {col}')
        clusters[col]=x.astype(int)
    cells=collect_counts(INPUT_DIR)
    jobs=clusters.merge(cells[['injection_id','source_csv','n_cells']],on='injection_id',how='left',validate='one_to_one')
    if jobs.source_csv.isna().any():raise FileNotFoundError('Missing prepared files: '+', '.join(jobs.loc[jobs.source_csv.isna(),'injection_id']))
    print('Loading the explorer U/V lookup volumes...')
    u,_=nrrd.read(str(U_FILE));v,_=nrrd.read(str(V_FILE))
    if u.shape!=v.shape or u.ndim!=3:raise ValueError('U/V must be matching 3D AP,DV,ML arrays')
    regions=load_polygons(REGIONS_FILE);rois=load_polygons(ROIS_FILE)
    names=list(rois);all_maps={t:[] for t in tolerances};qc=[];rows=[];diagnostic_rows=[];diagnostic_summaries=[];reassigned_rows=[]
    for job in jobs.to_dict('records'):
        raw=pd.read_csv(job['source_csv'],dtype=str)
        excluded_aip=int(raw.acronym.fillna('').str.startswith('AIp').sum())
        data=select_pfc_cells(raw)
        n=len(data);side=str(job['injection_side']).strip().lower()
        if side not in ['left','right']:raise ValueError(f'Invalid injection side: {job["injection_id"]}')
        uv,valid,hemi=project_cells(data,u,v)
        idx=np.full(n,-1,dtype=int);hits=np.zeros(n,dtype=int)
        idx[valid],hits[valid]=assign_rois(uv[valid],rois)
        strict_idx=idx.copy()
        nearest,distance=nearest_roi_edges(uv,valid,strict_idx,rois)
        idx,rescued=apply_tolerance(strict_idx,nearest,distance,ROI_TOLERANCE_UV)
        detail,summary=mapping_diagnostics(data,job,uv,valid,hemi,idx,regions,u.shape)
        diagnostic_rows.append(detail);diagnostic_summaries.append(summary)
        laterality=np.where(hemi==side,'ipsilateral','contralateral')
        for tolerance in tolerances:
            current,rescued=apply_tolerance(strict_idx,nearest,distance,tolerance)
            counts,pct=normalized_counts(current,laterality,n,len(names));all_maps[tolerance].append(pct)
            unmapped=int((current<0).sum())
            qc.append(dict(injection_id=job['injection_id'],tolerance_uv=tolerance,pfc_cells=n,
                excluded_aip_cells=excluded_aip,invalid_uv=int((~valid).sum()),
                outside_rois=int((valid&(current<0)).sum()),multiple_roi_hits=int((hits>1).sum()),
                reassigned_cells=int(rescued.sum()),unmapped_cells=unmapped,
                unmapped_percent=100*unmapped/n if n else np.nan,
                mapped_cells=int((current>=0).sum()),mapped_percent=100*(current>=0).sum()/n if n else np.nan))
            rr=data.loc[rescued].copy()
            rr['source_csv_row']=data.index.to_numpy()[rescued]+2
            rr['source_csv']=job['source_csv'];rr['injection_id']=job['injection_id']
            rr['tolerance_uv']=tolerance;rr['u']=uv[rescued,0];rr['v']=uv[rescued,1]
            rr['distance_uv']=distance[rescued];rr['assigned_roi']=[names[j] for j in current[rescued]]
            reassigned_rows.append(rr)
            for h,label in enumerate(SIDES):
                for j,roi in enumerate(names):rows.append(dict(injection_id=job['injection_id'],tolerance_uv=tolerance,laterality=label,roi=roi,n_cells=counts[h,j],pfc_denominator=n,percent=pct[h,j]))
    all_qc=pd.DataFrame(qc)
    all_qc.to_csv(OUTPUT_DIR/'tolerance_sensitivity_by_injection.csv',index=False)
    all_qc.loc[all_qc.tolerance_uv.eq(ROI_TOLERANCE_UV)].to_csv(OUTPUT_DIR/'mapping_qc.csv',index=False)
    all_rows=pd.DataFrame(rows)
    all_rows.to_csv(OUTPUT_DIR/'tolerance_sensitivity_injection_roi_percentages.csv',index=False)
    all_rows.loc[all_rows.tolerance_uv.eq(ROI_TOLERANCE_UV)].to_csv(OUTPUT_DIR/'injection_roi_percentages.csv',index=False)
    pd.concat(reassigned_rows,ignore_index=True).to_csv(OUTPUT_DIR/'boundary_reassignments.csv',index=False)
    save_mapping_diagnostics(diagnostic_rows,diagnostic_summaries,OUTPUT_DIR)
    save_coverage_summaries(all_qc,jobs)
    if not ALLOW_UNMAPPED_CELLS and all_qc.loc[all_qc.tolerance_uv.eq(ROI_TOLERANCE_UV),'unmapped_cells'].sum():
        raise ValueError('Unmapped cells remain. Inspect mapping_qc.csv.')
    if all_qc.unmapped_cells.sum():warnings.warn('Unmapped cells remain in denominators; inspect coverage summaries.')
    # A single scale includes all tolerances, k values, clusters and hemispheres.
    maxima=[]
    for t,values in all_maps.items():
        values=np.stack(values)
        for k in CLUSTER_SOLUTIONS:
            for c in range(1,k+1):
                mask=jobs[f'cluster_k{k}'].eq(c).to_numpy()&np.isfinite(values).all(axis=(1,2))
                if mask.any():maxima.append(values[mask].mean(axis=0).max())
    shared_max=COLOR_MAX_PERCENT if COLOR_MAX_PERCENT is not None else max(maxima,default=0)
    for tolerance in tolerances:
        maps=np.stack(all_maps[tolerance])
        qc=all_qc.loc[all_qc.tolerance_uv.eq(tolerance)].reset_index(drop=True)
        unmapped=int(qc.unmapped_cells.sum())
        folder=OUTPUT_DIR if tolerance==ROI_TOLERANCE_UV else OUTPUT_DIR/f'tolerance_{tolerance:g}'
        folder.mkdir(parents=True,exist_ok=True)
        render_maps(maps,qc,jobs,names,rois,regions,shared_max,tolerance,folder,unmapped)
    print(f'DONE! {len(jobs)} injections; maps saved in {OUTPUT_DIR}')


def render_maps(maps,qc,jobs,names,rois,regions,shared_max,tolerance,folder,unmapped):
    solutions={};summary=[]
    for k in CLUSTER_SOLUTIONS:
        means=[];ns=[]
        for c in range(1,k+1):
            mask=jobs[f'cluster_k{k}'].eq(c).to_numpy()&(qc.pfc_cells.to_numpy()>0)
            ns.append(int(mask.sum()))
            if not mask.any():raise ValueError(f'k={k}, cluster {c}: no nonzero-PFC injections for normalization')
            mean=maps[mask].mean(axis=0);means.append(mean)
            sem=maps[mask].std(axis=0,ddof=1)/np.sqrt(mask.sum()) if mask.sum()>1 else np.full_like(mean,np.nan)
            for h,label in enumerate(['ipsilateral','contralateral']):
                for j,roi in enumerate(names):summary.append(dict(k=k,cluster=c,laterality=label,roi=roi,n_injections=int(mask.sum()),mean_percent=mean[h,j],sem_percent=sem[h,j]))
        solutions[k]=(np.stack(means),ns)
    pd.DataFrame(summary).to_csv(folder/'cluster_roi_percentages.csv',index=False)
    vmax=shared_max
    if not np.isfinite(vmax) or vmax<=0:raise ValueError('Invalid color maximum or no mapped cells')
    norm=Normalize(0,vmax);cmap=plt.get_cmap('magma_r')
    vertices=np.vstack(list(rois.values())+list(regions.values()))
    lo=vertices.min(axis=0);hi=vertices.max(axis=0);pad=(hi-lo)*.04
    for k,(means,ns) in solutions.items():
        fig,axes=plt.subplots(k,2,figsize=(12,3.2*k),squeeze=False,layout='constrained')
        for c in range(k):
            for h,label in enumerate(['Ipsilateral','Contralateral']):
                ax=axes[c,h]
                coll=PolyCollection(list(rois.values()),array=means[c,h],cmap=cmap,norm=norm,edgecolors='#bbbbbb',linewidths=.35)
                ax.add_collection(coll)
                for name,verts in regions.items():
                    closed=np.vstack([verts,verts[0]])
                    ax.plot(closed[:,0],closed[:,1],color='#333333',lw=.6)
                    if not str(name).isdigit():
                        center=verts.mean(axis=0);ax.text(*center,str(name),fontsize=6,ha='center')
                ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(hi[1]+pad[1],lo[1]-pad[1])
                if h==1:ax.invert_xaxis()
                coverage=qc.loc[jobs[f'cluster_k{k}'].eq(c+1).to_numpy(),'unmapped_percent'].mean()
                ax.set_aspect('equal');ax.axis('off');ax.set_title(f'Cluster {c+1} · {label} · n={ns[c]} injections\nUnmapped (both sides): {coverage:.1f}% mean per injection')
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axes.ravel().tolist(),shrink=.6,label='Mean % of total PFC cells per ROI')
        fig.suptitle(f'{k} spatial clusters — normalized PFC flat maps\nAIp excluded; both hemispheres in denominator; tolerance {tolerance:g} U/V units',fontsize=13)
        for ext in ['png','pdf','svg']:fig.savefig(folder/f'normalized_pfc_flatmap_k{k}.{ext}',dpi=300)
        plt.close(fig)
    (folder/'normalization_notes.txt').write_text(
        f'AIp excluded from numerator and denominator. Template regions: {REGION_ORDER}.\n'
        f'Nearest-ROI boundary tolerance: {tolerance:g} U/V units, not microns. Only valid, strictly unassigned cells are reassigned. Ties use first sorted ROI.\n'
        f'100 * ROI cell count / total PFC cells of that injection, both hemispheres. Average percentages across injections, all mouse lines pooled.\n'
        f'Zero-PFC injections excluded: {int(qc.pfc_cells.eq(0).sum())}. Unmapped cells: {unmapped}; never removed from denominator.\n'
        f'One color scale for all solutions: 0 to {vmax:g} percent per ROI. This is relative distribution, not cells per physical area or absolute projection strength.\n'
        'Projection uses supplied explorer convention: AP,DV,ML integer voxel indices; left cells use n_ML-1-ML. Original injection side defines ipsi/contra.\n'
        'ROIs are taken directly from supplied H5; their equal physical surface area is not independently verified. No interpolation across empty bins.\n'
        'Normalization does not correct incomplete tissue coverage, injection volume, viral efficiency, or repeated-animal dependence.\n',encoding='utf-8')

if __name__=='__main__':main()
