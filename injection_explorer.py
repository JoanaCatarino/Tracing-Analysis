#!/usr/bin/env python3
"""
injection_explorer.py  —  Joana/Tracing edition

Adapted for the folder structure at /Volumes/Dmclab/Joana/Tracing:

  <tracing-root>/
    Fezf2/Cohort N/<animal_id>/results/injection_site/green/<id>_injection_site.csv
    Fezf2/Cohort N/<animal_id>/results/cells/cy3/<id>_cells.csv
    Tlx3/ ...
    Fmr1-Fezf2/ ...

Static resources (workbook, meshes, NRRDs, flatmap SVG) live in the same
folder as this script (injection_explorer/).

Usage (run from the injection_explorer/ folder):
  python3 injection_explorer.py \\
      --tracing-root /Volumes/Dmclab/Joana/Tracing \\
      --workbook     Tracing_PT_IT_neurons_inUSE1.xlsx \\
      --meshdir      allen_mesh \\
      --flatmap      flatmap_PFC_template.svg \\
      --u-nrrd       anno_U_filled.nrrd \\
      --v-nrrd       anno_V_filled.nrrd \\
      --out          injection_explorer.html
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import meshio
import numpy as np
import openpyxl
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
import nrrd

from pathlib import Path


# ---------------------------------------------------------------------------
# Allen CCF striatum classification  (same set as fill_injection_fractions)
# ---------------------------------------------------------------------------
_STRIATUM_EXACT = {"CP", "ACB"}

def is_striatal(acronym: str) -> bool:
    return str(acronym).strip() in _STRIATUM_EXACT

# ---------------------------------------------------------------------------
# PFC prefixes for cell flatmap projection
# ---------------------------------------------------------------------------
PFC_PREFIXES = [
    "FRP", "ORBl", "ORBvl", "ORBm", "ACAd", "ACAv",
    "PL", "ILA", "AId", "AIv", "MOs",
]

def is_pfc(acronym: str) -> bool:
    ac = str(acronym).strip()
    return any(ac.startswith(p) for p in PFC_PREFIXES)

# ---------------------------------------------------------------------------
# Atlas / render constants  (same as injection_cortex_3d.py)
# ---------------------------------------------------------------------------
ML_MIDLINE  = 5700    # µm  (Allen CCF ML midline)
ML_MID_VOX  = 570     # voxel units (10 µm each)
GRID_STEP   = 50      # µm per marching-cubes grid cell
# Anisotropic Gaussian: stronger along AP axis to bridge inter-section gaps
# (sections every 100 µm, so 2-grid-cell gaps become connected at AP sigma=2)
SIGMA       = [2.0, 1.0, 1.0]   # [AP, DV, ML] in grid units
MC_LEVEL    = 0.2
INJ_OPACITY = 0.65

# Allen-style light blue for striatal meshes — both same color, low opacity
MESH_DEFS = {
    "997": ("Brain surface",   "lightgray", 0.05),
    "672": ("Caudoputamen",    "#ADD8E6",   0.08),
    "56":  ("Nuc. Accumbens",  "#ADD8E6",   0.08),
}

# ---------------------------------------------------------------------------
# Colour palette  (20 perceptually distinct colours, cycling for >20 inj.)
# ---------------------------------------------------------------------------
PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    "#1f77b4", "#aec7e8", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b",
]

# ---------------------------------------------------------------------------
# Master workbook sheets
# ---------------------------------------------------------------------------
SHEET_STRAIN = {
    "Analysis_Tlx3":       "tlx3",
    "Analysis_Fezf2":      "fezf2",
    "Analysis_Fmr1_Fezf2": "fmr1-fezf2",
}

# Maps top-level folder name in the Joana Tracing tree → strain key used in
# SHEET_STRAIN / metadata lookups.
JOANA_FOLDER_TO_STRAIN = {
    "Fezf2":     "fezf2",
    "Tlx3":      "tlx3",
    "Fmr1-Fezf2": "fmr1-fezf2",
}

# Column indices (1-based) — same for all Analysis sheets after fill_injection_fractions ran
COL_MOUSE   = 2   # B  animal ID
COL_SLICE   = 7   # G  Slice AP (Start / Middle / End)
COL_HEMI    = 8   # H  Hemisph
COL_REGION  = 9   # I  Region (DMS, VMS, DLS, VLS, dorsal, medial …)
COL_STR     = 39  # AM %STR
COL_CORTEX  = 40  # AN %Cortex


# ===========================================================================
# Helpers
# ===========================================================================

def hex_to_rgba(h: str, alpha: float) -> str:
    h = h.lstrip("#")
    r, g, b = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    return f"rgba({r},{g},{b},{alpha})"


def voxels_to_mesh_and_filled(coords_um: np.ndarray):
    """
    Builds a Gaussian-smoothed volume from a point cloud, then:
      - runs marching cubes to get the surface mesh
      - extracts all grid cells whose smoothed value exceeds MC_LEVEL
        (i.e. the *filled interior* of the injection volume)

    Returns (verts, faces, filled_um) where filled_um is an (N,3) array of
    µm coordinates at GRID_STEP resolution covering the interior.
    Any of the three can be None if the input is too sparse.
    """
    if len(coords_um) < 20:
        return None, None, None
    lo = coords_um.min(axis=0) - GRID_STEP * 3
    hi = coords_um.max(axis=0) + GRID_STEP * 3
    shape = np.ceil((hi - lo) / GRID_STEP).astype(int) + 1
    if np.any(shape < 4):
        return None, None, None
    vol = np.zeros(shape, dtype=np.float32)
    idx = np.round((coords_um - lo) / GRID_STEP).astype(int)
    idx = np.clip(idx, 0, np.array(shape) - 1)
    vol[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    vol = gaussian_filter(vol, sigma=SIGMA)

    # ── filled interior voxels ────────────────────────────────────────────
    # Every grid cell above the isosurface threshold is "inside" the mesh.
    # This naturally interpolates across the gaps between imaged sections.
    inside_idx = np.argwhere(vol > MC_LEVEL)          # (N, 3) local grid coords
    filled_um  = inside_idx * GRID_STEP + lo           # convert to µm

    # ── surface mesh ──────────────────────────────────────────────────────
    try:
        verts, faces, _, _ = marching_cubes(vol, level=MC_LEVEL)
    except (RuntimeError, ValueError):
        return None, None, filled_um
    return verts * GRID_STEP + lo, faces, filled_um


# ---------------------------------------------------------------------------
# Overlap voxel packing  (50 µm global grid)
# ---------------------------------------------------------------------------
# Atlas at 10 µm spans ~1320 × 800 × 1140 voxels = 13200 × 8000 × 11400 µm.
# At 50 µm step: AP 0-263, DV 0-159, ML 0-227  →  229 × 161 strides.
OVERLAP_STEP = 50   # µm
_ML_N  = 229        # ML stride
_DV_N  = 161        # DV stride
_DV_ML = _ML_N * _DV_N   # = 36 869  (AP stride)

def pack_voxels(filled_um: np.ndarray) -> list[int]:
    """Convert µm coords to packed 50 µm grid integers for overlap computation."""
    ap = np.round(filled_um[:, 0] / OVERLAP_STEP).astype(int)
    dv = np.round(filled_um[:, 1] / OVERLAP_STEP).astype(int)
    ml = np.round(filled_um[:, 2] / OVERLAP_STEP).astype(int)
    ap = np.clip(ap, 0, 263)
    dv = np.clip(dv, 0, 160)
    ml = np.clip(ml, 0, 228)
    packed = ap * _DV_ML + dv * _ML_N + ml
    return list(map(int, np.unique(packed)))


def load_atlas_trace(obj_path: Path, name: str, color: str, opacity: float) -> dict:
    mesh = meshio.read(str(obj_path))
    v = mesh.points
    f = mesh.cells[0].data
    return dict(
        type="mesh3d",
        x=v[:, 0].tolist(), y=v[:, 1].tolist(), z=v[:, 2].tolist(),
        i=f[:, 0].tolist(), j=f[:, 1].tolist(), k=f[:, 2].tolist(),
        opacity=opacity, color=color, name=name,
        showlegend=False, hoverinfo="skip",
    )


def index_nrrd(u_data, v_data, ap, dv, ml):
    shape = u_data.shape
    in_b = (ap >= 0) & (ap < shape[0]) & (dv >= 0) & (dv < shape[1]) & (ml >= 0) & (ml < shape[2])
    u = np.full(len(ap), np.nan)
    v = np.full(len(ap), np.nan)
    u[in_b] = u_data[ap[in_b], dv[in_b], ml[in_b]]
    v[in_b] = v_data[ap[in_b], dv[in_b], ml[in_b]]
    return u, v


# ---------------------------------------------------------------------------
# Centroid-based striatal subregion classification
# ---------------------------------------------------------------------------
# Uses ml_mm and dv_mm from the injection CSV (mm from bregma / midline).
# Thresholds are typical for the mouse DMS/DLS/VMS/VLS distinction:
#   Lateral threshold:  |ml_mm| >= 1.5 mm → lateral (DLS/VLS)
#   Ventral threshold:  dv_mm  <= -4.2 mm → ventral (VMS/VLS)
#   ACB-dominant:       >50% of striatal voxels are ACB → VS

def compute_centroid_subregion(df_str: pd.DataFrame) -> tuple[dict, str]:
    """
    Given the striatal-voxel dataframe, compute centroid in mm and return
    ({ap_mm, dv_mm, ml_mm}, subregion_label).
    """
    if df_str.empty:
        return {"ap": None, "dv": None, "ml": None}, "?"

    # Use abs(ml_mm) so both hemispheres map to positive lateral distance
    ml_abs = df_str["ml_mm"].abs()
    ap_c   = float(df_str["ap_mm"].mean())
    dv_c   = float(df_str["dv_mm"].mean())
    ml_c   = float(ml_abs.mean())

    # Fraction of voxels that are ACB (nucleus accumbens)
    acb_frac = (df_str["acronym"] == "ACB").mean()

    if acb_frac > 0.5:
        label = "VS"
    elif ml_c >= 1.5:
        label = "DLS" if dv_c > -4.2 else "VLS"
    else:
        label = "DMS" if dv_c > -4.2 else "VMS"

    return {"ap": round(ap_c, 2), "dv": round(dv_c, 2), "ml": round(ml_c, 2)}, label


# ===========================================================================
# 1. Read master workbook → injection metadata
# ===========================================================================

def read_metadata(wb_path: Path) -> dict:
    """
    Returns {(strain, animal_id_str, hemi): {ap_pos, region, pct_str}} for
    every data row in the three Analysis sheets.
    """
    wb = openpyxl.load_workbook(str(wb_path), read_only=True, data_only=True)
    meta = {}
    for sheet_name, strain in SHEET_STRAIN.items():
        if sheet_name not in wb.sheetnames:
            continue
        ws = wb[sheet_name]
        cur_animal = None
        cur_ap     = None
        for ri in range(4, ws.max_row + 1):
            mouse  = ws.cell(ri, COL_MOUSE).value
            slc    = ws.cell(ri, COL_SLICE).value
            hemi    = ws.cell(ri, COL_HEMI).value
            region  = ws.cell(ri, COL_REGION).value
            pct     = ws.cell(ri, COL_STR).value
            pct_ctx = ws.cell(ri, COL_CORTEX).value

            if mouse is None and hemi is None:
                continue
            if mouse is not None:
                cur_animal = str(int(mouse))
                cur_ap     = str(slc).strip().lower() if slc else None
            if cur_animal is None or cur_ap is None:
                continue
            h = str(hemi).strip().lower() if hemi else None
            if h not in ("left", "right"):
                continue

            key = (strain, cur_animal, h)
            meta[key] = {
                "ap_pos":      cur_ap,
                "region":      str(region).strip() if region else "?",
                "pct_str":     float(pct)     if pct     is not None else None,
                "pct_cortex":  float(pct_ctx) if pct_ctx is not None else None,
                "genotype":    strain,
            }
    return meta


# ===========================================================================
# 2. Build per-injection data
# ===========================================================================

def build_injection(
    animal_id: str, hemi: str, meta: dict,
    inj_csv: Path,
    u_data, v_data,
    color_idx: int,
) -> dict | None:
    """
    Build the full data dict for one injection (animal × hemisphere).
    inj_csv  — path to <animal>/results/injection_site/green/<id>_injection_site.csv
    Cell CSV is resolved relative to inj_csv:
      inj_csv.parent.parent.parent / "cells" / "cy3" / "<id>_cells.csv"
    Returns None if the injection CSV is missing.
    """
    strain     = meta["genotype"]
    ap_pos     = meta["ap_pos"]
    region     = meta["region"]
    pct_str    = meta["pct_str"]
    pct_cortex = meta.get("pct_cortex", None)
    mirror     = (hemi == "right")   # all injections shown in left-hemisphere coords

    if not inj_csv.exists():
        return None

    print(f"  [{animal_id} {hemi}] loading injection CSV …", end=" ", flush=True)
    df = pd.read_csv(inj_csv)

    # ── hemisphere selection ────────────────────────────────────────────────
    ml_um = df["ml_coords"].values * 10
    if hemi == "left":
        mask = ml_um >= ML_MIDLINE
    else:
        mask = ml_um < ML_MIDLINE
    df = df[mask].copy()

    if df.empty:
        print("no voxels — skipping")
        return None

    # ── striatal subset ──────────────────────────────────────────────────────
    str_mask = df["acronym"].apply(is_striatal).values
    df_str   = df[str_mask]
    print(f"{len(df)} hem voxels, {len(df_str)} striatal")

    # Full voxel-level axis extents — used by the JS slider filter (centroid alone
    # is too coarse; slider should match any voxel of the reconstructed volume)
    if not df_str.empty:
        ap_range = [round(float(df_str["ap_mm"].min()), 2),
                    round(float(df_str["ap_mm"].max()), 2)]
        dv_range = [round(float(df_str["dv_mm"].min()), 2),
                    round(float(df_str["dv_mm"].max()), 2)]
        _ml_abs  = df_str["ml_mm"].abs()
        ml_range = [round(float(_ml_abs.min()), 2),
                    round(float(_ml_abs.max()), 2)]
    else:
        ap_range = [None, None]
        dv_range = [None, None]
        ml_range = [None, None]

    ml_raw = df_str["ml_coords"].values * 10
    if mirror:
        ml_plot = 2.0 * ML_MIDLINE - ml_raw
    else:
        ml_plot = ml_raw.copy()

    coords_um = np.column_stack([
        df_str["ap_coords"].values * 10,
        df_str["dv_coords"].values * 10,
        ml_plot,
    ])

    # ── centroid-based subregion ──────────────────────────────────────────────
    centroid, subregion = compute_centroid_subregion(df_str)

    # ── marching-cubes mesh + filled interior voxels ─────────────────────────
    verts, faces, filled_um = voxels_to_mesh_and_filled(coords_um)

    if verts is None:
        print(f"  [{animal_id} {hemi}] mesh failed — using empty mesh")
        vx, vy, vz, fi, fj, fk = [], [], [], [], [], []
    else:
        vx = verts[:, 0].tolist()
        vy = verts[:, 1].tolist()
        vz = verts[:, 2].tolist()
        fi = faces[:, 0].tolist()
        fj = faces[:, 1].tolist()
        fk = faces[:, 2].tolist()

    # ── overlap voxels: filled mesh interior at 50 µm grid ───────────────────
    # filled_um comes from the Gaussian-smoothed volume so it bridges the
    # gaps between imaged sections — much more robust than raw slice voxels.
    if filled_um is not None and len(filled_um) > 0:
        packed = pack_voxels(filled_um)
    else:
        packed = []

    # ── PFC cell flatmap projection ──────────────────────────────────────────
    # cells_ipsi:  cells ipsilateral to injection → shown on left-hem SVG (unmirrored)
    # cells_contra: cells contralateral to injection → shown on mirrored (right-hem) SVG
    #
    # NRRD lookup convention (from flatmap_per_animal.py):
    #   • Right-side voxels (ml_mm > 0, low ml_vx in CCF):  use ml_vx directly
    #   • Left-side voxels  (ml_mm < 0, high ml_vx in CCF): flip  n_ml - 1 - ml_vx
    # Both yield UV coordinates in left-hemisphere flatmap space.
    cells_ipsi  = {"u": [], "v": []}
    cells_contra = {"u": [], "v": []}

    # Cell CSV lives at: <animal_root>/results/cells/cy3/<id>_cells.csv
    # inj_csv is at:     <animal_root>/results/injection_site/green/<id>_injection_site.csv
    # so go up 3 levels (green → injection_site → results) then down to cells/cy3
    animal_results = inj_csv.parent.parent.parent   # <animal_root>/results/
    cell_csv = animal_results / "cells" / "cy3" / f"{animal_id}_cells.csv"
    if not cell_csv.exists():
        cell_csv = None

    if cell_csv and cell_csv.exists():
        try:
            dfc = pd.read_csv(cell_csv)
            for col in ("ap_coords", "ml_coords", "dv_coords"):
                dfc[col] = pd.to_numeric(dfc[col], errors="coerce")
            dfc = dfc.dropna(subset=["ap_coords", "ml_coords", "dv_coords"])
            dfc = dfc[dfc["acronym"].apply(is_pfc)].copy()

            if not dfc.empty:
                ml_mm_vals = pd.to_numeric(dfc["ml_mm"], errors="coerce").fillna(0).values
                ap_vx = dfc["ap_coords"].values.astype(int)
                dv_vx = dfc["dv_coords"].values.astype(int)
                ml_vx = dfc["ml_coords"].values.astype(int)
                n_ml  = u_data.shape[2]

                def _project(mask, flip_ml):
                    if not mask.any():
                        return {"u": [], "v": []}
                    ml_lk = (n_ml - 1 - ml_vx[mask]) if flip_ml else ml_vx[mask]
                    u_, v_ = index_nrrd(u_data, v_data,
                                        ap_vx[mask], dv_vx[mask], ml_lk)
                    ok = ~np.isnan(u_) & ~np.isnan(v_) & (u_ > -1) & (v_ > -1)
                    return {"u": u_[ok].tolist(), "v": v_[ok].tolist()}

                if hemi == "right":
                    # Ipsi = right cells (ml_mm > 0) → use ml_vx directly
                    cells_ipsi  = _project(ml_mm_vals > 0, flip_ml=False)
                    # Contra = left cells  (ml_mm < 0) → flip ml_vx
                    cells_contra = _project(ml_mm_vals < 0, flip_ml=True)
                else:
                    # Ipsi = left cells (ml_mm < 0) → flip ml_vx
                    cells_ipsi  = _project(ml_mm_vals < 0, flip_ml=True)
                    # Contra = right cells (ml_mm > 0) → use ml_vx directly
                    cells_contra = _project(ml_mm_vals > 0, flip_ml=False)

            print(f"     cells ipsi={len(cells_ipsi['u'])} contra={len(cells_contra['u'])}")
        except Exception as e:
            print(f"     [WARN] cell CSV error: {e}")

    color = PALETTE[color_idx % len(PALETTE)]

    return {
        "id":         f"{animal_id}_{hemi}",
        "animal":     animal_id,
        "hemi":       hemi,
        "ap_pos":     ap_pos,
        "region":     region,
        "subregion":  subregion,   # centroid-derived: DMS/DLS/VMS/VLS/VS/?
        "centroid":   centroid,    # {ap, dv, ml} in mm
        "geno":       strain,
        "pct_str":    pct_str,
        "pct_cortex": pct_cortex,
        "color":      color,
        # Plotly mesh
        "mx": vx, "my": vy, "mz": vz,
        "mi": fi, "mj": fj, "mk": fk,
        # Overlap voxels (50 µm grid, packed integers)
        "vox": packed,
        # Axis extents of the full striatal voxel cloud (mm from bregma/midline)
        # Used by the slider filter; absolute ML so both hemispheres are comparable
        "ap_range": ap_range,
        "dv_range": dv_range,
        "ml_range": ml_range,
        # Flatmap cell projections
        # cells_ipsi:  ipsilateral cells → draw on left (unmirrored) SVG
        # cells_contra: contralateral cells → draw on right (mirrored) SVG
        "cells_ipsi":  cells_ipsi,
        "cells_contra": cells_contra,
    }


# ===========================================================================
# 3. Pre-compute all pairwise overlaps
# ===========================================================================

def compute_all_overlaps(injections: list[dict]) -> list[dict]:
    """
    Compute Jaccard overlap for every pair of injections using the
    filled mesh interior voxels (50 µm grid).  Returns list sorted
    by descending overlap %, zero-overlap pairs excluded.
    """
    print("Computing pairwise overlaps …")
    sets = {inj["id"]: set(inj["vox"]) for inj in injections}
    ids  = [inj["id"] for inj in injections]
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_set = sets[ids[i]]
            b_set = sets[ids[j]]
            if not a_set or not b_set:
                continue
            isect = len(a_set & b_set)
            if isect == 0:
                continue
            union = len(a_set) + len(b_set) - isect
            pct = round(isect / union * 100, 1) if union > 0 else 0.0
            pairs.append({"a": ids[i], "b": ids[j], "pct": pct,
                          "na": len(a_set), "nb": len(b_set), "isect": isect})
    pairs.sort(key=lambda x: -x["pct"])
    print(f"  {len(pairs)} pairs with overlap (of {len(ids)*(len(ids)-1)//2} total)")
    if pairs:
        print(f"  Top 5: {[p['pct'] for p in pairs[:5]]}")
    return pairs


# ===========================================================================
# 4. HTML generation
# ===========================================================================

def _svg_tight_bbox(svg_text: str, padding: float = 20.0):
    """Return (x0, y0, w, h) tight bbox of polygon/path content in the SVG."""
    all_x, all_y = [], []
    for pts in re.findall(r'points\s*=\s*["\']([^"\']+)["\']', svg_text):
        nums = re.findall(r'[-\d.]+', pts)
        for i in range(0, len(nums) - 1, 2):
            all_x.append(float(nums[i])); all_y.append(float(nums[i + 1]))
    for d in re.findall(r'\bd\s*=\s*["\']([^"\']+)["\']', svg_text):
        for tx, ty in re.findall(r'[MLCSQTmlcsqt]\s*([-\d.]+)[,\s]+([-\d.]+)', d):
            all_x.append(float(tx)); all_y.append(float(ty))
    if not all_x:
        return 0.0, 0.0, 1500.0, 1500.0
    x0 = min(all_x) - padding; y0 = min(all_y) - padding
    return x0, y0, max(all_x) + padding - x0, max(all_y) + padding - y0


def crop_svg_to_content(svg_path: Path, padding: float = 20.0):
    """Crop the SVG viewBox to its polygon/path content bounds."""
    txt = svg_path.read_text(encoding="utf-8")
    x0, y0, w, h = _svg_tight_bbox(txt, padding)
    new_vb = f"{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}"
    txt_cropped = re.sub(
        r'viewBox\s*=\s*["\'][^"\']+["\']',
        f'viewBox="{new_vb}"', txt, flags=re.I,
    )
    return txt_cropped.encode("utf-8"), x0, y0, w, h


def create_right_hem_svg(svg_path: Path, orig_width: float = 1500.0,
                         padding: float = 20.0):
    """
    Build a right-hemisphere mirror of the SVG:
      • Polygon x-coordinates flipped:  x → orig_width - x
      • Text matrix translation flipped: e → orig_width - e
      • Text matrix rotation negated: (a,b,c,d) → (a,-b,-c,d)
        so labels sit at the anatomically correct mirrored position
        AND individual letter glyphs remain readable (not backwards).
    Returns (svg_bytes, x0, y0, w, h) with the tight-cropped viewBox.
    """
    txt = svg_path.read_text(encoding="utf-8")

    # ── mirror polygon point coordinates ────────────────────────────────────
    def mirror_polygon(m):
        nums = re.findall(r'[-\d.]+', m.group(1))
        pts = []
        for i in range(0, len(nums) - 1, 2):
            pts.append(f"{orig_width - float(nums[i]):.2f},{float(nums[i+1]):.2f}")
        return f'points="{" ".join(pts)}"'

    result = re.sub(r'points\s*=\s*"([^"]+)"', mirror_polygon, txt)

    # ── mirror text matrix transforms ────────────────────────────────────────
    # Only the translation (e) is mirrored: e → orig_width - e.
    # Rotation components (a,b,c,d) are left unchanged so glyphs stay readable.
    # text-anchor="end" is added so the text extends *leftward* from the mirrored
    # anchor, landing inside the mirrored polygon (instead of past its right edge).
    def patch_text_element(m):
        elem = m.group(0)
        # Mirror only the translation component e
        def mirror_matrix(mm):
            parts = mm.group(1).split()
            if len(parts) != 6:
                return mm.group(0)
            a, b, c, d, e, f = (float(p) for p in parts)
            return (f'matrix({a:.6g} {b:.6g} {c:.6g} {d:.6g} '
                    f'{orig_width - e:.4f} {f:.4f})')
        elem = re.sub(r'matrix\(([^)]+)\)', mirror_matrix, elem)
        # Anchor text at its end so it flows into the mirrored polygon
        elem = elem.replace('<text ', '<text text-anchor="end" ')
        return elem

    result = re.sub(r'<text\b.*?</text>', patch_text_element,
                    result, flags=re.DOTALL)

    # ── crop viewBox to mirrored content ────────────────────────────────────
    x0, y0, w, h = _svg_tight_bbox(result, padding)
    new_vb = f"{x0:.2f} {y0:.2f} {w:.2f} {h:.2f}"
    result = re.sub(r'viewBox\s*=\s*"[^"]+"', f'viewBox="{new_vb}"',
                    result, flags=re.I)

    return result.encode("utf-8"), x0, y0, w, h


def build_html(
    injections: list[dict],
    atlas_traces: list[dict],
    overlap_pairs: list[dict],
    flatmap_svg_path: Path,
    plotly_ver: str,
) -> str:
    # Crop SVG to content bounds — left hemisphere (ipsi flatmap)
    svg_bytes, svg_x0, svg_y0, svg_w, svg_h = crop_svg_to_content(flatmap_svg_path)
    flatmap_data_url = f"data:image/svg+xml;base64,{base64.b64encode(svg_bytes).decode('ascii')}"
    print(f"  SVG (L) viewBox: x0={svg_x0:.1f} y0={svg_y0:.1f} w={svg_w:.1f} h={svg_h:.1f}")

    # Right-hemisphere mirror SVG — polygons flipped, text readable at mirrored positions
    svg_bytes_r, svg_x0_r, svg_y0_r, svg_w_r, svg_h_r = create_right_hem_svg(flatmap_svg_path)
    flatmap_data_url_r = f"data:image/svg+xml;base64,{base64.b64encode(svg_bytes_r).decode('ascii')}"
    print(f"  SVG (R) viewBox: x0={svg_x0_r:.1f} y0={svg_y0_r:.1f} w={svg_w_r:.1f} h={svg_h_r:.1f}")

    payload = json.dumps({
        "injections":    injections,
        "atlas_traces":  atlas_traces,
        "pairs":         overlap_pairs,
        "flatmap_url":   flatmap_data_url,
        "flatmap_url_r": flatmap_data_url_r,
        "svg_x0":   svg_x0,   "svg_y0":   svg_y0,   "svg_w":   svg_w,   "svg_h":   svg_h,
        "svg_x0_r": svg_x0_r, "svg_y0_r": svg_y0_r, "svg_w_r": svg_w_r, "svg_h_r": svg_h_r,
        "svg_orig_w": 1500.0,
    }, separators=(",", ":"))

    plotly_cdn = f"https://cdn.plot.ly/plotly-{plotly_ver}.min.js"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Injection Explorer</title>
<script src="{plotly_cdn}"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:system-ui,sans-serif;display:flex;flex-direction:column;height:100vh;overflow:hidden;background:#fafafa;}}
#top-bar{{
  min-height:48px;height:auto;display:flex;align-items:center;padding:4px 12px;gap:5px;
  background:#fff;border-bottom:1px solid #ddd;flex-shrink:0;flex-wrap:wrap;
}}
#top-bar strong{{margin-right:6px;font-size:13px;white-space:nowrap;}}
.sep{{width:1px;height:24px;background:#ddd;margin:0 4px;flex-shrink:0;}}
#main{{display:flex;flex:1;overflow:hidden;}}
/* LEFT */
#left{{width:260px;border-right:1px solid #ddd;display:flex;flex-direction:column;background:#fff;flex-shrink:0;}}
.tab-bar{{display:flex;border-bottom:1px solid #eee;flex-shrink:0;}}
.tab-btn{{flex:1;padding:7px 4px;font-size:11.5px;border:none;background:#f9f9f9;cursor:pointer;border-bottom:2px solid transparent;}}
.tab-btn.active{{background:#fff;border-bottom-color:#1565c0;color:#1565c0;font-weight:600;}}
.tab-content{{display:none;flex:1;overflow:hidden;flex-direction:column;}}
.tab-content.active{{display:flex;}}
#inj-list{{overflow-y:auto;flex:1;padding:4px;}}
.inj-item{{
  display:flex;align-items:flex-start;padding:5px 6px;cursor:pointer;
  border-radius:5px;margin-bottom:2px;transition:background 0.1s;
}}
.inj-item:hover{{background:#f4f4f4;}}
.inj-item.selected{{background:#e3f0ff;}}
.inj-item.dim{{display:none;}}
.dot{{width:10px;height:10px;border-radius:50%;margin-right:7px;flex-shrink:0;margin-top:3px;}}
.inj-txt{{font-size:11.5px;line-height:1.5;}}
.inj-txt .main{{font-weight:600;}}
.inj-txt .sub{{color:#666;}}
/* Overlap tab */
#ovl-controls{{padding:6px 8px;border-bottom:1px solid #eee;flex-shrink:0;font-size:11px;display:flex;align-items:center;gap:6px;}}
#ovl-table-wrap{{overflow-y:auto;flex:1;}}
#ovl-table{{width:100%;border-collapse:collapse;font-size:11px;}}
#ovl-table thead th{{position:sticky;top:0;background:#f5f5f5;padding:5px 4px;text-align:left;border-bottom:1px solid #ddd;cursor:pointer;user-select:none;white-space:nowrap;}}
#ovl-table thead th:hover{{background:#ebebeb;}}
#ovl-table tbody tr{{cursor:pointer;border-bottom:1px solid #f0f0f0;}}
#ovl-table tbody tr:hover{{background:#f0f6ff;}}
#ovl-table td{{padding:4px 4px;vertical-align:middle;}}
.pct-bar{{height:6px;border-radius:3px;background:#e53935;display:inline-block;margin-right:4px;}}
/* CENTER */
#center{{flex:1;position:relative;}}
#plot3d{{width:100%;height:100%;}}
#overlap-box{{
  display:none;position:absolute;bottom:12px;left:12px;z-index:10;
  background:rgba(255,255,255,0.95);border:1px solid #ccc;border-radius:6px;
  padding:8px 12px;font-size:11.5px;max-width:340px;box-shadow:0 2px 6px rgba(0,0,0,0.15);
  max-height:60vh;overflow-y:auto;
}}
#overlap-box h4{{font-size:12px;margin-bottom:5px;color:#333;}}
.ovl-section{{font-size:10.5px;font-weight:600;color:#555;margin:6px 0 2px;border-top:1px solid #eee;padding-top:4px;}}
/* RIGHT */
#right{{width:920px;border-left:1px solid #ddd;display:flex;flex-direction:column;background:#fff;flex-shrink:0;}}
#right-top{{display:flex;align-items:center;padding:6px 10px;gap:6px;border-bottom:1px solid #eee;flex-shrink:0;flex-wrap:wrap;}}
#right-top span{{font-size:11px;color:#555;white-space:nowrap;}}
#right-scroll{{overflow-y:auto;flex:1;padding:10px;}}
.fm-card{{margin-bottom:22px;}}
.fm-card .fm-title{{font-size:12px;font-weight:600;margin-bottom:4px;color:#333;display:flex;align-items:center;gap:6px;}}
.fm-row{{display:flex;gap:6px;align-items:flex-start;}}
.fm-col{{display:flex;flex-direction:column;align-items:flex-start;}}
.fm-col-label{{font-size:10px;color:#888;margin-bottom:2px;font-weight:600;text-transform:uppercase;}}
.fm-wrap{{position:relative;display:inline-block;}}
.fm-wrap img{{display:block;}}
.fm-wrap canvas{{position:absolute;top:0;left:0;pointer-events:none;}}
.fm-dl-row{{display:flex;gap:4px;margin-top:4px;}}
/* Buttons */
.btn{{
  padding:4px 10px;border:1px solid #ccc;border-radius:4px;cursor:pointer;
  font-size:11.5px;background:#fff;transition:background 0.1s,color 0.1s;white-space:nowrap;
}}
.btn:hover{{background:#f0f0f0;}}
.btn.active{{background:#1565c0;color:#fff;border-color:#1565c0;}}
.btn-clear{{padding:4px 10px;border:1px solid #e53935;border-radius:4px;cursor:pointer;
  font-size:11.5px;background:#fff;color:#e53935;white-space:nowrap;}}
.btn-clear:hover{{background:#fce4e4;}}
.btn-dl{{padding:3px 7px;border:1px solid #bbb;border-radius:3px;cursor:pointer;
  font-size:10px;background:#fafafa;color:#444;white-space:nowrap;}}
.btn-dl:hover{{background:#e8f0fe;color:#1565c0;border-color:#a0c0e8;}}
</style>
</head>
<body>

<div id="top-bar">
  <strong>Injection Explorer</strong>
  <span class="sep"></span>
  <span style="font-size:11px;color:#555;white-space:nowrap">STR:</span>
  <button class="btn active" id="btn-all"   onclick="setFilter('all')">All</button>
  <button class="btn"        id="btn-str80" onclick="setFilter('str80')">≥80%</button>
  <button class="btn"        id="btn-str90" onclick="setFilter('str90')">≥90%</button>
  <span class="sep"></span>
  <span style="font-size:11px;color:#555;white-space:nowrap">Genotype:</span>
  <button class="btn active" id="btn-geno-all"   onclick="setGeno('all')">All</button>
  <button class="btn"        id="btn-geno-fezf2" onclick="setGeno('fezf2')">Fezf2</button>
  <button class="btn"        id="btn-geno-tlx3"  onclick="setGeno('tlx3')">Tlx3</button>
  <span class="sep"></span>
  <span style="font-size:11px;color:#555;white-space:nowrap">Size:</span>
  <button class="btn active" id="btn-size-all"   onclick="setSizeFilter('all')">All</button>
  <button class="btn"        id="btn-size-small" onclick="setSizeFilter('small')">Small</button>
  <button class="btn"        id="btn-size-large" onclick="setSizeFilter('large')">Large</button>
  <span class="sep"></span>
  <span style="font-size:11px;color:#555;white-space:nowrap">Color:</span>
  <button class="btn active" id="btn-col-orig" onclick="setColorMode('orig')">Original</button>
  <button class="btn"        id="btn-col-ap"   onclick="setColorMode('ap')">AP (R→B)</button>
  <button class="btn"        id="btn-col-dv"   onclick="setColorMode('dv')">DV (R→Y)</button>
  <button class="btn"        id="btn-col-ml"   onclick="setColorMode('ml')">ML (B→O)</button>
  <span class="sep"></span>
  <span style="font-size:11px;color:#555;white-space:nowrap">Flatmaps:</span>
  <button class="btn active" id="btn-separate" onclick="setFM('separate')">L/R Separate</button>
  <button class="btn"        id="btn-overlay"  onclick="setFM('overlay')">Overlay</button>
  <span class="sep"></span>
  <button class="btn" onclick="selectAllVisible()">Select All</button>
  <button class="btn-clear" onclick="clearSel()">Clear</button>
  <label style="font-size:11px;color:#555;white-space:nowrap;display:flex;align-items:center;gap:3px;cursor:pointer;">
    <input type="checkbox" id="chk-overlap-cloud" checked onchange="toggleOverlapCloud(this.checked)">
    Overlap cloud
  </label>
  <label style="font-size:11px;color:#555;white-space:nowrap;display:flex;align-items:center;gap:3px;cursor:pointer;" title="Only show injections that have PFC cells projected onto the flatmap">
    <input type="checkbox" id="chk-pfc-filter" onchange="togglePfcFilter(this.checked)">
    Has PFC cells
  </label>
  <span id="sel-count" style="margin-left:4px;font-size:11px;color:#888;white-space:nowrap;"></span>
</div>

<div id="main">
  <!-- LEFT PANEL -->
  <div id="left">
    <div class="tab-bar">
      <button class="tab-btn active" id="tab-btn-list" onclick="showTab('list')">Injections</button>
      <button class="tab-btn"        id="tab-btn-ovl"  onclick="showTab('ovl')">Top Overlaps</button>
    </div>

    <!-- Injections tab -->
    <div class="tab-content active" id="tab-list">
      <div id="inj-list"></div>
    </div>

    <!-- Overlaps tab -->
    <div class="tab-content" id="tab-ovl">
      <div id="ovl-controls">
        Min %:
        <input type="range" id="ovl-min-slider" min="0" max="40" step="1" value="0"
               oninput="ovlMinChanged(this.value)" style="width:80px">
        <span id="ovl-min-val" style="width:28px">0%</span>
        <span id="ovl-count" style="color:#888"></span>
      </div>
      <div id="ovl-table-wrap">
        <table id="ovl-table">
          <thead>
            <tr>
              <th onclick="sortOvl('a')">Animal A ↕</th>
              <th onclick="sortOvl('b')">Animal B ↕</th>
              <th onclick="sortOvl('pct')" style="text-align:right">% ↕</th>
            </tr>
          </thead>
          <tbody id="ovl-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- CENTER PANEL -->
  <div id="center">
    <div id="plot3d"></div>
    <div id="overlap-box">
      <h4 id="overlap-title">Overlap (mesh-interpolated, 50 µm grid)</h4>
      <div id="overlap-content"></div>
    </div>
  </div>

  <!-- RIGHT PANEL -->
  <div id="right">
    <div id="right-top">
      <span>PFC flatmaps</span>
      <div id="grad-legend" style="display:none;align-items:center;gap:5px;font-size:10px;color:#555;margin-left:10px;">
        <span id="grad-lo-label"></span>
        <canvas id="grad-bar" width="110" height="11" style="border-radius:3px;vertical-align:middle;"></canvas>
        <span id="grad-hi-label"></span>
      </div>
      <div id="axis-slider-section" style="display:none;align-items:center;gap:6px;font-size:10.5px;color:#555;margin-left:14px;flex-wrap:nowrap;">
        <span style="white-space:nowrap;font-size:10px;">Slice at:</span>
        <span id="axis-sl-lo" style="white-space:nowrap;font-size:10px;color:#888;"></span>
        <input type="range" id="axis-slider" min="0" max="1" step="0.01" value="0.5"
          style="width:160px;cursor:pointer;"
          oninput="onAxisSliderInput(this.value)"
          onchange="onAxisSliderChange(this.value)">
        <span id="axis-sl-hi" style="white-space:nowrap;font-size:10px;color:#888;"></span>
        <span id="axis-sl-val" style="min-width:54px;font-size:11px;font-weight:600;color:#1565c0;"></span>
        <button class="btn" style="padding:2px 7px;font-size:10px;" onclick="clearAxisSlider()">All</button>
      </div>
    </div>
    <div id="right-scroll">
      <div id="flatmap-container"></div>
    </div>
  </div>
</div>

<script>
// ============================================================
// DATA
// ============================================================
const DATA = {payload};

// ============================================================
// STATE
// ============================================================
let selectedIds = new Set();
let filterMode  = 'all';
let genoFilter  = 'all';      // 'all' | 'fezf2' | 'tlx3'
let sizeFilter  = 'all';      // 'all' | 'small' | 'large'
let pfcFilter   = false;      // show only injections with ≥1 PFC cell projected
let colorMode   = 'orig';     // 'orig' | 'ap' | 'dv' | 'ml'
let showOverlapCloud = true;
let axisSliderVal = null;     // null = no filter; number = mm value on current axis
let fmMode      = 'separate';   // 'separate' or 'overlay'
let ovlMinPct   = 0;
let ovlSortKey  = 'pct';
let ovlSortAsc  = false;

// Plotly trace bookkeeping
const N_ATLAS = DATA.atlas_traces.length;
let overlapTraceIdx  = -1;
let combinedTraceIdx = -1;

// When ≤ MESH_THRESHOLD injections are selected, use individual mesh3d surfaces.
// For larger selections, automatically switch to a single combined scatter3d
// point-cloud (one WebGL draw call — much faster than 77 mesh surfaces).
const MESH_THRESHOLD = 8;

// ── Gradient colour ranges — computed once from all injection centroids ───────
const _apVals = DATA.injections.map(i => i.centroid && i.centroid.ap != null ? i.centroid.ap : null).filter(v => v !== null);
const _dvVals = DATA.injections.map(i => i.centroid && i.centroid.dv != null ? i.centroid.dv : null).filter(v => v !== null);
const AP_MIN = _apVals.length ? Math.min(..._apVals) : 0;
const AP_MAX = _apVals.length ? Math.max(..._apVals) : 1;
const DV_MIN = _dvVals.length ? Math.min(..._dvVals) : -5;
const DV_MAX = _dvVals.length ? Math.max(..._dvVals) : 0;

const _mlVals = DATA.injections.map(i => i.centroid && i.centroid.ml != null ? i.centroid.ml : null).filter(v => v !== null);
const ML_MIN = _mlVals.length ? Math.min(..._mlVals) : 0;
const ML_MAX = _mlVals.length ? Math.max(..._mlVals) : 3;

// ── Injection size threshold (50 µm voxel count) ─────────────────────────────
const _vols = DATA.injections.map(i => i.vox.length).slice().sort((a,b)=>a-b);
const VOL_MEDIAN = _vols.length ? _vols[Math.floor(_vols.length / 2)] : 0;
const VOL_THRESHOLD = 1.5 * VOL_MEDIAN;   // boundary between small and large

// ── Voxel packing constants — 50 µm grid (must match Python pack_voxels) ──
const _ML_N  = 229;
const _DV_N  = 161;
const _DV_ML = _ML_N * _DV_N;  // 36869

function unpackVox(v) {{
  const ml = v % _ML_N;
  const dv = Math.floor(v / _ML_N) % _DV_N;
  const ap = Math.floor(v / _DV_ML);
  return [ap * 50, dv * 50, ml * 50];
}}

// ============================================================
// HELPERS
// ============================================================
function injById(id) {{
  return DATA.injections.find(i => i.id === id);
}}
function injLabel(inj) {{
  return `${{inj.animal}} ${{inj.hemi}} (${{inj.ap_pos}}, ${{inj.subregion}})`;
}}

// Return display color for injection, respecting the current colorMode.
// Uses direct RGB interpolation so no intermediate hues appear.
function getInjColor(inj) {{
  if (colorMode === 'ap') {{
    const ap = inj.centroid ? inj.centroid.ap : null;
    if (ap == null) return '#aaa';
    const t = AP_MAX > AP_MIN ? (ap - AP_MIN) / (AP_MAX - AP_MIN) : 0.5;
    // t=0 → posterior → blue (0,0,220); t=1 → anterior → red (220,0,0)
    const r = Math.round(t * 220);
    const b = Math.round((1 - t) * 220);
    return 'rgb(' + r + ',0,' + b + ')';
  }}
  if (colorMode === 'dv') {{
    const dv = inj.centroid ? inj.centroid.dv : null;
    if (dv == null) return '#aaa';
    const t = DV_MAX > DV_MIN ? (dv - DV_MIN) / (DV_MAX - DV_MIN) : 0.5;
    // t=0 → ventral → yellow (255,220,0); t=1 → dorsal → red (220,0,0)
    const r = Math.round(220 + (1 - t) * 35);
    const g = Math.round((1 - t) * 220);
    return 'rgb(' + r + ',' + g + ',0)';
  }}
  if (colorMode === 'ml') {{
    const ml = inj.centroid ? inj.centroid.ml : null;
    if (ml == null) return '#aaa';
    const t = ML_MAX > ML_MIN ? (ml - ML_MIN) / (ML_MAX - ML_MIN) : 0.5;
    // t=0 → medial → blue (0,0,220); t=1 → lateral → orange (255,140,0)
    const r = Math.round(t * 255);
    const g = Math.round(t * 140);
    const b = Math.round((1 - t) * 220);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }}
  return inj.color;
}}

function updateGradLegend() {{
  const el = document.getElementById('grad-legend');
  if (colorMode === 'orig') {{ el.style.display = 'none'; return; }}
  el.style.display = 'flex';
  const canvas = document.getElementById('grad-bar');
  const ctx = canvas.getContext('2d');
  const grd = ctx.createLinearGradient(0, 0, 110, 0);
  if (colorMode === 'ap') {{
    document.getElementById('grad-lo-label').textContent = 'Post ' + AP_MIN.toFixed(1) + 'mm';
    document.getElementById('grad-hi-label').textContent = 'Ant ' + AP_MAX.toFixed(1) + 'mm';
    grd.addColorStop(0, 'rgb(0,0,220)');
    grd.addColorStop(1, 'rgb(220,0,0)');
  }} else if (colorMode === 'dv') {{
    document.getElementById('grad-lo-label').textContent = 'Vent ' + DV_MIN.toFixed(1) + 'mm';
    document.getElementById('grad-hi-label').textContent = 'Dors ' + DV_MAX.toFixed(1) + 'mm';
    grd.addColorStop(0, 'rgb(255,220,0)');
    grd.addColorStop(1, 'rgb(220,0,0)');
  }} else {{
    document.getElementById('grad-lo-label').textContent = 'Med ' + ML_MIN.toFixed(1) + 'mm';
    document.getElementById('grad-hi-label').textContent = 'Lat ' + ML_MAX.toFixed(1) + 'mm';
    grd.addColorStop(0, 'rgb(0,0,220)');
    grd.addColorStop(1, 'rgb(255,140,0)');
  }}
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, 110, 11);
}}

// updateInjColors → now handled inside updatePlot() to support both mesh and scatter modes

// ============================================================
// FILTER HELPERS
// ============================================================
function visibleInjections() {{
  let injs = DATA.injections;
  if (filterMode === 'str90') injs = injs.filter(i => i.pct_str !== null && i.pct_str >= 90);
  else if (filterMode === 'str80') injs = injs.filter(i => i.pct_str !== null && i.pct_str >= 80);
  if (genoFilter === 'fezf2') injs = injs.filter(i => i.geno === 'fezf2');
  else if (genoFilter === 'tlx3') injs = injs.filter(i => i.geno === 'tlx3');
  if (sizeFilter === 'small') injs = injs.filter(i => i.vox.length <= VOL_THRESHOLD);
  else if (sizeFilter === 'large') injs = injs.filter(i => i.vox.length > VOL_THRESHOLD);
  if (pfcFilter) injs = injs.filter(i =>
    (i.cells_ipsi  && i.cells_ipsi.u  && i.cells_ipsi.u.length  > 0) ||
    (i.cells_contra && i.cells_contra.u && i.cells_contra.u.length > 0));
  return injs;
}}

// ============================================================
// TABS
// ============================================================
function showTab(name) {{
  document.getElementById('tab-list').classList.toggle('active', name==='list');
  document.getElementById('tab-ovl' ).classList.toggle('active', name==='ovl');
  document.getElementById('tab-btn-list').classList.toggle('active', name==='list');
  document.getElementById('tab-btn-ovl' ).classList.toggle('active', name==='ovl');
  if (name === 'ovl') buildOvlTable();
}}

// ============================================================
// INJECTION LIST
// ============================================================
function buildList() {{
  const vis  = new Set(visibleInjections().map(i => i.id));
  const list = document.getElementById('inj-list');
  list.innerHTML = '';
  DATA.injections.forEach(inj => {{
    const strTxt  = inj.pct_str    !== null ? inj.pct_str.toFixed(1)+'%'    : 'n/a';
    const ctxTxt  = inj.pct_cortex !== null ? inj.pct_cortex.toFixed(1)+'%' : 'n/a';
    const item = document.createElement('div');
    item.className = 'inj-item' +
      (selectedIds.has(inj.id) ? ' selected' : '') +
      (!vis.has(inj.id)        ? ' dim'      : '');
    item.innerHTML = `
      <div class="dot" style="background:${{getInjColor(inj)}}"></div>
      <div class="inj-txt">
        <div class="main">${{inj.animal}} ${{inj.hemi}}</div>
        <div class="sub">${{inj.ap_pos}} · <b>${{inj.subregion}}</b> · ${{inj.geno}}</div>
        <div class="sub">STR: ${{strTxt}} | CTX: ${{ctxTxt}}</div>
      </div>`;
    item.onclick = () => toggleSel(inj.id);
    list.appendChild(item);
  }});
}}

function toggleSel(id) {{
  const vis = new Set(visibleInjections().map(i => i.id));
  if (!vis.has(id)) return;
  if (selectedIds.has(id)) selectedIds.delete(id);
  else                      selectedIds.add(id);
  onSelectionChange();
}}

function clearSel() {{
  selectedIds.clear();
  onSelectionChange();
}}

// ============================================================
// OVERLAP TABLE (pre-computed pairs)
// ============================================================
function filteredPairs() {{
  const vis = new Set(visibleInjections().map(i => i.id));
  return DATA.pairs.filter(p =>
    p.pct >= ovlMinPct && vis.has(p.a) && vis.has(p.b)
  );
}}

function buildOvlTable() {{
  const pairs = filteredPairs();
  const sorted = [...pairs].sort((x, y) => {{
    let va = x[ovlSortKey], vb = y[ovlSortKey];
    if (typeof va === 'string') va = va.toLowerCase(), vb = vb.toLowerCase();
    return ovlSortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  }});

  document.getElementById('ovl-count').textContent = `${{sorted.length}} pairs`;
  const maxPct = sorted.length ? sorted.reduce((m,p)=>Math.max(m,p.pct),0) : 40;

  const tbody = document.getElementById('ovl-tbody');
  tbody.innerHTML = '';
  sorted.forEach(p => {{
    const ia = injById(p.a), ib = injById(p.b);
    const tr = document.createElement('tr');
    const barW = Math.round(p.pct / maxPct * 36);
    tr.innerHTML = `
      <td>
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${{ia ? getInjColor(ia) : '#ccc'}};margin-right:4px"></span>
        ${{ia ? ia.animal+' '+ia.hemi : p.a}}
        <div style="font-size:10px;color:#999">${{ia?.ap_pos}} ${{ia?.subregion}}</div>
      </td>
      <td>
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${{ib ? getInjColor(ib) : '#ccc'}};margin-right:4px"></span>
        ${{ib ? ib.animal+' '+ib.hemi : p.b}}
        <div style="font-size:10px;color:#999">${{ib?.ap_pos}} ${{ib?.subregion}}</div>
      </td>
      <td style="text-align:right;white-space:nowrap">
        <span class="pct-bar" style="width:${{barW}}px"></span>
        <b>${{p.pct}}%</b>
      </td>`;
    tr.onclick = () => {{ selectPair(p.a, p.b); }};
    tbody.appendChild(tr);
  }});
}}

function sortOvl(key) {{
  if (ovlSortKey === key) ovlSortAsc = !ovlSortAsc;
  else {{ ovlSortKey = key; ovlSortAsc = key !== 'pct'; }}
  buildOvlTable();
}}

function ovlMinChanged(val) {{
  ovlMinPct = parseFloat(val);
  document.getElementById('ovl-min-val').textContent = val + '%';
  buildOvlTable();
}}

function selectPair(idA, idB) {{
  selectedIds.clear();
  selectedIds.add(idA);
  selectedIds.add(idB);
  showTab('list');
  onSelectionChange();
}}

// ============================================================
// FILTER / FLATMAP-MODE CONTROLS
// ============================================================
function setGeno(mode) {{
  genoFilter = mode;
  ['btn-geno-all','btn-geno-fezf2','btn-geno-tlx3'].forEach(function(id) {{
    document.getElementById(id).classList.remove('active');
  }});
  document.getElementById(
    mode==='fezf2' ? 'btn-geno-fezf2' : mode==='tlx3' ? 'btn-geno-tlx3' : 'btn-geno-all'
  ).classList.add('active');
  const vis = new Set(visibleInjections().map(i => i.id));
  for (const id of [...selectedIds]) {{ if (!vis.has(id)) selectedIds.delete(id); }}
  buildList();
  buildOvlTable();
  updatePlot();
  updateFlatmaps();
}}

function setColorMode(mode) {{
  colorMode = mode;
  ['btn-col-orig','btn-col-ap','btn-col-dv','btn-col-ml'].forEach(function(id) {{
    document.getElementById(id).classList.remove('active');
  }});
  document.getElementById(
    mode==='ap' ? 'btn-col-ap' : mode==='dv' ? 'btn-col-dv' :
    mode==='ml' ? 'btn-col-ml' : 'btn-col-orig'
  ).classList.add('active');
  axisSliderVal = null;   // reset slider position when axis changes
  updatePlot();
  updateFlatmaps();
  updateGradLegend();
  updateAxisSlider();
}}

function selectAllVisible() {{
  visibleInjections().forEach(function(inj) {{ selectedIds.add(inj.id); }});
  onSelectionChange();
}}

function toggleOverlapCloud(checked) {{
  showOverlapCloud = checked;
  updatePlot();
}}

function togglePfcFilter(checked) {{
  pfcFilter = checked;
  const vis = new Set(visibleInjections().map(i => i.id));
  for (const id of [...selectedIds]) {{ if (!vis.has(id)) selectedIds.delete(id); }}
  buildList();
  buildOvlTable();
  updatePlot();
  updateFlatmaps();
}}

// Return injs filtered to those whose full voxel volume spans the slider position.
// If slider is inactive (axisSliderVal===null) or mode is 'orig', returns all.
function getAxisFilteredInjs(injs) {{
  if (axisSliderVal === null || colorMode === 'orig') return injs;
  const v = axisSliderVal;
  return injs.filter(function(inj) {{
    let r;
    if (colorMode === 'ap') r = inj.ap_range;
    else if (colorMode === 'dv') r = inj.dv_range;
    else r = inj.ml_range;
    return r && r[0] != null && r[0] <= v && v <= r[1];
  }});
}}

function updateAxisSlider() {{
  const sec = document.getElementById('axis-slider-section');
  if (colorMode === 'orig') {{
    sec.style.display = 'none';
    axisSliderVal = null;
    return;
  }}
  sec.style.display = 'flex';
  let lo, hi, loLabel, hiLabel;
  if (colorMode === 'ap') {{
    lo = AP_MIN; hi = AP_MAX;
    loLabel = 'Post ' + AP_MIN.toFixed(2) + ' mm';
    hiLabel = 'Ant '  + AP_MAX.toFixed(2) + ' mm';
  }} else if (colorMode === 'dv') {{
    lo = DV_MIN; hi = DV_MAX;
    loLabel = 'Vent ' + DV_MIN.toFixed(2) + ' mm';
    hiLabel = 'Dors ' + DV_MAX.toFixed(2) + ' mm';
  }} else {{
    lo = ML_MIN; hi = ML_MAX;
    loLabel = 'Med ' + ML_MIN.toFixed(2) + ' mm';
    hiLabel = 'Lat ' + ML_MAX.toFixed(2) + ' mm';
  }}
  document.getElementById('axis-sl-lo').textContent = loLabel;
  document.getElementById('axis-sl-hi').textContent = hiLabel;
  const sl = document.getElementById('axis-slider');
  sl.min = lo; sl.max = hi;
  sl.step = ((hi - lo) / 200).toFixed(4);
  if (axisSliderVal === null) {{
    sl.value = (lo + hi) / 2;
    document.getElementById('axis-sl-val').textContent = '— all —';
  }} else {{
    sl.value = Math.max(lo, Math.min(hi, axisSliderVal));
    document.getElementById('axis-sl-val').textContent = parseFloat(sl.value).toFixed(2) + ' mm';
  }}
}}

function onAxisSliderInput(val) {{
  axisSliderVal = parseFloat(val);
  document.getElementById('axis-sl-val').textContent = axisSliderVal.toFixed(2) + ' mm';
  updateFlatmaps();  // fast: canvas only
}}

function onAxisSliderChange(val) {{
  axisSliderVal = parseFloat(val);
  updatePlot();      // also refresh 3D
}}

function clearAxisSlider() {{
  axisSliderVal = null;
  document.getElementById('axis-sl-val').textContent = '— all —';
  const sl = document.getElementById('axis-slider');
  sl.value = (parseFloat(sl.min) + parseFloat(sl.max)) / 2;
  updateFlatmaps();
  updatePlot();
}}

function setSizeFilter(mode) {{
  sizeFilter = mode;
  ['btn-size-all','btn-size-small','btn-size-large'].forEach(function(id) {{
    document.getElementById(id).classList.remove('active');
  }});
  document.getElementById(
    mode==='small' ? 'btn-size-small' : mode==='large' ? 'btn-size-large' : 'btn-size-all'
  ).classList.add('active');
  const vis = new Set(visibleInjections().map(i => i.id));
  for (const id of [...selectedIds]) {{ if (!vis.has(id)) selectedIds.delete(id); }}
  buildList();
  buildOvlTable();
  updatePlot();
  updateFlatmaps();
}}

function setFilter(mode) {{
  filterMode = mode;
  ['btn-all','btn-str80','btn-str90'].forEach(id =>
    document.getElementById(id).classList.remove('active'));
  document.getElementById(
    mode==='str90' ? 'btn-str90' : mode==='str80' ? 'btn-str80' : 'btn-all'
  ).classList.add('active');
  const vis = new Set(visibleInjections().map(i => i.id));
  for (const id of [...selectedIds]) {{ if (!vis.has(id)) selectedIds.delete(id); }}
  buildList();
  buildOvlTable();
  updatePlot();
  updateFlatmaps();
}}

function setFM(mode) {{
  fmMode = mode;
  document.getElementById('btn-separate').classList.toggle('active', mode==='separate');
  document.getElementById('btn-overlay' ).classList.toggle('active', mode==='overlay');
  updateFlatmaps();
}}

// ============================================================
// 3D PLOT
// ============================================================
function makePlotlyInjTrace(inj) {{
  return {{
    type: 'mesh3d',
    x: inj.mx, y: inj.my, z: inj.mz,
    i: inj.mi, j: inj.mj, k: inj.mk,
    opacity: 0.65,
    color: getInjColor(inj),
    name: injLabel(inj),
    showlegend: true,
    hoverinfo: 'name',
    visible: false,   // hidden until selected — drastically speeds up initial render
  }};
}}

function initPlot() {{
  const traces = [
    ...DATA.atlas_traces,
    ...DATA.injections.map(makePlotlyInjTrace),
  ];
  const layout = {{
    scene: {{
      xaxis: {{title:'AP (µm)',showgrid:false,showticklabels:false,zeroline:false,showbackground:false}},
      yaxis: {{title:'DV (µm)',showgrid:false,showticklabels:false,zeroline:false,showbackground:false}},
      zaxis: {{title:'ML (µm)',showgrid:false,showticklabels:false,zeroline:false,showbackground:false,autorange:'reversed'}},
      aspectmode:'data', bgcolor:'white',
    }},
    legend: {{x:0.01,y:0.99,bgcolor:'rgba(255,255,255,0.85)',bordercolor:'#ccc',borderwidth:1,font:{{size:10}}}},
    margin: {{t:10,b:10,l:10,r:10}},
    showlegend: true,
  }};
  Plotly.newPlot('plot3d', traces, layout, {{responsive:true}});
}}

function updatePlot() {{
  const visIds    = new Set(visibleInjections().map(i => i.id));
  const selInjs   = DATA.injections.filter(i => selectedIds.has(i.id) && visIds.has(i.id));
  // Apply axis slider filter for 3D display (uses full voxel extent, not centroid)
  const displayInjs = getAxisFilteredInjs(selInjs);
  const n = displayInjs.length;

  // ── Remove stale extra traces (highest index first to preserve others) ─────
  const stale = [overlapTraceIdx, combinedTraceIdx].filter(i => i >= 0).sort((a,b) => b-a);
  stale.forEach(function(idx) {{ Plotly.deleteTraces('plot3d', idx); }});
  overlapTraceIdx = -1; combinedTraceIdx = -1;

  const useMesh = n > 0 && n <= MESH_THRESHOLD;
  const displaySet = new Set(displayInjs.map(i => i.id));

  // ── Individual mesh visibility ─────────────────────────────────────────────
  const injVis = DATA.injections.map(function(inj) {{
    return useMesh && displaySet.has(inj.id);
  }});
  const allIndices = Array.from({{length: N_ATLAS + DATA.injections.length}}, function(_,i){{return i;}});
  Plotly.restyle('plot3d', {{visible: [...Array(N_ATLAS).fill(true), ...injVis]}}, allIndices);

  if (useMesh) {{
    // ── Update colors of visible mesh traces (one batched call) ───────────────
    const ci = [], cc = [];
    displayInjs.forEach(function(inj) {{
      ci.push(N_ATLAS + DATA.injections.indexOf(inj));
      cc.push(getInjColor(inj));
    }});
    if (ci.length) Plotly.restyle('plot3d', {{color: cc}}, ci);

    // ── Overlap cloud + stats for small mesh selections ────────────────────
    if (n >= 2) {{
      const {{overlapVox, allStats}} = computeOverlapStats(displayInjs);
      if (showOverlapCloud && overlapVox.length > 0) {{
        const voxCountMap = new Map();
        const sets = displayInjs.map(function(inj) {{ return new Set(inj.vox); }});
        for (const s of sets) for (const v of s)
          voxCountMap.set(v, (voxCountMap.get(v)||0)+1);
        const xs=[], ys=[], zs=[], cols=[];
        for (const v of overlapVox) {{
          const [a,d,m] = unpackVox(v);
          xs.push(a); ys.push(d); zs.push(m);
          const cnt = voxCountMap.get(v)||2;
          const t = (cnt-2)/Math.max(1,n-2);
          cols.push('hsl('+Math.round(240-t*180)+',80%,50%)');
        }}
        Plotly.addTraces('plot3d', {{
          type:'scatter3d', x:xs, y:ys, z:zs, mode:'markers',
          marker:{{size:3, color:cols, opacity:0.8}},
          name:'Overlap ('+overlapVox.length+' voxels, 50µm)',
          hoverinfo:'name', showlegend:true,
        }});
        overlapTraceIdx = document.getElementById('plot3d').data.length - 1;
      }}
      showOverlapInfo(selInjs, allStats);
    }} else {{
      document.getElementById('overlap-box').style.display = 'none';
    }}

  }} else if (n > 0) {{
    // ── Combined scatter3d for large selections ────────────────────────────
    const voxMap = new Map();
    displayInjs.forEach(function(inj, i) {{
      for (const v of inj.vox) {{
        const cur = voxMap.get(v);
        if (!cur) voxMap.set(v, {{injIdx:i, count:1}});
        else cur.count++;
      }}
    }});
    const xs=[], ys=[], zs=[], cols=[];
    for (const [v, info] of voxMap) {{
      const [a,d,m] = unpackVox(v);
      xs.push(a); ys.push(d); zs.push(m);
      if (showOverlapCloud && info.count > 1) {{
        const t = (info.count-1)/Math.max(1,n-1);
        cols.push('hsl('+Math.round(240-t*180)+',80%,50%)');
      }} else {{
        cols.push(getInjColor(displayInjs[info.injIdx]));
      }}
    }}
    Plotly.addTraces('plot3d', {{
      type:'scatter3d', x:xs, y:ys, z:zs, mode:'markers',
      marker:{{size:2.5, color:cols, opacity:0.75}},
      name:n+' injections ('+xs.length.toLocaleString()+' voxels)',
      hoverinfo:'skip', showlegend:false,
    }});
    combinedTraceIdx = document.getElementById('plot3d').data.length - 1;
    document.getElementById('overlap-box').style.display = 'none';

  }} else {{
    document.getElementById('overlap-box').style.display = 'none';
  }}

  document.getElementById('sel-count').textContent = n > 0 ? n+' selected' : '';
  buildList();
}}

// ============================================================
// OVERLAP COMPUTATION — all combinations of selected injections
// ============================================================

// Returns all size-k subsets of arr
function combinations(arr, k) {{
  if (k === 0) return [[]];
  if (arr.length < k) return [];
  const [head, ...tail] = arr;
  return [
    ...combinations(tail, k-1).map(c => [head, ...c]),
    ...combinations(tail, k),
  ];
}}

function computeOverlapStats(selInjs) {{
  const sets = selInjs.map(inj => new Set(inj.vox));

  // Voxels shared by at least 2 injections (for 3D cloud)
  const voxCount = new Map();
  for (const s of sets) for (const v of s)
    voxCount.set(v, (voxCount.get(v)||0)+1);
  const overlapVox = [];
  for (const [v,c] of voxCount) {{ if (c >= 2) overlapVox.push(v); }}

  const n = selInjs.length;
  const allStats = [];

  // All pairwise (k=2) through k=n combinations
  const maxK = Math.min(n, 5);  // cap at 5-way to avoid excessive display
  for (let k = 2; k <= maxK; k++) {{
    const combos = combinations(selInjs.map((_,i)=>i), k);
    for (const idxs of combos) {{
      const sub = idxs.map(i => selInjs[i]);
      const subSets = idxs.map(i => sets[i]);
      // Intersection of all k sets
      let isect = new Set(subSets[0]);
      for (let p = 1; p < subSets.length; p++) {{
        isect = new Set([...isect].filter(v => subSets[p].has(v)));
      }}
      // Union of all k sets
      const union = new Set();
      for (const s of subSets) for (const v of s) union.add(v);
      const isectSz = isect.size, unionSz = union.size;
      const pct = unionSz > 0 ? (isectSz/unionSz*100).toFixed(1) : '0.0';
      allStats.push({{
        k, labels: sub.map(i => i.animal+' '+i.hemi),
        colors: sub.map(i => i.color),
        pct, isect: isectSz, union: unionSz,
      }});
    }}
  }}
  return {{overlapVox, allStats}};
}}

function showOverlapInfo(selInjs, allStats) {{
  const box = document.getElementById('overlap-box');
  if (!allStats.length) {{ box.style.display='none'; return; }}

  // Group by k
  const byK = {{}};
  for (const s of allStats) {{
    if (!byK[s.k]) byK[s.k] = [];
    byK[s.k].push(s);
  }}

  let html = '';
  for (const k of Object.keys(byK).map(Number).sort()) {{
    const kName = k===2 ? 'Pairwise' : k===3 ? 'Triple' : k===4 ? 'Quadruple' : `${{k}}-way`;
    html += `<div class="ovl-section">${{kName}}</div>`;
    for (const s of byK[k]) {{
      const dots = s.colors.map(c =>
        `<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${{c}};margin-right:2px"></span>`
      ).join('');
      html += `<div style="margin-bottom:3px">
        ${{dots}}<span style="font-size:10.5px">${{s.labels.join(' ∩ ')}}</span><br>
        <b style="color:#e53935;font-size:12px">${{s.pct}}%</b>
        <span style="color:#999;font-size:10px"> (${{s.isect}} shared / ${{s.union}} union)</span>
      </div>`;
    }}
  }}
  document.getElementById('overlap-content').innerHTML = html;
  box.style.display = 'block';
}}

// ============================================================
// FLATMAPS — L/R separate or overlay
// ============================================================
// Left flatmap:  cells_ipsi  — ipsilateral cells, unmirrored SVG
// Right flatmap: cells_contra — contralateral cells, SVG mirrored horizontally
//   (SVG image gets CSS scaleX(-1); canvas draws at x = FM_HALF - u*sx)
// ============================================================
const FM_HALF = 445;   // width of each hemisphere flatmap

// Map a NRRD UV coordinate (original 1500-wide space) to canvas pixels.
// mirror=false → left-hem SVG (ipsi):  straightforward crop-and-scale
// mirror=true  → right-hem SVG (contra): u is mirrored in original space first,
//                then mapped into the right-hem SVG's own cropped viewBox
function uvToCanvas(u, v, w, h, mirror) {{
  if (mirror) {{
    const u_mir = DATA.svg_orig_w - u;           // mirror in original space
    const sx = w / DATA.svg_w_r, sy = h / DATA.svg_h_r;
    return [(u_mir - DATA.svg_x0_r) * sx, (v - DATA.svg_y0_r) * sy];
  }} else {{
    const sx = w / DATA.svg_w,   sy = h / DATA.svg_h;
    return [(u - DATA.svg_x0) * sx,   (v - DATA.svg_y0) * sy];
  }}
}}

// Overlay draw: pool every cell from every selected injection into one array,
// Fisher-Yates shuffle it, then draw in that random order.
// This removes z-order bias — no injection systematically paints on top.
function drawOverlayShuffled(ctx, selInjs, w, h, mirror) {{
  const pool = [];
  selInjs.forEach(function(inj) {{
    const cells = mirror ? inj.cells_contra : inj.cells_ipsi;
    if (!cells || !cells.u || !cells.u.length) return;
    const color = getInjColor(inj);
    for (let i = 0; i < cells.u.length; i++) {{
      pool.push({{ u: cells.u[i], v: cells.v[i], color: color }});
    }}
  }});
  if (!pool.length) return;

  // Fisher-Yates shuffle
  for (let i = pool.length - 1; i > 0; i--) {{
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
  }}

  ctx.globalAlpha = 0.75;
  pool.forEach(function(pt) {{
    const xy = uvToCanvas(pt.u, pt.v, w, h, mirror);
    ctx.fillStyle = pt.color;
    ctx.beginPath(); ctx.arc(xy[0], xy[1], 2.2, 0, Math.PI*2); ctx.fill();
  }});
  ctx.globalAlpha = 1.0;
}}

// Draw cells onto a canvas using the correct hemisphere's coordinate mapping.
function drawCellsOnCanvas(ctx, cells, color, w, h, mirror) {{
  if (!cells || !cells.u || !cells.u.length) return;
  ctx.fillStyle = color; ctx.globalAlpha = 0.75;
  for (let k = 0; k < cells.u.length; k++) {{
    const [cx, cy] = uvToCanvas(cells.u[k], cells.v[k], w, h, mirror);
    ctx.beginPath(); ctx.arc(cx, cy, 2.2, 0, Math.PI*2); ctx.fill();
  }}
  ctx.globalAlpha = 1.0;
}}

// Build one complete flatmap column (label + SVG + canvas overlay + download row)
function buildFlatmapCol(cells, color, labelText, mirror, downloadName) {{
  const svgW = mirror ? DATA.svg_w_r : DATA.svg_w;
  const svgH = mirror ? DATA.svg_h_r : DATA.svg_h;
  const fmH = Math.round(FM_HALF * svgH / svgW);
  const col = document.createElement('div');
  col.className = 'fm-col';

  const lbl = document.createElement('div');
  lbl.className = 'fm-col-label'; lbl.textContent = labelText;
  col.appendChild(lbl);

  const wrap = document.createElement('div');
  wrap.className = 'fm-wrap';
  wrap.style.width  = FM_HALF + 'px';
  wrap.style.height = fmH     + 'px';

  // SVG background — use pre-mirrored SVG for right hemisphere
  const img = document.createElement('img');
  img.src = mirror ? DATA.flatmap_url_r : DATA.flatmap_url;
  img.style.cssText = `width:${{FM_HALF}}px;height:${{fmH}}px;display:block;`;
  wrap.appendChild(img);

  // Canvas overlay
  const canvas = document.createElement('canvas');
  canvas.width = FM_HALF; canvas.height = fmH;
  canvas.style.cssText = `position:absolute;top:0;left:0;pointer-events:none;width:${{FM_HALF}}px;height:${{fmH}}px;`;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, FM_HALF, fmH);
  drawCellsOnCanvas(ctx, cells, color, FM_HALF, fmH, mirror);
  wrap.appendChild(canvas);
  col.appendChild(wrap);

  // Download row
  const nCells = cells?.u?.length || 0;
  const dlRow = document.createElement('div');
  dlRow.className = 'fm-dl-row';
  dlRow.innerHTML = `<span style="font-size:10px;color:#999">${{nCells.toLocaleString()}} cells</span>`;

  const bSVG = document.createElement('button');
  bSVG.className = 'btn-dl'; bSVG.textContent = '↓ SVG';
  bSVG.onclick = () => downloadFlatmapSVG([{{cells, color}}], mirror, downloadName+'.svg');
  dlRow.appendChild(bSVG);

  const bPNG = document.createElement('button');
  bPNG.className = 'btn-dl'; bPNG.textContent = '↓ PNG';
  bPNG.onclick = () => downloadFlatmapPNG(img, canvas, mirror, downloadName+'.png');
  dlRow.appendChild(bPNG);
  col.appendChild(dlRow);
  return col;
}}

// SVG download — applies mirror transform in SVG if needed
function downloadFlatmapSVG(layers, mirror, filename) {{
  const b64 = DATA.flatmap_url.split(',')[1];
  let svgText = '';
  try {{ svgText = decodeURIComponent(escape(atob(b64))); }}
  catch(e) {{ svgText = atob(b64); }}

  // Circles are in the SVG's own coordinate space (after mirroring for right-hem SVG).
  // For right-hem: u is mirrored → orig_w - u, which lands in the mirrored SVG's coords.
  let circles = '';
  for (const {{cells, color}} of layers) {{
    if (!cells || !cells.u) continue;
    for (let k = 0; k < cells.u.length; k++) {{
      const cx = mirror ? (DATA.svg_orig_w - cells.u[k]) : cells.u[k];
      circles += `<circle cx="${{cx.toFixed(1)}}" cy="${{cells.v[k].toFixed(1)}}" r="2.2" fill="${{color}}" opacity="0.75"/>`;
    }}
  }}

  // If mirrored, wrap SVG content in horizontal flip group
  if (mirror) {{
    // Insert flip transform around everything, then add circles
    svgText = svgText.replace(
      /(<svg[^>]*>)/,
      `$1<g transform="translate(${{DATA.svg_w}},0) scale(-1,1)">`
    ).replace('</svg>', `</g><g id="cells">${{circles}}</g></svg>`);
  }} else {{
    svgText = svgText.replace('</svg>', `<g id="cells">${{circles}}</g></svg>`);
  }}

  const blob = new Blob([svgText], {{type:'image/svg+xml'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}}

// PNG download — composites SVG (accounting for CSS mirror) + canvas
function downloadFlatmapPNG(imgEl, canvasEl, mirror, filename) {{
  const off = document.createElement('canvas');
  off.width = canvasEl.width; off.height = canvasEl.height;
  const ctx = off.getContext('2d');
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, off.width, off.height);
  try {{ ctx.drawImage(imgEl, 0, 0, off.width, off.height); }} catch(e) {{}}
  ctx.drawImage(canvasEl, 0, 0);
  const url = off.toDataURL('image/png');
  const a = document.createElement('a'); a.href = url; a.download = filename; a.click();
}}

function updateFlatmaps() {{
  const container = document.getElementById('flatmap-container');
  container.innerHTML = '';

  const selInjs = DATA.injections.filter(i => selectedIds.has(i.id));
  if (!selInjs.length) {{
    container.innerHTML =
      '<div style="font-size:11px;color:#aaa;padding:8px">Select one or more injections to view flatmaps</div>';
    return;
  }}

  if (fmMode === 'separate') {{
    // One card per injection — ipsilateral (left SVG) + contralateral (mirrored SVG)
    selInjs.forEach(inj => {{
      const card = document.createElement('div');
      card.className = 'fm-card';

      const strTxt = inj.pct_str    !== null ? ` · STR ${{inj.pct_str.toFixed(1)}}%` : '';
      const ctxTxt = inj.pct_cortex !== null ? ` · CTX ${{inj.pct_cortex.toFixed(1)}}%` : '';
      card.innerHTML = `
        <div class="fm-title">
          <span style="width:10px;height:10px;border-radius:50%;background:${{getInjColor(inj)}};display:inline-block"></span>
          ${{inj.animal}} ${{inj.hemi}} · ${{inj.ap_pos}} · <b>${{inj.subregion}}</b> · ${{inj.geno}}${{strTxt}}${{ctxTxt}}
        </div>`;

      const row = document.createElement('div');
      row.className = 'fm-row';
      const baseName = `${{inj.animal}}_${{inj.hemi}}`;
      // Ipsilateral side label matches injection hemisphere
      const ipsiLabel  = inj.hemi === 'right' ? 'Ipsi (R)' : 'Ipsi (L)';
      const contraLabel = inj.hemi === 'right' ? 'Contra (L)' : 'Contra (R)';
      row.appendChild(buildFlatmapCol(inj.cells_ipsi,  getInjColor(inj), ipsiLabel,   false, baseName+'_ipsi_PFC'));
      row.appendChild(buildFlatmapCol(inj.cells_contra, getInjColor(inj), contraLabel, true,  baseName+'_contra_PFC'));
      card.appendChild(row);
      container.appendChild(card);
    }});

  }} else {{
    // Overlay mode: one pair of flatmaps (ipsi / contra) with all selected injections overlaid
    const fmH = Math.round(FM_HALF * DATA.svg_h / DATA.svg_w);
    const card = document.createElement('div');
    card.className = 'fm-card';
    card.innerHTML = `<div class="fm-title" style="margin-bottom:6px">Overlay — ${{selInjs.length}} injections selected</div>`;

    // Flex row so both hemispheres sit side by side
    const hemRow = document.createElement('div');
    hemRow.style.cssText = 'display:flex;flex-direction:row;gap:6px;align-items:flex-start;';
    card.appendChild(hemRow);

    for (const [side, mirror] of [['Ipsilateral', false], ['Contralateral', true]]) {{
      const col = document.createElement('div');
      col.style.cssText = 'flex:0 0 auto;';

      const lbl = document.createElement('div');
      lbl.className = 'fm-col-label'; lbl.textContent = side;
      col.appendChild(lbl);

      const wrap = document.createElement('div');
      wrap.className = 'fm-wrap';
      wrap.style.width  = FM_HALF + 'px';
      wrap.style.height = fmH + 'px';

      const img = document.createElement('img');
      img.src = mirror ? DATA.flatmap_url_r : DATA.flatmap_url;
      img.style.cssText = `width:${{FM_HALF}}px;height:${{fmH}}px;display:block;`;
      wrap.appendChild(img);

      const canvas = document.createElement('canvas');
      canvas.width = FM_HALF; canvas.height = fmH;
      canvas.style.cssText = `position:absolute;top:0;left:0;pointer-events:none;width:${{FM_HALF}}px;height:${{fmH}}px;`;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, FM_HALF, fmH);
      drawOverlayShuffled(ctx, getAxisFilteredInjs(selInjs), FM_HALF, fmH, mirror);
      wrap.appendChild(canvas);
      col.appendChild(wrap);

      // Download row
      const dlRow = document.createElement('div');
      dlRow.className = 'fm-dl-row';
      const layers = selInjs.map(inj => ({{
        cells: mirror ? inj.cells_contra : inj.cells_ipsi,
        color: getInjColor(inj),
      }}));
      const hemFile = `overlay_${{side.toLowerCase()}}_PFC`;
      const bSVG = document.createElement('button'); bSVG.className='btn-dl'; bSVG.textContent='↓ SVG';
      bSVG.onclick = () => downloadFlatmapSVG(layers, mirror, hemFile+'.svg');
      dlRow.appendChild(bSVG);
      const bPNG = document.createElement('button'); bPNG.className='btn-dl'; bPNG.textContent='↓ PNG';
      bPNG.onclick = () => downloadFlatmapPNG(img, canvas, mirror, hemFile+'.png');
      dlRow.appendChild(bPNG);
      col.appendChild(dlRow);
      hemRow.appendChild(col);
    }}

    // Legend
    const leg = document.createElement('div');
    leg.style.cssText = 'margin-top:8px;font-size:11px;';
    selInjs.forEach(inj => {{
      leg.innerHTML += `<span style="display:inline-flex;align-items:center;margin-right:10px;margin-bottom:3px;">
        <span style="width:9px;height:9px;border-radius:50%;background:${{getInjColor(inj)}};display:inline-block;margin-right:4px;"></span>
        ${{inj.animal}} ${{inj.hemi}} (${{inj.subregion}})
        ${{inj.pct_str!==null?' STR '+inj.pct_str.toFixed(1)+'%':''}}
        ${{inj.pct_cortex!==null?' CTX '+inj.pct_cortex.toFixed(1)+'%':''}}
      </span>`;
    }});
    card.appendChild(leg);
    container.appendChild(card);
  }}
}}

// ============================================================
// MAIN SELECTION CHANGE HANDLER
// ============================================================
function onSelectionChange() {{
  updatePlot();
  updateFlatmaps();
}}

// ============================================================
// INIT
// ============================================================
initPlot();
buildList();
buildOvlTable();
updateFlatmaps();
updateGradLegend();
updateAxisSlider();
</script>
</body>
</html>
"""


# ===========================================================================
# 4. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    
    # your existing parser.add_argument(...) lines here
    # for example:
    # parser.add_argument("--u-nrrd", default="anno_U_filled.nrrd")
    # parser.add_argument("--v-nrrd", default="anno_V_filled.nrrd")
    # parser.add_argument("--meshdir", default="allen_mesh")
    # parser.add_argument("--flatmap", default="flatmap_PFC_template.svg")
    # parser.add_argument("--workbook", default="Tracing_PT_IT_neurons_inUSE1.xlsx")
    # parser.add_argument("--out", default="injection_explorer.html")

    args = parser.parse_args()

    # ── Fixed folder containing support files ─────────────────────────────
    DATA_DIR = Path(r"L:\dmclab\Joana\Tracing\injection_explorer")

    args.u_nrrd = DATA_DIR / "anno_U_filled.nrrd"
    args.v_nrrd = DATA_DIR / "anno_V_filled.nrrd"
    args.meshdir = DATA_DIR / "allen_mesh"
    args.flatmap = DATA_DIR / "flatmap_PFC_template.svg"
    args.workbook = DATA_DIR / "Tracing_PT_IT_neurons_inUSE1.xlsx"
    args.out = DATA_DIR / "injection_explorer.html"

    print("Using files from:", DATA_DIR)
    print("U NRRD exists:", args.u_nrrd.exists())
    print("V NRRD exists:", args.v_nrrd.exists())    
  
    tracing_root = Path(r"L:\dmclab\Joana\Tracing")
    print("Using files from:", DATA_DIR)

    mesh_dir = args.meshdir
    flatmap_path = args.flatmap
    out_path = args.out

    # ── Load NRRDs ───────────────────────────────────────────────────────────
    print("Loading NRRD files …")
    u_data, _ = nrrd.read(str(args.u_nrrd))
    v_data, _ = nrrd.read(str(args.v_nrrd))

    # ── Read metadata ─────────────────────────────────────────────────────────
    print("Reading workbook metadata …")
    meta = read_metadata(Path(args.workbook))
    print(f"  {len(meta)} injection rows found")

    # ── Enumerate all injection CSVs ──────────────────────────────────────────
    # Find every <animal>/results/injection_site/green/<id>_injection_site.csv
    # under the known genotype subfolders.  Skipping other subdirs (Analysis, etc.)
    # avoids picking up unrelated CSV files with the same name pattern.
    all_csvs = []
    for folder_name in JOANA_FOLDER_TO_STRAIN:
        geno_dir = tracing_root / folder_name
        if geno_dir.is_dir():
            for csv in sorted(geno_dir.rglob("*_injection_site.csv")):
                # Only include CSVs that live in results/injection_site/green/
                # (exclude segmentation/injection_site/green/ per-slice CSVs)
                # Also require the stem prefix to be numeric (e.g. 986172),
                # skipping summary files like quantification_injection_site.csv
                animal_stem = csv.stem.replace("_injection_site", "")
                if (csv.parent.name == "green"
                        and csv.parent.parent.name == "injection_site"
                        and csv.parent.parent.parent.name == "results"
                        and animal_stem.isdigit()):
                    all_csvs.append(csv)

    print(f"\nFound {len(all_csvs)} injection CSVs")

    injections = []
    color_idx  = 0

    for csv_path in all_csvs:
        # Derive strain from the top-level genotype folder name
        # Path relative to tracing_root: <Genotype>/Cohort N/<animal_id>/results/injection_site/green/<id>.csv
        parts = csv_path.relative_to(tracing_root).parts
        folder_name = parts[0]   # e.g. "Fezf2"
        strain = JOANA_FOLDER_TO_STRAIN.get(folder_name, folder_name.lower())
        animal_id = csv_path.stem.replace("_injection_site", "")

        for hemi in ("left", "right"):
            key = (strain, animal_id, hemi)
            if key not in meta:
                # Skip hemispheres not recorded in the workbook.
                # Creating a spurious entry for the non-injected hemisphere
                # causes ipsi/contra to appear swapped, because the cells CSV
                # contains cells from the real injection on the OTHER side.
                continue
            inj_meta = meta[key]

            data = build_injection(
                animal_id, hemi, inj_meta,
                csv_path,
                u_data, v_data,
                color_idx,
            )
            if data is not None:
                injections.append(data)
                color_idx += 1

    print(f"\nBuilt {len(injections)} injection entries")

    # ── Atlas meshes ──────────────────────────────────────────────────────────
    print("Loading atlas meshes …")
    atlas_traces = []
    for sid, (name, color, opacity) in MESH_DEFS.items():
        obj = mesh_dir / f"{sid}.obj"
        if obj.exists():
            atlas_traces.append(load_atlas_trace(obj, name, color, opacity))
            print(f"  loaded {obj.name}")
        else:
            print(f"  [WARN] {obj} not found")

    # ── Pre-compute pairwise overlaps ─────────────────────────────────────────
    print("Computing pairwise overlaps …")
    pairs = compute_all_overlaps(injections)
    print(f"  {len(pairs)} non-zero overlap pairs")

    # ── Generate HTML ─────────────────────────────────────────────────────────
    print("Generating HTML …")
    html = build_html(
        injections, atlas_traces, pairs,
        Path(args.flatmap),
        get_plotlyjs_version(),
    )
    out_path.write_text(html, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")
    print(f"  {len(injections)} injections · {len(atlas_traces)} atlas meshes")


if __name__ == "__main__":
    main()
