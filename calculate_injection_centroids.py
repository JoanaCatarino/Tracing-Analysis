"""Batch injection centroids, section interpolation and interactive 3D inspection.

One injection = overview animal/channel/side; masks always read from green.
All selected-side segmentation points are used, including non-striatal spillover.
The >=80% striatum criterion is taken from the overview, not recomputed here.

Reconstruction: fit a common section orientation in registered Allen space,
project masks onto parallel planes, then linearly interpolate their signed
2D distance fields at the measured section positions. First/last masks extend
25 um beyond their sampled centre (half a retained 50 um section). No Gaussian
smoothing or largest-component selection is used. Oversized gaps and excessive
nonplanarity are rejected. Reconstructed volumes are estimates, not observations.

All display/clustering coordinates use a common LEFT hemisphere by default.
Atlas coordinates are 10 um units with ML midline 570, matching the supplied
explorer. ml_mm is checked against this convention before reconstruction.
Original sampled centroids remain in the output. The reconstruction centroid is
the mean of occupied equal-volume grid cells, NOT the mean of surface vertices.

Run in Spyder. Dependencies: numpy pandas openpyxl scipy scikit-image plotly.
Set OVERVIEW_FILE first. Inputs are read-only; outputs are rebuilt each run.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import json
from functools import lru_cache
from urllib.request import urlopen
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes
import plotly.graph_objects as go

BASE_DIR = Path(r"L:\dmclab\Joana\Tracing")
OUTPUT_DIR = BASE_DIR / "Analysis" / "injection_centroids"
OVERVIEW_FILE = Path(r"L:\dmclab\Joana\Tracing\tracing_overview.xlsx")
SLICE_THICKNESS_UM = 50.0
KEEP_EVERY_NTH_SLICE = 2
SECTION_SPACING_UM = SLICE_THICKNESS_UM * KEEP_EVERY_NTH_SLICE
# Preserve the inclusion criterion used in the previous analyses.
# Set to None to calculate all overview injections regardless of % Striatum.
MIN_STRIATUM_PERCENT = 80.0
STRAIN_FOLDERS = {}  # e.g. {"Fmr1-Tlx3": "Fmr1_Tlx3"}
POINT_AREA_COLUMN = None  # Optional positive pixel area per exported row.
# Export selected points for a later 3D reconstruction; can produce large files.
SAVE_SELECTED_POINTS = False
# Reconstructed geometry and HTML are saved under OUTPUT_DIR / reconstruction_3d.
ATLAS_VOXEL_UM = 10.0
ML_MIDLINE_COORD = 570.0
TARGET_HEMISPHERE = "left"
RECON_GRID_UM = 25.0
MAX_SECTION_GAP_UM = 2 * SECTION_SPACING_UM
END_EXTENSION_UM = SLICE_THICKNESS_UM / 2
MAX_PLANE_RESIDUAL_95_UM = RECON_GRID_UM
MAX_GRID_VOXELS = 20_000_000
MAX_VIEW_POINTS = 6000  # Display only; all points enter calculations.
# Reuse your explorer's 997.obj, 672.obj and 56.obj by pointing this directory
# to its allen_mesh folder. Missing files are downloaded once from Allen.
ATLAS_MESH_DIR = BASE_DIR / "Analysis" / "allen_mesh"
DOWNLOAD_MISSING_ATLAS_MESHES = True
ALLEN_MESH_BASE_URL = (
    "https://download.alleninstitute.org/informatics-archive/current-release/"
    "mouse_ccf/annotation/ccf_2017/structure_meshes"
)
ATLAS_STRUCTURES = [
    (997, "Allen brain outline", "#B8B8B8", .06),
    (672, "CP — Caudoputamen", "#55A9D5", .22),
    (56, "ACB — Nucleus accumbens", "#E6A14A", .35),
]
# Every successful reconstruction gets NPZ geometry and an offline HTML viewer.
# All centroids are recalculated on each run; summary outputs are replaced.
AXES = ["ap_coords", "ml_coords", "dv_coords"]

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


def component(value):
    value = str(value)
    if not value or value in {".", ".."} or re.search(r'[<>:"/\\|?*]', value):
        raise ValueError(f"Invalid path component: {value!r}")
    return value


def injection_path(job):
    """All injection masks are stored under green, including cy3 injections."""
    strain = component(STRAIN_FOLDERS.get(job["strain"], job["strain"]))
    return (BASE_DIR / strain / f"Cohort {component(job['cohort'])}"
            / component(job["animal_id"]) / "results" / "injection_site"
            / "green" / f"{component(job['animal_id'])}_injection_site.csv")


def compute_centroid(raw, injection_side):
    """Return centroid/QC dict, section table and selected points.

    Repeated atlas coordinates are retained: different original mask pixels
    may map to the same rounded atlas location. Their count is reported.
    No anatomical-region filtering is applied to the selected-side mask.
    """
    side = str(injection_side).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError("injection side must be left or right for this injection row.")
    required = set(AXES + ["ml_mm", "section_name"])
    if required - set(raw.columns):
        raise ValueError(f"Missing segmentation columns: {sorted(required - set(raw.columns))}")
    if not np.isfinite(SECTION_SPACING_UM) or SECTION_SPACING_UM <= 0:
        raise ValueError("SECTION_SPACING_UM must be positive.")
    ml = pd.to_numeric(raw.ml_mm, errors="coerce")
    if not np.isfinite(ml.to_numpy(dtype=float)).all():
        raise ValueError("Missing/invalid ml_mm; cannot determine which hemisphere to keep.")
    mask = ml.gt(0) if side == "right" else ml.lt(0)
    selected = raw.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"No segmentation points on requested {side} side.")
    selected["ml_mm"] = ml.loc[mask]
    for axis in AXES:
        selected[axis] = pd.to_numeric(selected[axis], errors="coerce")
        if not np.isfinite(selected[axis].to_numpy(dtype=float)).all():
            raise ValueError(f"Missing/invalid {axis} on selected side.")
    if selected.section_name.isna().any() or selected.section_name.astype(str).str.strip().eq("").any():
        raise ValueError("Selected-side points have missing section_name.")
    selected["section_name"] = selected.section_name.astype(str).str.strip()
    if POINT_AREA_COLUMN is None:
        area = np.ones(len(selected))
        weight_basis = "equal_area_points_x_100um" if SECTION_SPACING_UM == 100 else "equal_area_points_x_spacing"
    else:
        if POINT_AREA_COLUMN not in selected:
            raise ValueError(f"Missing point area column: {POINT_AREA_COLUMN}")
        area = pd.to_numeric(selected[POINT_AREA_COLUMN], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(area).all() or (area <= 0).any():
            raise ValueError("Point areas must be finite and positive.")
        weight_basis = f"{POINT_AREA_COLUMN}_x_spacing"
    selected["centroid_weight"] = area * SECTION_SPACING_UM
    xyz = selected[AXES].to_numpy(dtype=float)
    weights = selected.centroid_weight.to_numpy()
    centroid = np.average(xyz, axis=0, weights=weights)
    qc = {
        "injection_side": side, "input_points": len(raw),
        "selected_points": len(selected),
        "opposite_side_points": int((ml.lt(0) if side == "right" else ml.gt(0)).sum()),
        "midline_points_excluded": int(ml.eq(0).sum()),
        "n_sections": selected.section_name.nunique(),
        "section_spacing_um": SECTION_SPACING_UM,
        "slice_thickness_um": SLICE_THICKNESS_UM,
        "centroid_weight_basis": weight_basis,
        "duplicate_section_atlas_xyz_rows_retained": int(selected.duplicated(["section_name"] + AXES).sum()),
    }
    for i, axis in enumerate(AXES):
        qc["centroid_" + axis] = float(centroid[i])
        qc["min_" + axis] = float(xyz[:, i].min())
        qc["max_" + axis] = float(xyz[:, i].max())
    # Optional signed physical-coordinate centroids, without assuming an atlas scale.
    for axis in ["ap_mm", "ml_mm", "dv_mm"]:
        if axis in selected:
            values = pd.to_numeric(selected[axis], errors="coerce").to_numpy(dtype=float)
            qc["centroid_" + axis] = float(np.average(values, weights=weights)) if np.isfinite(values).all() else np.nan
    if "acronym" in selected:
        qc["root_points_retained"] = int(selected.acronym.astype(str).str.strip().eq("root").sum())
    section_rows = []
    for name, section in selected.groupby("section_name", sort=False):
        w = section.centroid_weight.to_numpy()
        center = np.average(section[AXES].to_numpy(dtype=float), axis=0, weights=w)
        row = {"section_name": name, "n_points": len(section),
               "section_weight": float(w.sum()), "fraction_of_total_weight": float(w.sum() / weights.sum()),
               "section_spacing_um": SECTION_SPACING_UM}
        for i, axis in enumerate(AXES):
            row["centroid_" + axis] = float(center[i])
            row["min_" + axis] = float(section[axis].min())
            row["max_" + axis] = float(section[axis].max())
        section_rows.append(row)
    sections = pd.DataFrame(section_rows).sort_values("centroid_ap_coords")
    # Section area weighting must give the same center as weighting all pixels.
    reconstructed = np.average(sections[["centroid_" + axis for axis in AXES]],
                               axis=0, weights=sections.section_weight)
    if not np.allclose(centroid, reconstructed):
        raise ValueError("Point and section-weighted centroids disagree.")
    return qc, sections, selected


def common_coordinates(selected, side):
    if TARGET_HEMISPHERE not in {"left", "right"}:
        raise ValueError("TARGET_HEMISPHERE must be left or right.")
    xyz = selected[AXES].to_numpy(dtype=float) * ATLAS_VOXEL_UM
    expected_mm = (ML_MIDLINE_COORD - selected.ml_coords.to_numpy()) * ATLAS_VOXEL_UM / 1000
    if not np.allclose(selected.ml_mm.to_numpy(), expected_mm, atol=.025, rtol=0):
        raise ValueError("ml_mm disagrees with the configured atlas scale/midline. Check ATLAS_VOXEL_UM and ML_MIDLINE_COORD.")
    if side != TARGET_HEMISPHERE:
        xyz[:, 1] = 2 * ML_MIDLINE_COORD * ATLAS_VOXEL_UM - xyz[:, 1]
    return xyz


def reconstruct_injection(selected, side):
    """Interpolate signed section-mask distances in a fitted plane basis."""
    if RECON_GRID_UM <= 0 or ATLAS_VOXEL_UM <= 0:
        raise ValueError("Grid spacing and atlas scale must be positive.")
    xyz = common_coordinates(selected, side)
    groups = selected.groupby("section_name", sort=False).indices
    if len(groups) < 2:
        raise ValueError("At least two segmented sections are required for interpolation.")
    centered = np.vstack([xyz[idx] - xyz[idx].mean(axis=0) for idx in groups.values()])
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    normal = eigenvectors[:, 0]
    if normal[0] < 0:
        normal = -normal
    if abs(normal[0]) < .7 or eigenvalues[1] <= 1e-9:
        raise ValueError("Cannot identify a stable near-coronal common section plane.")
    u = np.array([0., 1., 0.])
    u -= normal * np.dot(u, normal)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    basis = np.column_stack([normal, u, v])
    local = xyz @ basis
    residual = np.concatenate([local[idx, 0] - local[idx, 0].mean() for idx in groups.values()])
    residual95 = float(np.percentile(np.abs(residual), 95))
    if residual95 > MAX_PLANE_RESIDUAL_95_UM:
        raise ValueError(f"Registered sections deviate from parallel planes by {residual95:.1f} um (95th percentile); inspect registration before interpolation.")
    ordered = sorted([(float(local[idx, 0].mean()), name, idx) for name, idx in groups.items()])
    positions = np.array([item[0] for item in ordered])
    gaps = np.diff(positions)
    if gaps.min() < RECON_GRID_UM / 2:
        raise ValueError("Section positions overlap or are too close; check section labels/registration.")
    if gaps.max() > MAX_SECTION_GAP_UM:
        raise ValueError(f"Section gap {gaps.max():.1f} um exceeds {MAX_SECTION_GAP_UM:g} um; missing sections are not silently interpolated.")
    step = RECON_GRID_UM
    lo = np.floor(local.min(axis=0) / step) * step - 2 * step
    hi = np.ceil(local.max(axis=0) / step) * step + 2 * step
    lo[0] = np.floor((positions[0] - END_EXTENSION_UM) / step) * step - 2 * step
    hi[0] = np.ceil((positions[-1] + END_EXTENSION_UM) / step) * step + 2 * step
    shape = (np.ceil((hi - lo) / step).astype(int) + 1)
    if int(np.prod(shape)) > MAX_GRID_VOXELS:
        raise ValueError("Reconstruction grid is too large; inspect outliers or increase RECON_GRID_UM.")
    fields = []
    for t, name, idx in ordered:
        mask = np.zeros(tuple(shape[1:]), dtype=bool)
        indices = np.rint((local[idx, 1:] - lo[1:]) / step).astype(int)
        mask[indices[:, 0], indices[:, 1]] = True
        # Signed distance at pixel centres, with a half-pixel boundary correction.
        inside = distance_transform_edt(mask)
        outside = distance_transform_edt(~mask)
        fields.append((np.where(mask, inside - .5, .5 - outside) * step).astype(np.float32))
    volume = np.zeros(tuple(shape), dtype=bool)
    for k in range(shape[0]):
        t = lo[0] + k * step
        if t < positions[0] - END_EXTENSION_UM or t > positions[-1] + END_EXTENSION_UM:
            continue
        upper = int(np.searchsorted(positions, t))
        if upper == 0:
            field = fields[0]
        elif upper == len(positions):
            field = fields[-1]
        else:
            alpha = (t - positions[upper - 1]) / (positions[upper] - positions[upper - 1])
            field = (1 - alpha) * fields[upper - 1] + alpha * fields[upper]
        volume[k] = field > 0
    occupied = np.argwhere(volume)
    if len(occupied) == 0:
        raise ValueError("Interpolated volume is empty.")
    reconstructed = (occupied.mean(axis=0) * step + lo) @ basis.T
    sampled = np.average(xyz, axis=0, weights=selected.centroid_weight.to_numpy())
    verts, faces, _, _ = marching_cubes(volume.astype(np.float32), level=.5, spacing=(step,) * 3)
    verts = (verts + lo) @ basis.T
    shift = float(np.linalg.norm(reconstructed - sampled))
    stats = {
        "reconstruction_status": "OK", "common_hemisphere": TARGET_HEMISPHERE,
        "reconstruction_method": "linear_signed_distance_parallel_sections",
        "reconstruction_grid_um": step, "end_extension_um": END_EXTENSION_UM,
        "atlas_voxel_um": ATLAS_VOXEL_UM, "ml_midline_atlas_coord": ML_MIDLINE_COORD,
        "section_gap_min_um": float(gaps.min()), "section_gap_max_um": float(gaps.max()),
        "section_plane_residual95_um": residual95,
        "estimated_reconstructed_volume_mm3": len(occupied) * step ** 3 / 1e9,
        "sampled_to_reconstructed_shift_um": shift,
    }
    for i, axis in enumerate(["ap", "ml", "dv"]):
        stats[f"sampled_common_{axis}_um"] = float(sampled[i])
        stats[f"sampled_common_{axis}_coords"] = float(sampled[i] / ATLAS_VOXEL_UM)
        stats[f"reconstructed_common_{axis}_um"] = float(reconstructed[i])
        stats[f"reconstructed_common_{axis}_coords"] = float(reconstructed[i] / ATLAS_VOXEL_UM)
    rng = np.random.default_rng(42)
    show_idx = rng.choice(len(xyz), min(MAX_VIEW_POINTS, len(xyz)), replace=False)
    geometry = dict(vertices_um=verts, faces=faces, sampled_display_points_um=xyz[show_idx],
                    sampled_centroid_um=sampled, reconstructed_centroid_um=reconstructed,
                    occupied_grid_indices=occupied, grid_origin_local_um=lo,
                    grid_step_um=np.array(step), local_to_atlas_basis=basis,
                    section_positions_local_um=positions)
    return stats, geometry


def read_allen_obj(path):
    """Read OBJ vertices/faces; Allen mesh axes AP,DV,ML in um -> AP,ML,DV."""
    vertices, triangles = [], []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f":
                indices = []
                for token in fields[1:]:
                    index = int(token.split("/")[0])
                    if index == 0:
                        raise ValueError(f"Invalid zero OBJ index: {path}")
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                for i in range(1, len(indices) - 1):
                    triangles.append([indices[0], indices[i], indices[i + 1]])
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(triangles):
        raise ValueError(f"No usable surface in Allen OBJ file: {path}")
    if not np.isfinite(vertices).all() or triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError(f"Invalid vertices/faces in Allen OBJ file: {path}")
    # Swapping axes changes handedness; reverse winding to preserve normals.
    return vertices[:, [0, 2, 1]], triangles[:, [0, 2, 1]]


@lru_cache(maxsize=1)
def load_atlas_meshes():
    """Download once, then reuse local meshes and parsed arrays for all viewers."""
    ATLAS_MESH_DIR.mkdir(parents=True, exist_ok=True)
    atlas = []
    for structure_id, name, color, opacity in ATLAS_STRUCTURES:
        path = ATLAS_MESH_DIR / f"{structure_id}.obj"
        if not path.is_file():
            if not DOWNLOAD_MISSING_ATLAS_MESHES:
                raise FileNotFoundError(f"Missing Allen mesh: {path}. Set ATLAS_MESH_DIR to your existing allen_mesh folder.")
            temporary = path.with_suffix(".obj.downloading")
            print(f"Downloading Allen mesh: {structure_id}.obj")
            try:
                with urlopen(f"{ALLEN_MESH_BASE_URL}/{structure_id}.obj", timeout=30) as response:
                    with temporary.open("wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                read_allen_obj(temporary)  # Validate before replacing the cached file.
                temporary.replace(path)
            except Exception as exc:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Could not obtain {path.name}. Put the Allen OBJ files in {ATLAS_MESH_DIR}, "
                                   "or set ATLAS_MESH_DIR to your explorer's allen_mesh folder.") from exc
        vertices, faces = read_allen_obj(path)
        atlas.append(dict(name=name, color=color, opacity=opacity, vertices=vertices, faces=faces))
    return atlas


def write_viewer(entries, path, title="Injection reconstructions"):
    """Offline viewer; legend controls each injection, menus switch geometry."""
    fig = go.Figure()
    for atlas in load_atlas_meshes():
        verts, faces = atlas["vertices"], atlas["faces"]
        fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], opacity=atlas["opacity"], color=atlas["color"],
            name=atlas["name"], legendgroup=atlas["name"], showlegend=True, hoverinfo="name"))
    n_atlas = len(fig.data)
    palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#b07aa1", "#edc948", "#ff9da7"]
    for j, entry in enumerate(entries):
        geo, name = entry["geometry"], entry["injection_id"]
        color = palette[j % len(palette)]
        verts, faces = geo["vertices_um"], geo["faces"]
        fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], opacity=.3, color=color,
            name=name, legendgroup=name, showlegend=False, hoverinfo="name"))
        points = geo["sampled_display_points_um"]
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode="markers",
            marker=dict(size=1.5, color=color, opacity=.45), name=name + " sampled points",
            legendgroup=name, showlegend=False, visible=False))
        for key, symbol, label in [("sampled_centroid_um", "diamond", "Sampled centroid"),
                                    ("reconstructed_centroid_um", "circle", "Reconstructed centroid")]:
            center = geo[key]
            fig.add_trace(go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]], mode="markers",
                marker=dict(size=6, color=color, symbol=symbol, line=dict(color="black", width=2)),
                name=name if key == "reconstructed_centroid_um" else name + " " + label,
                legendgroup=name, showlegend=(key == "reconstructed_centroid_um"),
                hovertemplate=name + "<br>" + label + "<br>AP %{x:.1f} µm<br>ML %{y:.1f} µm<br>DV %{z:.1f} µm<extra></extra>"))
        ends = np.vstack([geo["sampled_centroid_um"], geo["reconstructed_centroid_um"]])
        fig.add_trace(go.Scatter3d(x=ends[:, 0], y=ends[:, 1], z=ends[:, 2], mode="lines",
            line=dict(color=color, width=3), legendgroup=name, showlegend=False, hoverinfo="skip"))
    buttons = []
    for label, trace_types in [("Surfaces + centroids", {0, 2, 3, 4}),
                                ("Sampled points + centroids", {1, 2, 3, 4}),
                                ("Everything", {0, 1, 2, 3, 4}), ("Centroids only", {2, 3, 4})]:
        injection_indices = list(range(n_atlas, len(fig.data)))
        buttons.append(dict(label=label, method="restyle", args=[
            {"visible": [(i - n_atlas) % 5 in trace_types for i in injection_indices]}, injection_indices]))
    fig.update_layout(title=title + f" — common {TARGET_HEMISPHERE} hemisphere", height=850,
        scene=dict(xaxis_title="Allen AP (µm)", yaxis_title="Allen ML (µm)", zaxis_title="Allen DV (µm)",
                   aspectmode="data", zaxis=dict(autorange="reversed")),
        legend=dict(groupclick="togglegroup"), margin=dict(l=0, r=0, t=115, b=60),
        updatemenus=[dict(buttons=buttons, direction="right", type="buttons", x=0, y=1.08)],
        annotations=[dict(text="Diamond: sampled centroid · Circle: reconstructed centroid · Click legend to hide/isolate injections.<br>"
            "Allen CCF: grey brain, blue CP, orange ACB. Toggle atlas structures in the legend.<br>"
            "Reconstruction is estimated. View buttons reset injection visibility; atlas visibility is preserved.",
            x=.5, y=-.05, xref="paper", yref="paper", showarrow=False)])
    if not entries:
        fig.add_annotation(text="No successful reconstructions in this run. See processing log.", x=.5, y=.5, xref="paper", yref="paper", showarrow=False)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)


def main():
    if not BASE_DIR.is_dir():
        raise FileNotFoundError(f"Tracing folder not accessible: {BASE_DIR}")
    jobs = read_injection_overview(OVERVIEW_FILE)
    if MIN_STRIATUM_PERCENT is not None and not 0 <= MIN_STRIATUM_PERCENT <= 100:
        raise ValueError("MIN_STRIATUM_PERCENT must be 0–100 or None.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results, section_tables, audit, viewer_entries = [], [], [], []
    recon_dir = OUTPUT_DIR / "reconstruction_3d"
    recon_dir.mkdir(parents=True, exist_ok=True)
    load_atlas_meshes()  # Check/download atlas once before processing injections.
    for index, job in enumerate(jobs.to_dict("records"), 1):
        injection_id = "_".join(str(job[k]) for k in ["strain", "cohort", "animal_id", "channel"])
        entry = dict(job, injection_id=injection_id, source_csv="", status="", message="")
        try:
            if job["cells"] != "y":
                entry.update(status="EXCLUDED", message="Cells not marked y in overview.")
            if not entry["status"] and MIN_STRIATUM_PERCENT is not None:
                percent = job["striatum_percent"]
                if pd.isna(percent) or not np.isfinite(percent) or not 0 <= percent <= 100:
                    entry.update(status="EXCLUDED", message="Missing/invalid % Striatum.")
                elif percent < MIN_STRIATUM_PERCENT:
                    entry.update(status="EXCLUDED", message=f"% Striatum < {MIN_STRIATUM_PERCENT:g}.")
            if not entry["status"]:
                if job["injection_side"] not in {"left", "right"}:
                    raise ValueError("Overview injection side is missing/invalid; expected left or right.")
                source = injection_path(job)
                entry["source_csv"] = str(source)
                if not source.is_file():
                    raise FileNotFoundError(f"Segmentation CSV not found: {source}")
                # All outputs are separate from the source segmentation tree.
                if OUTPUT_DIR.resolve() == source.parent.resolve() or OUTPUT_DIR.resolve() in source.parents:
                    raise ValueError("OUTPUT_DIR must not contain source segmentation CSVs.")
                qc, sections, selected = compute_centroid(pd.read_csv(source), job["injection_side"])
                # A valid sampled centroid does not depend on interpolation succeeding.
                common_xyz = common_coordinates(selected, job["injection_side"])
                sampled_center = np.average(common_xyz, axis=0, weights=selected.centroid_weight)
                qc.update(common_hemisphere=TARGET_HEMISPHERE, atlas_voxel_um=ATLAS_VOXEL_UM,
                          ml_midline_atlas_coord=ML_MIDLINE_COORD)
                for axis, value in zip(["ap", "ml", "dv"], sampled_center):
                    qc[f"sampled_common_{axis}_um"] = float(value)
                    qc[f"sampled_common_{axis}_coords"] = float(value / ATLAS_VOXEL_UM)
                try:
                    recon_stats, geometry = reconstruct_injection(selected, job["injection_side"])
                    np.savez_compressed(recon_dir / f"{injection_id}.npz", **geometry)
                    display_keys = ["vertices_um", "faces", "sampled_display_points_um",
                                    "sampled_centroid_um", "reconstructed_centroid_um"]
                    entry3d = dict(injection_id=injection_id,
                                   geometry={key: geometry[key] for key in display_keys})
                    write_viewer([entry3d], recon_dir / f"{injection_id}.html", injection_id)
                    viewer_entries.append(entry3d)
                    qc.update(recon_stats)
                except Exception as exc:
                    qc.update(reconstruction_status="ERROR", reconstruction_error=str(exc))
                    print(f"  Reconstruction failed; sampled centroid retained: {exc}")
                results.append(dict(job, injection_id=injection_id, source_csv=str(source), **{
                    key: value for key, value in qc.items() if key != "injection_side"}))
                for key in ["strain", "cohort", "animal_id", "channel", "injection_side"]:
                    sections[key] = job[key]
                sections["injection_id"] = injection_id
                sections["source_csv"] = str(source)
                section_tables.append(sections)
                if SAVE_SELECTED_POINTS:
                    point_dir = OUTPUT_DIR / "selected_points"
                    point_dir.mkdir(parents=True, exist_ok=True)
                    selected.to_csv(point_dir / f"{injection_id}_{job['injection_side']}_points.csv", index=False)
                entry.update(status="OK", message="", **qc)
        except Exception as exc:
            entry.update(status="ERROR", message=f"{type(exc).__name__}: {exc}")
        audit.append(entry)
        print(f"[{index}/{len(jobs)}] {injection_id}: {entry['status']} {entry['message']}")
    # Always rewrite summaries, even if this run has no successful injections.
    result_columns = list(jobs.columns) + ["injection_id", "source_csv"]
    output = pd.DataFrame(results) if results else pd.DataFrame(columns=result_columns + ["centroid_" + axis for axis in AXES])
    output.to_csv(OUTPUT_DIR / "injection_centroids.csv", index=False)
    clustering_columns = ["injection_id", "strain", "cohort", "animal_id", "channel", "injection_side",
        "common_hemisphere", "sampled_common_ap_um", "sampled_common_ml_um", "sampled_common_dv_um",
        "reconstructed_common_ap_um", "reconstructed_common_ml_um", "reconstructed_common_dv_um",
        "sampled_to_reconstructed_shift_um", "estimated_reconstructed_volume_mm3"]
    good = output.loc[output.get("reconstruction_status", pd.Series(index=output.index, dtype=str)).eq("OK")]
    good.reindex(columns=clustering_columns).to_csv(OUTPUT_DIR / "reconstructed_centroids_for_clustering.csv", index=False)
    write_viewer(viewer_entries, recon_dir / "all_injections_3d.html")
    sections_out = (pd.concat(section_tables, ignore_index=True) if section_tables else
                    pd.DataFrame(columns=["injection_id", "section_name", "n_points"] + ["centroid_" + axis for axis in AXES]))
    sections_out.to_csv(OUTPUT_DIR / "injection_centroids_by_section.csv", index=False)
    audit_out = pd.DataFrame(audit)
    audit_out.to_csv(OUTPUT_DIR / "injection_centroid_processing_log.csv", index=False)
    # Preserve every eligible injection, including errors with missing coordinates.
    # The clustering script must stop if any eligible coordinate is missing.
    eligible = audit_out.loc[audit_out.status.ne("EXCLUDED")].copy()
    eligible.to_csv(OUTPUT_DIR / "centroids_for_clustering.csv", index=False)
    sampled_cols = [f"sampled_common_{a}_um" for a in ["ap", "ml", "dv"]]
    valid = eligible.reindex(columns=sampled_cols).notna().all(axis=1)
    print(f"Clustering eligibility: {len(eligible)} injections; {valid.sum()} sampled centroids; "
          f"{(~valid).sum()} unresolved. Reconstruction success is not an inclusion filter.")
    print(audit_out.status.value_counts().to_string())
    print(f"DONE! Centroids calculated for {len(output)} injections.\nSaved in: {OUTPUT_DIR}")
    return output


if __name__ == "__main__":
    main()
