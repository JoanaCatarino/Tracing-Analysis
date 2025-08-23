# -*- coding: utf-8 -*-
"""
Created on Tue May 20 10:51:34 2025

@author: JoanaCatarino
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

    # All cells divided by injection region

# Load dataset
df = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/fm_it.csv') 

# Define regions of interest
regions_of_interest = ["FRP", "MOs", "ACAd", "ACAv", "PL", "ILA", "ORBm", "ORBvl", "ORBl", "AId", "AIv"]

# Extract base region from acronym (e.g., "MOs5" → "MOs")
df["main_region"] = df["acronym"].str.extract(r"^(" + "|".join(regions_of_interest) + ")")

# Clean and filter data
filtered_df = df[df["main_region"].isin(regions_of_interest)].copy()
filtered_df["ap_mm"] = pd.to_numeric(filtered_df["ap_mm"], errors="coerce")
filtered_df = filtered_df.dropna(subset=["ap_mm"])

# Create 0.5 mm AP bins starting from a clean rounded base
bin_width = 0.5
bin_start = np.floor(filtered_df["ap_mm"].min())
bin_end = np.ceil(filtered_df["ap_mm"].max()) #+ bin_width
ap_bins = pd.interval_range(start=bin_start, end=bin_end, freq=bin_width)

filtered_df["ap_bin"] = pd.cut(filtered_df["ap_mm"], bins=ap_bins)
filtered_df["ap_mid"] = filtered_df["ap_bin"].apply(lambda x: x.mid)

# Choose colormap
cube_cmap = sns.cubehelix_palette(as_cmap=True)
cube_array = cube_cmap(np.linspace(0, 1, 256))

# Insert a light gray at the start for 0 values
gray_start = np.array([[0.98, 0.94, 0.97, 1]])  # RGBA for light gray
custom_colors = np.vstack([gray_start, cube_array[1:]])
custom_cmap = ListedColormap(custom_colors)

# Get unique injection regions
unique_inj_regions = filtered_df["inj_region"].dropna().unique()

# Plot individual heatmaps for each injection region
for inj in unique_inj_regions:
    sub_df = filtered_df[filtered_df["inj_region"] == inj]

    # Group and pivot for heatmap
    grouped = (
        sub_df.groupby(["main_region", "ap_mid"])
        .size()
        .reset_index(name="cell_count")
    )
    heatmap_data = grouped.pivot_table(index="main_region", columns="ap_mid", values="cell_count", fill_value=0)
    heatmap_data = heatmap_data.reindex(index=regions_of_interest, fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(
        heatmap_data,
        cmap=custom_cmap,
        linewidths=2,
        linecolor="white",
        cbar_kws={'label': 'Cell Count'},
        square=False,
        ax=ax
    )
    ax.set_aspect(0.5)  # compress cells along x-axis
    ax.set_title(f"Injection Region: {inj}")
    ax.set_ylabel('')
    ax.set_xlabel("AP Coordinate (mm)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.invert_xaxis() 
    
    fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/AllCells_by_injection_heatmap_{inj}.png", dpi=500)
    fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/AllCells_by_injection_heatmap_{inj}.pdf", dpi=500)
    fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/AllCells_by_injection_heatmap_{inj}.svg", dpi=500)  
    
#%%

    # All cells divided by ipsi and contra 
    
# Load dataset
df = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/fm_it.csv') 

# Define regions of interest
regions_of_interest = ["FRP", "MOs", "ACAd", "ACAv", "PL", "ILA", "ORBm", "ORBvl", "ORBl", "AId", "AIv"]

# Extract base region from acronym (e.g., "MOs5" → "MOs")
df["main_region"] = df["acronym"].str.extract(r"^(" + "|".join(regions_of_interest) + ")")

# Clean and filter data
filtered_df = df[df["main_region"].isin(regions_of_interest)].copy()
filtered_df["ap_mm"] = pd.to_numeric(filtered_df["ap_mm"], errors="coerce")
filtered_df = filtered_df.dropna(subset=["ap_mm"])

# Create 0.5 mm AP bins starting from a clean rounded base
bin_width = 0.5
bin_start = np.floor(filtered_df["ap_mm"].min())
bin_end = np.ceil(filtered_df["ap_mm"].max()) #+ bin_width
ap_bins = pd.interval_range(start=bin_start, end=bin_end, freq=bin_width)

filtered_df["ap_bin"] = pd.cut(filtered_df["ap_mm"], bins=ap_bins)
filtered_df["ap_mid"] = filtered_df["ap_bin"].apply(lambda x: x.mid)

# Choose colormap
cube_cmap = sns.cubehelix_palette(as_cmap=True)
cube_array = cube_cmap(np.linspace(0, 1, 256))

# Insert a light gray at the start for 0 values
gray_start = np.array([[0.98, 0.94, 0.97, 1]])  # RGBA for light gray
custom_colors = np.vstack([gray_start, cube_array[1:]])
custom_cmap = ListedColormap(custom_colors)

# Get unique injection regions
unique_inj_regions = filtered_df["inj_region"].dropna().unique()    

# Plot heatmaps: one figure per inj_region with 2 rows (ipsi & contra)
for inj in unique_inj_regions:
    inj_df = filtered_df[filtered_df["inj_region"] == inj]
    unique_projections = inj_df["projection"].dropna().unique()

    # Only proceed if there are at least 2 projections (or at least 1)
    fig, axes = plt.subplots(nrows=len(unique_projections), figsize=(16, 6 * len(unique_projections)))
    
    if len(unique_projections) == 1:
        axes = [axes]  # make it iterable

    for i, proj in enumerate(unique_projections):
        sub_df = inj_df[inj_df["projection"] == proj]

        # Group and pivot
        grouped = (
            sub_df.groupby(["main_region", "ap_mid"])
            .size()
            .reset_index(name="cell_count")
        )
        heatmap_data = grouped.pivot_table(index="main_region", columns="ap_mid", values="cell_count", fill_value=0)
        heatmap_data = heatmap_data.reindex(index=regions_of_interest, fill_value=0)

        if heatmap_data.empty:
            continue

        sns.heatmap(
            heatmap_data,
            cmap=custom_cmap,
            linewidths=2,
            linecolor="white",
            cbar_kws={'label': 'Cell Count'},
            square=False,
            ax=axes[i]
        )
        axes[i].set_aspect(0.5)
        axes[i].set_title(f"{inj} – Projection: {proj}")
        axes[i].set_ylabel('')
        axes[i].set_xlabel("AP Coordinate (mm)")
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
        axes[i].invert_xaxis()
        plt.subplots_adjust(hspace=0.4)

        fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/IpsiContra_by_injection_heatmap_{inj}.png", dpi=500)
        fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/IpsiContra_by_injection_heatmap_{inj}.pdf", dpi=500)
        fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/IpsiContra_by_injection_heatmap_{inj}.svg", dpi=500)  


#%%

    # All cells divided by AP slice of injection site

df = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/fm_it.csv') 

# Define regions of interest
regions_of_interest = ["FRP", "MOs", "ACAd", "ACAv", "PL", "ILA", "ORBm", "ORBvl", "ORBl", "AId", "AIv"]

# Extract base region from acronym (e.g., "MOs5" → "MOs")
df["main_region"] = df["acronym"].str.extract(r"^(" + "|".join(regions_of_interest) + ")")

# Clean and filter data
filtered_df = df[df["main_region"].isin(regions_of_interest)].copy()
filtered_df["ap_mm"] = pd.to_numeric(filtered_df["ap_mm"], errors="coerce")
filtered_df = filtered_df.dropna(subset=["ap_mm"])

# Create 0.5 mm AP bins starting from a clean rounded base
bin_width = 0.5
bin_start = np.floor(filtered_df["ap_mm"].min())
bin_end = np.ceil(filtered_df["ap_mm"].max()) #+ bin_width
ap_bins = pd.interval_range(start=bin_start, end=bin_end, freq=bin_width)

filtered_df["ap_bin"] = pd.cut(filtered_df["ap_mm"], bins=ap_bins)
filtered_df["ap_mid"] = filtered_df["ap_bin"].apply(lambda x: x.mid)

# Choose colormap
cube_cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
cube_array = cube_cmap(np.linspace(0, 1, 256))

# Insert a light gray at the start for 0 values
gray_start = np.array([[0.90, 0.96, 0.94, 1]])  # RGBA for light gray
custom_colors = np.vstack([gray_start, cube_array[1:]])
custom_cmap = ListedColormap(custom_colors)

# Get unique injection regions
unique_inj_regions = filtered_df["inj_region"].dropna().unique()

# Plot heatmaps grouped by inj_region and slice
for inj in unique_inj_regions:
    inj_df = filtered_df[filtered_df["inj_region"] == inj]
    unique_slices = inj_df["slice"].dropna().unique()

    for slice_label in unique_slices:
        sub_df = inj_df[inj_df["slice"] == slice_label]

        # Group and pivot for heatmap
        grouped = (
            sub_df.groupby(["main_region", "ap_mid"])
            .size()
            .reset_index(name="cell_count")
        )
        heatmap_data = grouped.pivot_table(index="main_region", columns="ap_mid", values="cell_count", fill_value=0)
        heatmap_data = heatmap_data.reindex(index=regions_of_interest, fill_value=0)

        # Skip if no data remains after filtering
        if heatmap_data.empty:
            continue

        # Plot
        fig, ax = plt.subplots(figsize=(16, 4))
        sns.heatmap(
            heatmap_data,
            cmap=custom_cmap,
            linewidths=2,
            linecolor="white",
            cbar_kws={'label': 'Cell Count'},
            square=False,
            ax=ax
        )
        ax.set_aspect(0.5)
        ax.set_title(f"Injection Region: {inj} – Slice: {slice_label}")
        ax.set_ylabel('')
        ax.set_xlabel("AP Coordinate (mm)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.invert_xaxis()

        fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/SliceAP_by_injection_heatmap_{inj}_{slice_label}.png", dpi=500)
        fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/SliceAP_by_injection_heatmap_{inj}_{slice_label}.pdf", dpi=500)
        fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/SliceAP_by_injection_heatmap_{inj}_{slice_label}.svg", dpi=500)  
        
        
#%%

# test


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Load dataset
df = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/fm_it.csv') 

# Define regions of interest
regions_of_interest = ["FRP", "MOs", "ACAd", "ACAv", "PL", "ILA", "ORBm", "ORBvl", "ORBl", "AId", "AIv"]

# Extract base region from acronym (e.g., "MOs5" → "MOs")
df["main_region"] = df["acronym"].str.extract(r"^(" + "|".join(regions_of_interest) + ")")

# Clean and filter data
filtered_df = df[df["main_region"].isin(regions_of_interest)].copy()
filtered_df["ap_mm"] = pd.to_numeric(filtered_df["ap_mm"], errors="coerce")
filtered_df = filtered_df.dropna(subset=["ap_mm"])

# Create 0.5 mm AP bins starting from a clean rounded base
bin_width = 0.5
bin_start = np.floor(filtered_df["ap_mm"].min())
bin_end = np.ceil(filtered_df["ap_mm"].max())
ap_bins = pd.interval_range(start=bin_start, end=bin_end, freq=bin_width)

filtered_df["ap_bin"] = pd.cut(filtered_df["ap_mm"], bins=ap_bins)
filtered_df["ap_mid"] = filtered_df["ap_bin"].apply(lambda x: x.mid)

# Choose colormap
cube_cmap = sns.cubehelix_palette(as_cmap=True)
cube_array = cube_cmap(np.linspace(0, 1, 256))

# Insert a light gray at the start for 0 values
gray_start = np.array([[0.98, 0.94, 0.97, 1]])  # RGBA for light gray
custom_colors = np.vstack([gray_start, cube_array[1:]])
custom_cmap = ListedColormap(custom_colors)

# Get unique injection regions
unique_inj_regions = filtered_df["inj_region"].dropna().unique()

# Step 1: Find global max for consistent color scaling
global_max = (
    filtered_df.groupby(["inj_region", "main_region", "ap_mid"])
    .size()
    .reset_index(name="cell_count")["cell_count"]
    .max()
)

# Plot individual heatmaps for each injection region
for inj in unique_inj_regions:
    sub_df = filtered_df[filtered_df["inj_region"] == inj]

    # Group and pivot for heatmap
    grouped = (
        sub_df.groupby(["main_region", "ap_mid"])
        .size()
        .reset_index(name="cell_count")
    )
    heatmap_data = grouped.pivot_table(index="main_region", columns="ap_mid", values="cell_count", fill_value=0)
    heatmap_data = heatmap_data.reindex(index=regions_of_interest, fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(
        heatmap_data,
        cmap=custom_cmap,
        linewidths=2,
        linecolor="white",
        cbar_kws={'label': 'Cell Count'},
        square=False,
        ax=ax,
        vmin=0,
        vmax=global_max  # Set consistent scale
    )
    ax.set_aspect(0.5)  # compress cells along x-axis
    ax.set_title(f"Injection Region: {inj}")
    ax.set_ylabel('')
    ax.set_xlabel("AP Coordinate (mm)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.invert_xaxis()
    
    
    fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/AllCells_by_injection_heatmap_{inj}_scaled.png", dpi=500)
    fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/AllCells_by_injection_heatmap_{inj}_scaled.pdf", dpi=500)
    fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/AllCells_by_injection_heatmap_{inj}_scaled.svg", dpi=500)  
    
#%%


# test 2

df = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/fm_it.csv') 

# Define regions of interest
regions_of_interest = ["FRP", "MOs", "ACAd", "ACAv", "PL", "ILA", "ORBm", "ORBvl", "ORBl", "AId", "AIv"]

# Extract base region from acronym (e.g., "MOs5" → "MOs")
df["main_region"] = df["acronym"].str.extract(r"^(" + "|".join(regions_of_interest) + ")")

# Clean and filter data
filtered_df = df[df["main_region"].isin(regions_of_interest)].copy()
filtered_df["ap_mm"] = pd.to_numeric(filtered_df["ap_mm"], errors="coerce")
filtered_df = filtered_df.dropna(subset=["ap_mm"])

# Create 0.5 mm AP bins starting from a clean rounded base
bin_width = 0.5
bin_start = np.floor(filtered_df["ap_mm"].min())
bin_end = np.ceil(filtered_df["ap_mm"].max()) #+ bin_width
ap_bins = pd.interval_range(start=bin_start, end=bin_end, freq=bin_width)

filtered_df["ap_bin"] = pd.cut(filtered_df["ap_mm"], bins=ap_bins)
filtered_df["ap_mid"] = filtered_df["ap_bin"].apply(lambda x: x.mid)

# Choose colormap
cube_cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
cube_array = cube_cmap(np.linspace(0, 1, 256))

# Insert a light gray at the start for 0 values
gray_start = np.array([[0.90, 0.96, 0.94, 1]])  # RGBA for light gray
custom_colors = np.vstack([gray_start, cube_array[1:]])
custom_cmap = ListedColormap(custom_colors)

# Get unique injection regions
unique_inj_regions = filtered_df["inj_region"].dropna().unique()

# Plot heatmaps grouped by inj_region and slice
for inj in unique_inj_regions:
    inj_df = filtered_df[filtered_df["inj_region"] == inj]
    unique_slices = inj_df["slice"].dropna().unique()

    for slice_label in unique_slices:
        sub_df = inj_df[inj_df["slice"] == slice_label]

        # Group and pivot for heatmap
        grouped = (
            sub_df.groupby(["main_region", "ap_mid"])
            .size()
            .reset_index(name="cell_count")
        )
        heatmap_data = grouped.pivot_table(index="main_region", columns="ap_mid", values="cell_count", fill_value=0)
        heatmap_data = heatmap_data.reindex(index=regions_of_interest, fill_value=0)

        # Skip if no data remains after filtering
        if heatmap_data.empty:
            continue

        # Plot
        fig, ax = plt.subplots(figsize=(16, 4))
        sns.heatmap(
            heatmap_data,
            cmap=custom_cmap,
            linewidths=2,
            linecolor="white",
            cbar_kws={'label': 'Cell Count'},
            square=False,
            ax=ax,
            vmin=0,
            vmax=500 
        )
        ax.set_aspect(0.5)
        ax.set_title(f"Injection Region: {inj} – Slice: {slice_label}")
        ax.set_ylabel('')
        ax.set_xlabel("AP Coordinate (mm)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.invert_xaxis()

        fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/SliceAP_by_injection_heatmap_{inj}_{slice_label}_scaled.png", dpi=500)
        fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/SliceAP_by_injection_heatmap_{inj}_{slice_label}_scaled.pdf", dpi=500)
        fig.savefig(f"Z:/dmclab/Joana/Tracing/Analysis/plots/Heatmaps/SliceAP_by_injection_heatmap_{inj}_{slice_label}_scaled.svg", dpi=500)  
      
      
    