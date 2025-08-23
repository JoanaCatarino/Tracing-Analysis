# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:49:29 2025

@author: JoanaCatarino

Flatmap for TLx3 tracing - voxels by regions

"""

import nrrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from StructureTree import StructureTree
from cairosvg import svg2png

# import volume for the flatmap

vol_u,_ = nrrd.read("Z:/dmclab/Joana/Tracing/Analysis/nrrd_volume/anno_U_filled.nrrd")
vol_v,_ = nrrd.read("Z:/dmclab/Joana/Tracing/Analysis/nrrd_volume/anno_V_filled.nrrd")


#%% 
      # All cells devided by injection site 

df = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/fm_it.csv') 
# filter only PFC regions
pfc_structures = ["MOs",
            "ACAv","ACAd",
            "PL",
            "ILA",
            "ORBm","ORBvl","ORBl",
            "AId","AIv", # excluded AIp
            "FRP"]

colors = ["#2C706C",
          "#93C0A4","#AAC18A",
          "#9177AE",
          "#9B5D87",
          "#E9E993","#D0D06A","#A7A752",
          "#CD9869","#9B714D",
          "#DDB0B8"]



stree = StructureTree()
pfc_ids = np.ravel(stree.id.from_acronym(stree.acronym.startswith(pfc_structures)))
# filter cells with pfc_ids
df = df[df["structure_id"].isin(pfc_ids)]
# create figure and axis
fig, ax = plt.subplots(ncols=1,nrows=3,figsize=(8,15))
# load flatmap svg template
svg_path = "C:/Users/JoanaCatarino/Joana/flatmap_PFC_template.svg"
svg2png(url=svg_path,  write_to="flatmap_PFC_template.png")

ax_n = 0
title_dict = {"dls":"Dorsal Lateral Striatum",
              "dms":"Dorsal Medial Striatum",
              "medial":"Medial Striatum"}

for inj in ["dls","dms","medial"]:
    df_inj = df[df["inj_region"]==inj].copy()
    # split by pfc regions
    for acro,c in zip(pfc_structures,colors):
        # get structure ids of acro
        struct_ids = stree.id.from_acronym(stree.acronym.startswith(acro))
        df_acro = df_inj[df_inj["structure_id"].isin(struct_ids)]   
        coords_x = df_acro["ap_coords"].values.astype(int)
        coords_y = df_acro["dv_coords"].values.astype(int)
        coords_z = df_acro["ml_coords_corrected"].values.astype(int)

        # get u
        us = vol_u[coords_x,coords_y,coords_z]
        # get v
        vs = vol_v[coords_x,coords_y,coords_z]
        # plot scatter
        sh = ax[ax_n].scatter(us, vs,s=0.5,c="#CD9869",alpha=1)
        #sh = ax[ax_n].scatter(us, vs,s=0.5,c=c,alpha=1)

    ax[ax_n].imshow(plt.imread("flatmap_PFC_template.png"),zorder=3)
    ax[ax_n].set_xlim(300,1300)
    ax[ax_n].set_ylim(800,0)
    ax[ax_n].set_title(title_dict[inj],fontsize=20)
    ax[ax_n].axis('off')
    ax_n += 1

fig.savefig("Z:/dmclab/Joana/Tracing/Analysis/plots/flatmaps/AllCells_by_injection.png", dpi=500)
fig.savefig("Z:/dmclab/Joana/Tracing/Analysis/plots/flatmaps/AllCells_by_injection.pdf", dpi=500)
fig.savefig("Z:/dmclab/Joana/Tracing/Analysis/plots/flatmaps/AllCells_by_injection.svg", dpi=500)

#%%

    # All cells divided by injection site and ipsi and contra


df = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/fm_it.csv') 

# filter only PFC regions
pfc_structures = ["MOs",
            "ACAv","ACAd",
            "PL",
            "ILA",
            "ORBm","ORBvl","ORBl",
            "AId","AIv", # excluded AIp
            "FRP"]

colors = ["#2C706C",
          "#93C0A4","#AAC18A",
          "#9177AE",
          "#9B5D87",
          "#E9E993","#D0D06A","#A7A752",
          "#CD9869","#9B714D",
          "#DDB0B8"]


stree = StructureTree()
pfc_ids = np.ravel(stree.id.from_acronym(stree.acronym.startswith(pfc_structures)))
# filter cells with pfc_ids
df = df[df["structure_id"].isin(pfc_ids)]

# load flatmap svg template
svg_path = "C:/Users/JoanaCatarino/Joana/flatmap_PFC_template.svg"
svg2png(url=svg_path,  write_to="flatmap_PFC_template.png")

fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(30, 15))
ax = ax.ravel()  # Flatten axes for 1D indexing

ax_n = 0
title_dict = {
    "dls": "Dorsal Lateral Striatum",
    "dms": "Dorsal Medial Striatum",
    "medial": "Medial Striatum",
    "ipsi": "Ipsilateral",
    "contra": "Contralateral"
}

for inj in ["dls", "dms", "medial"]:
    for projection_type in ["ipsi", "contra"]:
        df_inj_proj = df[(df["inj_region"] == inj) & (df["projection"] == projection_type)].copy()
        

        for acro, c in zip(pfc_structures, colors):
            struct_ids = stree.id.from_acronym(stree.acronym.startswith(acro))
            df_acro = df_inj_proj[df_inj_proj["structure_id"].isin(struct_ids)]   
            
            if df_acro.empty:
                continue

            coords_x = df_acro["ap_coords"].values.astype(int)
            coords_y = df_acro["dv_coords"].values.astype(int)
            coords_z = df_acro["ml_coords"].values.astype(int)
            if coords_z[0] > 569:
                coords_z = coords_z*-1 + 569*2

            us = vol_u[coords_x, coords_y, coords_z]
            vs = vol_v[coords_x, coords_y, coords_z]
  
            ax[ax_n].scatter(us, vs, s=5, c=c, alpha=1)
        

        ax[ax_n].imshow(plt.imread("flatmap_PFC_template.png"), zorder=3)
        ax[ax_n].set_xlim(300, 1300)
        ax[ax_n].set_ylim(800, 0)
        ax[ax_n].set_title(f"{title_dict[inj]} – {title_dict[projection_type]}", fontsize=20)
        ax[ax_n].axis('off')
        ax_n += 1

fig.savefig("Z:/dmclab/Joana/Tracing/Analysis/plots/flatmaps/ipsi_contra_by_injection.png", dpi=500)
fig.savefig("Z:/dmclab/Joana/Tracing/Analysis/plots/flatmaps/ipsi_contra_by_injection.pdf", dpi=500)
fig.savefig("Z:/dmclab/Joana/Tracing/Analysis/plots/flatmaps/ipsi_contra_by_injection.svg", dpi=500)    

#%%

    # All cells divided by injection AP coordinate
    
    
df = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/fm_it.csv') 

# filter only PFC regions
pfc_structures = ["MOs",
            "ACAv","ACAd",
            "PL",
            "ILA",
            "ORBm","ORBvl","ORBl",
            "AId","AIv", # excluded AIp
            "FRP"]

colors = ["#2C706C",
          "#93C0A4","#AAC18A",
          "#9177AE",
          "#9B5D87",
          "#E9E993","#D0D06A","#A7A752",
          "#CD9869","#9B714D",
          "#DDB0B8"]


stree = StructureTree()
pfc_ids = np.ravel(stree.id.from_acronym(stree.acronym.startswith(pfc_structures)))
# filter cells with pfc_ids
df = df[df["structure_id"].isin(pfc_ids)]

# load flatmap svg template
svg_path = "C:/Users/JoanaCatarino/Joana/flatmap_PFC_template.svg"
svg2png(url=svg_path,  write_to="flatmap_PFC_template.png")

fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(30, 15))
ax = ax.ravel()  # Flatten axes for 1D indexing

ax_n = 0
title_dict = {
    "dls": "Dorsal Lateral Striatum",
    "dms": "Dorsal Medial Striatum",
    "medial": "Medial Striatum",
    "start": "start",
    "middle": "middle",
    "end": "end"
}

for inj in ["dls", "dms", "medial"]:
    for slice_ap in ["start", "middle", "end"]:
        df_slice_ap = df[(df["inj_region"] == inj) & (df["slice"] == slice_ap)].copy()
        

        for acro, c in zip(pfc_structures, colors):
            struct_ids = stree.id.from_acronym(stree.acronym.startswith(acro))
            df_acro = df_slice_ap[df_slice_ap["structure_id"].isin(struct_ids)]   
            
            if df_acro.empty:
                continue

            coords_x = df_acro["ap_coords"].values.astype(int)
            coords_y = df_acro["dv_coords"].values.astype(int)
            coords_z = df_acro["ml_coords"].values.astype(int)
            if coords_z[0] > 569:
                coords_z = coords_z*-1 + 569*2

            us = vol_u[coords_x, coords_y, coords_z]
            vs = vol_v[coords_x, coords_y, coords_z]
  
            ax[ax_n].scatter(us, vs, s=0.5, c='#2C706C', alpha=1)
        

        ax[ax_n].imshow(plt.imread("flatmap_PFC_template.png"), zorder=3)
        ax[ax_n].set_xlim(300, 1300)
        ax[ax_n].set_ylim(800, 0)
        ax[ax_n].set_title(f"{title_dict[inj]} – {title_dict[slice_ap]}", fontsize=20)
        ax[ax_n].axis('off')
        ax_n += 1

fig.savefig("Z:/dmclab/Joana/Tracing/Analysis/plots/flatmaps/slice_ap_by_injection.png", dpi=500)
fig.savefig("Z:/dmclab/Joana/Tracing/Analysis/plots/flatmaps/slice_ap_by_injection.pdf", dpi=500)
fig.savefig("Z:/dmclab/Joana/Tracing/Analysis/plots/flatmaps/slice_ap_by_injection.svg", dpi=500)   


