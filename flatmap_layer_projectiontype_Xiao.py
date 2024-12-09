#%%
import nrrd

vol_u,_ = nrrd.read("C:/Users/xiao/pfc_flatmap_nrrd/anno_U_filled.nrrd")
vol_v,_ = nrrd.read("C:/Users/xiao/pfc_flatmap_nrrd/anno_V_filled.nrrd")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cairosvg import svg2png
import sys
sys.path.append("../")
from StructureTree import StructureTree

df = pd.read_csv("../cells_20240328105016.csv",
                 dtype={"secondary_injection_site" : object,
                        "volume_secondary_nl" : object},
                index_col=0
                        ).replace([np.nan], ["None"])
# filter only single injection
df = df[df["secondary_injection_site"]=="None"]
# filter only green/cre+ cells
df = df[df["channel"]=="green"]
# exclude contralateral/right hemisphere labeling
df["ml_coords"] = df["ml_coords"].map({i:j for i,j in zip(np.arange(1140),np.arange(1139,-1,-1))})


# filter only PFC regions
pfc_structures = ["MOs",
            "ACAv","ACAd",
            "PL",
            "ILA",
            "ORBm","ORBvl","ORBl",
            "AId","AIv", # excluded AIp
            "FRP"]

colors = ["#a5cde2",
          "#b2de89","#2178b4",
          "#5db459",
          "#f89997",
          "#fdbe6f","#ff7e00","#cbb3d7",
          "#693d99","#b3b3b3",
          "#e3191c"]



stree = StructureTree()
pfc_ids = np.ravel(stree.id.from_acronym(stree.acronym.startswith(pfc_structures)))
# filter cells with pfc_ids
df = df[df["structure_id"].isin(pfc_ids)]
# create figure and axis
fig, ax = plt.subplots(ncols=1,nrows=3,figsize=(8,15))
# load flatmap svg template
svg_path = "C:/Users/xiao/pfc_flatmap_nrrd/flatmap_PFC_template.svg"
svg2png(url=svg_path,  write_to="flatmap_PFC_template.png")

ax_n = 0
title_dict = {"Rbp4-Cre":"Rbp4-Cre (L5 IT/PT)",
              "Cux2-Cre":"Cux2-Cre (L2/3 IT)",
              "Tlx3-Cre":"Tlx3-Cre (L5 IT)"}
for geno in ["Rbp4-Cre","Cux2-Cre","Tlx3-Cre"]:
    df_geno = df[df["genotype"]==geno].copy()
    # split by pfc regions
    for acro,c in zip(pfc_structures,colors):
        # get structure ids of acro
        struct_ids = stree.id.from_acronym(stree.acronym.startswith(acro))
        df_acro = df_geno[df_geno["structure_id"].isin(struct_ids)]   
        coords_x = df_acro["ap_coords"].values
        coords_y = df_acro["dv_coords"].values
        coords_z = df_acro["ml_coords"].values
        # get u
        us = vol_u[coords_x,coords_y,coords_z]
        # get v
        vs = vol_v[coords_x,coords_y,coords_z]
        # plot scatter
        sh = ax[ax_n].scatter(us, vs,s=5,c=c,alpha=1)

    ax[ax_n].imshow(plt.imread("flatmap_PFC_template.png"),zorder=3)
    ax[ax_n].set_xlim(300,1300)
    ax[ax_n].set_ylim(800,0)
    ax[ax_n].set_title(title_dict[geno],fontsize=20)
    ax[ax_n].axis('off')
    ax_n += 1

fig.tight_layout()
fig.savefig("layer_projection_distribution_ours.png",dpi=300)
