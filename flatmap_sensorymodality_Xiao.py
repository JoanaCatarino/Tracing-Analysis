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
# exclude contralateral/right hemisphere labeling
df["ml_coords"] = df["ml_coords"].map({i:j for i,j in zip(np.arange(1140),np.arange(1139,-1,-1))})

# filter only PFC regions
pfc_structures = ["MOs",
            "ACAv","ACAd",
            "PL",
            "ILA",
            "ORBm","ORBvl","ORBl",
            "AId","AIv",
            "FRP"]

# colors = ["#a5cde2",
#           "#b2de89","#2178b4",
#           "#5db459",
#           "#f89997",
#           "#fdbe6f","#ff7e00","#cbb3d7",
#           "#693d99","#b3b3b3",
#           "#e3191c"]

colors = {"AUD":"#00ff00",
          "SS":"b",
          "VIS":"r"}


stree = StructureTree()
pfc_ids = np.ravel(stree.id.from_acronym(stree.acronym.startswith(pfc_structures)))
# filter cells with pfc_ids
df = df[df["structure_id"].isin(pfc_ids)]
# create figure and axis
fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(8,8))
# load flatmap svg template
svg_path = "C:/Users/xiao/pfc_flatmap_nrrd/flatmap_PFC_template.svg"
svg2png(url=svg_path,  write_to="flatmap_PFC_template.png")

scatter_handle = []
for injsite in ["VIS","AUD","SS"]:
    df_sensory = df[df["injection_site"]==injsite].copy()
    coords_x = df_sensory["ap_coords"].values
    coords_y = df_sensory["dv_coords"].values
    coords_z = df_sensory["ml_coords"].values
    # get u
    us = vol_u[coords_x,coords_y,coords_z]
    # get v
    vs = vol_v[coords_x,coords_y,coords_z]
    # plot scatter
    sh = ax.scatter(us, vs,s=1,c=colors[injsite],zorder=0,label=injsite,alpha=1)
    scatter_handle.append(sh)

scatter_handle[0].set_zorder(1) 
scatter_handle[1].set_zorder(2)
scatter_handle[2].set_zorder(0)


ax.imshow(plt.imread("flatmap_PFC_template.png"),zorder=3)
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_xlim(300,1300)
ax.set_ylim(800,0)
ax.legend(["Visual (VIS)",
           "Auditory (AUD)",
           "Somatosensory (SS)"],loc="lower left",markerscale=10,fontsize=15)
ax.axis('off')
fig.savefig("functional_modality_PFC_soma_ours.png",dpi=300)
