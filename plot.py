# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:54:23 2024

@author: JoanaCatarino

File to plot different tracing info
"""

from color_function import color
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Select the animal to plot
animal_id = 857302

# Select channel to plot
channel = "cy3"

# Import prep data for a single animal
data = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/df_it_cells.csv')

#%%
 # Get dataframe specific to analyse single animal and select important info

# Filter the DataFrame for the selected animal
animal_data = data[(data["animal_id"] == animal_id) & (data["channel"] == channel)]

# Get injection region
inj_region = animal_data["region"].values[0]

# Get genotype
genotype = animal_data["genotype"].values[0]


#%%
 # Define color code/schematic for the different regions and use it for all plots 
 
region_colors = {'PL': '#B88A9F', #(RGB: 184, 138, 159)
                 'ILA': '#B0A5C4', #(RGB: 176, 165, 196)
                 'MOs': '#94D0CE', #(RGB: 148, 208, 206)
                 'ACA': '#C2DDB7', #(RGB: 194, 221, 183)
                 'ORB': '#EDD6A6', #(RGB: 237, 214, 166)
                 'AI': '#FFBDA2', #(RGB: 255, 189, 162)
                 'FRP': '#F5A0AF', #(RGB: 245, 160, 175)
                 }

new_colors = {'PL': '#99E2B4',
              'ILA': '#78C6A3',
              'MOs': '#56AB91', 
              'ACA': '#469D89', 
              'ORB': '#248277', 
              'AI': '#14746F',
              'FRP': '#F5A0AF',
                 }
#%%

 # Plot total cells and PFC cells (both hemispheres together)

# Extract the relevant values
total_cells = animal_data["total_cells"].values[0]
pfc_cells = animal_data["PFC_cells"].values[0]

# Plot 
labels = ["Total Cells", "PFC Cells"]
values = [total_cells, pfc_cells]
ymax = (max(values) + 1000)
color = ['#623B5A', '#08605F'] 

fig, ax = plt.subplots(1,1, figsize=(4,5), dpi= 500)
ax.bar(labels, values, width=0.4, color=color)
ax.set(ylim=(0,ymax))
ax.set_title(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold', x=0.4 , y=1.05)
ax.bar_label(ax.containers[0], fontsize=8.5, color='#494949', padding=3)
ax.margins(x=0.2)
plt.ylabel('Total number of cells', fontsize=10, labelpad=8)
plt.tight_layout()
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_Total_cells.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_Total_cells.svg')


#%%
# Plot total number of PFC cells in each hemisphere

# Extract the relevant values
PFC_ipsi = animal_data["PFC_ipsi"].values[0]
PFC_contra = animal_data["PFC_contra"].values[0]

# Plot
labels = ("PFC_ipsi", "PFC_contra")
values = (PFC_ipsi, PFC_contra)
ymax = (max(values) + 200)
color = ['#85BDA6', '#08605F']

fig, ax = plt.subplots(1,1, figsize=(4,5), dpi= 500)
ax.bar(labels, values, width=0.4, color=color)
ax.set(ylim=(0,ymax))
ax.set_title(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold', x=0.4 , y=1.05)
ax.bar_label(ax.containers[0], fontsize=8.5, color='#494949', padding=3)
ax.margins(x=0.2)
plt.ylabel('Total number of cells', fontsize=10, labelpad=8)
plt.tight_layout()
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_cells.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_cells.svg')

#%%

# Plot total cell in PFC subregions (both hemishphere together)

# Extract relevant values
PL = animal_data["PL"].values[0]
ILA = animal_data["ILA"].values[0]
MOs = animal_data["MOs"].values[0]
ACA = animal_data["ACA"].values[0]
ORB = animal_data["ORB"].values[0]
AI = animal_data["AI"].values[0]
FRP = animal_data["FRP"].values[0]
  
# Plot
labels = ('PL', 'ILA', 'MOs', 'ACA', 'ORB', 'AI', 'FRP')
values = (PL, ILA, MOs, ACA, ORB, AI, FRP)
ymax = (max(values) + 200)
color = [region_colors[region] for region in labels]

fig, ax = plt.subplots(1,1, figsize=(5,4), dpi= 500)
ax.bar(labels, values, width=0.6, color=color)
ax.set(ylim=(0,ymax))
ax.set_title(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold', x=0.4 , y=1.05)
ax.bar_label(ax.containers[0], fontsize=8.5, color='#494949', padding=3)
ax.margins(x=0.05)
plt.ylabel('Total number of cells', fontsize=10, labelpad=8)
plt.tight_layout()
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_regions_gruped.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_regions_grouped.svg')

#%%

# Plot total cells in PFC subregions per hemisphere

# Extract relevant values
PL_ipsi = animal_data["PL_ipsi"].values[0]
PL_contra = animal_data["PL_contra"].values[0]
ILA_ipsi = animal_data["ILA_ipsi"].values[0]
ILA_contra = animal_data["ILA_contra"].values[0]
MOs_ipsi = animal_data["MOs_ipsi"].values[0]
MOs_contra = animal_data["MOs_contra"].values[0]
ACA_ipsi = animal_data["ACA_ipsi"].values[0]
ACA_contra = animal_data["ACA_contra"].values[0]
ORB_ipsi = animal_data["ORB_ipsi"].values[0]
ORB_contra = animal_data["ORB_contra"].values[0]
AI_ipsi = animal_data["AI_ipsi"].values[0]
AI_contra = animal_data["AI_contra"].values[0]
FRP_ipsi = animal_data["FRP_ipsi"].values[0]
FRP_contra = animal_data["FRP_contra"].values[0]

# Plot
labels = ['PL', 'ILA', 'MOs', 'ACA', 'ORB', 'AI', 'FRP']
values_ipsi = [PL_ipsi, ILA_ipsi, MOs_ipsi, ACA_ipsi, ORB_ipsi, AI_ipsi, FRP_ipsi]
values_contra = [PL_contra, ILA_contra, MOs_contra, ACA_contra, ORB_contra, AI_contra, FRP_contra]

# X-axis positions for grouped bars
x = np.arange(len(labels))
width = 0.35

# Calculate the maximum value to set consistent y-axis limits
ymax = max(max(values_ipsi), max(values_contra)) + 200  # Add some padding for visual clarity

# Define color for each hemisphere
color_ipsi = ['#85BDA6']
color_contra = ['#08605F']

# Plotting
fig, ax = plt.subplots(figsize=(10, 6), dpi=500)
fig.suptitle(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold')

# Plot ipsilateral bars with hatch pattern
for i, region in enumerate(labels):
    ax.bar(x[i] - width/2, values_ipsi[i], width=width, color=color_ipsi, label='Ipsilateral Hemisphere' if i == 0 else "")

# Plot contralateral bars without hatch pattern
for i, region in enumerate(labels):
    ax.bar(x[i] + width/2, values_contra[i], width=width, color=color_contra, label='Contralateral Hemisphere' if i == 0 else "")


# Customize the plot
ax.set_xticks(x)
ax.set_ylim(0, ymax)
ax.set_xticklabels(labels)
ax.set_ylabel('Total number of cells', labelpad=10, fontsize=10)
ax.legend(loc='upper left', frameon=False)

# Add bar labels
for container in ax.containers:
    ax.bar_label(container, fontsize=8.5, color='#494949', padding=3)

plt.tight_layout(pad=2)
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_regions.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_regions.svg')

#%%

# Plot porpotion of cells (both hemispheres together)

# Extract relevant values
pp_PL = animal_data["pp_PL"].values[0]
pp_ILA = animal_data["pp_ILA"].values[0]
pp_MOs = animal_data["pp_MOs"].values[0]
pp_ACA = animal_data["pp_ACA"].values[0]
pp_ORB = animal_data["pp_ORB"].values[0]
pp_AI = animal_data["pp_AI"].values[0]
pp_FRP = animal_data["pp_FRP"].values[0]
  
# Plot
labels = ('PL', 'ILA', 'MOs', 'ACA', 'ORB', 'AI', 'FRP')
values = (pp_PL, pp_ILA, pp_MOs, pp_ACA, pp_ORB, pp_AI, pp_FRP)
ymax = 100
color = [region_colors[region] for region in labels]

fig, ax = plt.subplots(1,1, figsize=(5,4), dpi= 500)
ax.bar(labels, values, width=0.6, color=color)
ax.set(ylim=(0,ymax))
ax.set_title(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold', x=0.4 , y=1.05)
ax.bar_label(ax.containers[0], fontsize=8.5, color='#494949', padding=3)
ax.margins(x=0.05)
plt.ylabel('Proportion of cells', fontsize=10, labelpad=8)
plt.tight_layout()
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_pp_PFC_regions_grouped.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_pp_PFC_regions_grouped.svg')
 
#%%

# Plot proportion of cells per hemisphere


# Extract relevant values
pp_PL_ipsi = animal_data["pp_PL_ipsi"].values[0]
pp_PL_contra = animal_data["pp_PL_contra"].values[0]
pp_ILA_ipsi = animal_data["pp_ILA_ipsi"].values[0]
pp_ILA_contra = animal_data["pp_ILA_contra"].values[0]
pp_MOs_ipsi = animal_data["pp_MOs_ipsi"].values[0]
pp_MOs_contra = animal_data["pp_MOs_contra"].values[0]
pp_ACA_ipsi = animal_data["pp_ACA_ipsi"].values[0]
pp_ACA_contra = animal_data["pp_ACA_contra"].values[0]
pp_ORB_ipsi = animal_data["pp_ORB_ipsi"].values[0]
pp_ORB_contra = animal_data["pp_ORB_contra"].values[0]
pp_AI_ipsi = animal_data["pp_AI_ipsi"].values[0]
pp_AI_contra = animal_data["pp_AI_contra"].values[0]
pp_FRP_ipsi = animal_data["pp_FRP_ipsi"].values[0]
pp_FRP_contra = animal_data["pp_FRP_contra"].values[0]

# Plot
labels = ['PL', 'ILA', 'MOs', 'ACA', 'ORB', 'AI', 'FRP']
values_ipsi = [pp_PL_ipsi, pp_ILA_ipsi, pp_MOs_ipsi, pp_ACA_ipsi, pp_ORB_ipsi, pp_AI_ipsi, pp_FRP_ipsi]
values_contra = [pp_PL_contra, pp_ILA_contra, pp_MOs_contra, pp_ACA_contra, pp_ORB_contra, pp_AI_contra, pp_FRP_contra]

# X-axis positions for grouped bars
x = np.arange(len(labels))
width = 0.35

# Calculate the maximum value to set consistent y-axis limits
ymax = 100  # Add some padding for visual clarity

# Define color for each hemisphere
color_ipsi = ['#85BDA6']
color_contra = ['#08605F']

# Plotting
fig, ax = plt.subplots(figsize=(10, 6), dpi=500)
fig.suptitle(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold')

# Plot ipsilateral bars with hatch pattern
for i, region in enumerate(labels):
    ax.bar(x[i] - width/2, values_ipsi[i], width=width, color=color_ipsi, label='Ipsilateral Hemisphere' if i == 0 else "")

# Plot contralateral bars without hatch pattern
for i, region in enumerate(labels):
    ax.bar(x[i] + width/2, values_contra[i], width=width, color=color_contra, label='Contralateral Hemisphere' if i == 0 else "")


# Customize the plot
ax.set_xticks(x)
ax.set_ylim(0, ymax)
ax.set_xticklabels(labels)
ax.set_ylabel('Proportion of cells', labelpad=10, fontsize=10)
ax.legend(loc='upper left', frameon=False)

# Add bar labels
for container in ax.containers:
    ax.bar_label(container, fontsize=8.5, color='#494949', padding=3)

plt.tight_layout(pad=2)
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_pp_PFC_regions.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_pp_PFC_regions.svg')

#%%

# Plot porportion of cells but dividing by PFC cells instead of Total cells


# Extract relevant values
pp_PL_2 = round(((animal_data["pp_PL"].values[0])/(animal_data["PFC_cells"].values[0]))
pp_ILA_2 = animal_data["pp_ILA"].values[0]
pp_MOs_2 = animal_data["pp_MOs"].values[0]
pp_ACA_2 = animal_data["pp_ACA"].values[0]
pp_ORB_2 = animal_data["pp_ORB"].values[0]
pp_AI_2 = animal_data["pp_AI"].values[0]
pp_FRP_2 = animal_data["pp_FRP"].values[0]
  

pp_FRP_contra = round((FRP_contra/contra_cells)*100,1)

# Plot
labels = ('PL', 'ILA', 'MOs', 'ACA', 'ORB', 'AI', 'FRP')
values = (pp_PL, pp_ILA, pp_MOs, pp_ACA, pp_ORB, pp_AI, pp_FRP)
ymax = 100
color = [region_colors[region] for region in labels]

fig, ax = plt.subplots(1,1, figsize=(5,4), dpi= 500)
ax.bar(labels, values, width=0.6, color=color)
ax.set(ylim=(0,ymax))
ax.set_title(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold', x=0.4 , y=1.05)
ax.bar_label(ax.containers[0], fontsize=8.5, color='#494949', padding=3)
ax.margins(x=0.05)
plt.ylabel('Proportion of cells', fontsize=10, labelpad=8)
plt.tight_layout()
sns.despine()
