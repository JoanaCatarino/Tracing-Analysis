# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:59:17 2024

@author: JoanaCatarino
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Select the animal to plot
animal_id = 848599

# Select channel to plot
channel = "cy3"

# Import prep data for a single animal
data_p = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/data_prep/'f'{animal_id}_{channel}_data.csv')

# Remove the column 'Unnamed' because it is not giving any extra information
data_p = data_p.drop(columns=['Unnamed: 0'])

# Import table with animal info
animal_info = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/animal_info.csv')

# Select genotype 
genotype = animal_info.loc[animal_info['animal'] == animal_id, 'genotype'].values[0]

# Select sex
sex = animal_info.loc[animal_info['animal'] == animal_id, 'sex'].values[0]

# Select which injected region we are going to analyze:
inj_region = animal_info.loc[(animal_info['animal'] == animal_id) & (animal_info['channel_cells'] == channel), 'region'].values[0]

# Selected injected hemisphere
inj_hemisph = animal_info.loc[(animal_info['animal'] == animal_id) & (animal_info['region'] == inj_region),'hemisphere'].values[0]

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

 #%%
 
 # Plot total number of cells in each hemisphere

hemisphere_counts = data_p['hemisphere'].value_counts()

ipsi_cells = hemisphere_counts.get(inj_hemisph,0) # Same hemisphere as the injection
contra_cells = hemisphere_counts.sum() - ipsi_cells   # Opposite hemisphere to the injection

x = ['Ipsilateral\n Hemisphere', 'Contralateral\n Hemisphere']
y = [ipsi_cells, contra_cells] 
color= ['#85BDA6', '#08605F'] # Select color for the bars, light green for right hemisphere and dark green for left hemisphere
ymax = (max(y) + 1000)

fig, ax = plt.subplots(1,1, figsize=(4,5), dpi= 500)
ax.bar(x, y, width=0.4, color=color)
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

 # Plot cells in PFC for each hemisphere
     # PFC subregions included: PL, ILA, MOs, ACA, ORB, AI, FRP
     
PL_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Prelimbic area') & (data_p.layer == ' layer 5')]) 
ILA_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Infralimbic area') & (data_p.layer == ' layer 5')]) 
MOs_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Secondary motor area') & (data_p.layer == ' layer 5')]) 
ACA_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Anterior cingulate area') & (data_p.layer == ' layer 5')]) 
ORB_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Orbital area') & (data_p.layer == ' layer 5')]) 
AI_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Agranular insular area') & (data_p.layer == ' layer 5')]) 
FRP_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Frontal pole') & (data_p.layer == ' layer 5')])

PL_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Prelimbic area') & (data_p.layer == ' layer 5')]) 
ILA_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Infralimbic area') & (data_p.layer == ' layer 5')]) 
MOs_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Secondary motor area') & (data_p.layer == ' layer 5')]) 
ACA_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Anterior cingulate area') & (data_p.layer == ' layer 5')]) 
ORB_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Orbital area') & (data_p.layer == ' layer 5')]) 
AI_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Agranular insular area') & (data_p.layer == ' layer 5')]) 
FRP_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Frontal pole') & (data_p.layer == ' layer 5')])

# Get total amount of cells in PFC for each hemisphere
PFC_ipsi = PL_ipsi + ILA_ipsi + MOs_ipsi + ACA_ipsi + ORB_ipsi + AI_ipsi + FRP_ipsi
PFC_contra = PL_contra + ILA_contra + MOs_contra + ACA_contra + ORB_contra + AI_contra + FRP_contra

# Plot number of cells in PFC for each hemisphere
x = ['PFC\n Ipsilateral', 'PFC\n Contralateral']
y = [PFC_ipsi, PFC_contra] 
color= ['#B598AF', '#623B5A'] # Select color for the bars, light purple for right hemisphere and dark purple for left hemisphere
ymax = (max(y) + 1000)

fig, ax = plt.subplots(1,1, figsize=(4,5), dpi= 500)
ax.bar(x, y, width=0.4, color=color)
ax.set(ylim=(0,ymax))
ax.set_title(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold', x=0.4 , y=1.05)
ax.bar_label(ax.containers[0], fontsize=8.5, color='#494949', padding=3)
ax.margins(x=0.2)
plt.ylabel('Total number of cells', fontsize=10, labelpad=8)
plt.tight_layout()
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_Total_cells.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_Total_cells.svg')

#%%

 # Compare number of cells between different PFC subregions for each hemisphere
     # Regions to include: Same as in section above
     
# Data for plotting
x = ['PL', 'ILA', 'MOs', 'ACA', 'ORB', 'AI', 'FRP']
y_ipsi = [PL_ipsi, ILA_ipsi, MOs_ipsi, ACA_ipsi, ORB_ipsi, AI_ipsi, FRP_ipsi]
y_contra = [PL_contra, ILA_contra, MOs_contra, ACA_contra, ORB_contra, AI_contra, FRP_contra]

# Map colors based on the region order
colors_ipsi = [region_colors[region] for region in x]
colors_contra = [region_colors[region] for region in x]

# Calculate the maximum value to set consistent y-axis limits
ymax = max(max(y_ipsi), max(y_contra)) + 200  # Add some padding for visual clarity

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), dpi=500)
fig.suptitle(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold')
plt.subplots_adjust(hspace=0.1)

# Ipsilateral Hemisphere Plot
ax1.bar(x, y_ipsi, width=0.4, color=colors_ipsi)
ax1.set_ylim(0, ymax)
ax1.set_ylabel('Total number of cells', labelpad=10, fontsize=10)
ax1.set_title('Ipsilateral Hemisphere', fontsize=10, x=0.5, y=1.05)
ax1.bar_label(ax1.containers[0], fontsize=8.5, color='#494949', padding=3)

# Contralateral Hemisphere Plot
ax2.bar(x, y_contra, width=0.4, color=colors_contra)
ax2.set_ylim(0, ymax)
ax2.set_title('Contralateral Hemisphere', fontsize=10, x=0.5, y=1.05)
ax2.bar_label(ax2.containers[0], fontsize=8.5, color='#494949', padding=3)
ax2.set_ylabel('Total number of cells', labelpad=10, fontsize=10)

plt.tight_layout(pad=2)
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_regions.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_regions.svg')



# Plot for both hemispheres in the same figure - grouped bar plot
# Data for plotting
regions = ['PL', 'ILA', 'MOs', 'ACA', 'ORB', 'AI', 'FRP']
y_ipsi = [PL_ipsi, ILA_ipsi, MOs_ipsi, ACA_ipsi, ORB_ipsi, AI_ipsi, FRP_ipsi]
y_contra = [PL_contra, ILA_contra, MOs_contra, ACA_contra, ORB_contra, AI_contra, FRP_contra]

# X-axis positions for grouped bars
x = np.arange(len(regions))
width = 0.35

# Calculate the maximum value to set consistent y-axis limits
ymax = max(max(y_ipsi), max(y_contra)) + 200  # Add some padding for visual clarity

# Plotting
fig, ax = plt.subplots(figsize=(10, 6), dpi=500)
fig.suptitle(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold')

# Plot ipsilateral bars with hatch pattern
for i, region in enumerate(regions):
    ax.bar(x[i] - width/2, y_ipsi[i], width=width, color=region_colors[region], hatch='//', label='Ipsilateral Hemisphere' if i == 0 else "")

# Plot contralateral bars without hatch pattern
for i, region in enumerate(regions):
    ax.bar(x[i] + width/2, y_contra[i], width=width, color=region_colors[region], label='Contralateral Hemisphere' if i == 0 else "")

# Customize the plot
ax.set_xticks(x)
ax.set_ylim(0, ymax)
ax.set_xticklabels(regions)
ax.set_ylabel('Total number of cells', labelpad=10, fontsize=10)
ax.legend(loc='upper left', frameon=False)

# Add bar labels
for container in ax.containers:
    ax.bar_label(container, fontsize=8.5, color='#494949', padding=3)

plt.tight_layout(pad=2)
sns.despine()
plt.show()


plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_regions_grouped.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_regions_grouped.svg')

#%%

# Compare the number of cells between different PFC subregions for each hemisphere
    # Regions to include: PL, ILA, ACAd, ACAv, ORBm, ORBl, ORBvl, Mos, AId, AIv

PL_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Prelimbic area') & (data_p.layer == ' layer 5')]) 
ILA_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Infralimbic area') & (data_p.layer == ' layer 5')]) 
MOs_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Secondary motor area') & (data_p.layer == ' layer 5')]) 
ACAd_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Anterior cingulate area') & (data_p.layer == ' layer 5') & (data_p.part == ' dorsal part')]) 
ACAv_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Anterior cingulate area') & (data_p.layer == ' layer 5') & (data_p.part == ' ventral part')]) 
ORBm_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Orbital area') & (data_p.layer == ' layer 5') & (data_p.part == ' medial part')]) 
ORBl_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Orbital area') & (data_p.layer == ' layer 5') & (data_p.part == ' lateral part')]) 
ORBvl_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Orbital area') & (data_p.layer == ' layer 5')  & (data_p.part == ' ventrolateral part')]) 
AId_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Agranular insular area') & (data_p.layer == ' layer 5') & (data_p.part == ' dorsal part')]) 
AIv_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Agranular insular area') & (data_p.layer == ' layer 5') & (data_p.part == ' ventral part')])
FRP_ipsi = len(data_p[(data_p.hemisphere == inj_hemisph) & (data_p.region == 'Frontal pole') & (data_p.layer == ' layer 5')])

PL_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Prelimbic area') & (data_p.layer == ' layer 5')]) 
ILA_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Infralimbic area') & (data_p.layer == ' layer 5')]) 
MOs_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Secondary motor area') & (data_p.layer == ' layer 5')]) 
ACAd_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Anterior cingulate area') & (data_p.layer == ' layer 5') & (data_p.part == ' dorsal part')]) 
ACAv_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Anterior cingulate area') & (data_p.layer == ' layer 5') & (data_p.part == ' ventral part')]) 
ORBm_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Orbital area') & (data_p.layer == ' layer 5') & (data_p.part == ' medial part')]) 
ORBl_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Orbital area') & (data_p.layer == ' layer 5') & (data_p.part == ' lateral part')]) 
ORBvl_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Orbital area') & (data_p.layer == ' layer 5')  & (data_p.part == ' ventrolateral part')]) 
AId_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Agranular insular area') & (data_p.layer == ' layer 5') & (data_p.part == ' dorsal part')]) 
AIv_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Agranular insular area') & (data_p.layer == ' layer 5') & (data_p.part == ' ventral part')])
FRP_contra = len(data_p[(data_p.hemisphere != inj_hemisph) & (data_p.region == 'Frontal pole') & (data_p.layer == ' layer 5')])

# Data for plotting
x = ['PL', 'ILA', 'MOs', 'ACAd', 'ACAv', 'ORBm', 'ORBl', 'ORBvl', 'AId', 'AIv', 'FRP']
y_ipsi = [PL_ipsi, ILA_ipsi, MOs_ipsi, ACAd_ipsi, ACAv_ipsi, ORBm_ipsi, ORBl_ipsi, ORBvl_ipsi, AId_ipsi, AIv_ipsi, FRP_ipsi]
y_contra = [PL_contra, ILA_contra, MOs_contra, ACAd_contra, ACAv_contra, ORBm_contra, ORBl_contra, ORBvl_contra, AId_contra, AIv_contra, FRP_contra]


# Calculate the maximum value to set consistent y-axis limits
ymax = max(max(y_ipsi), max(y_contra)) + 200  # Add some padding for visual clarity

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), dpi=500)
fig.suptitle(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=11, fontweight='bold')
plt.subplots_adjust(hspace=0.1)

# Ipsilateral Hemisphere Plot
ax1.bar(x, y_ipsi, width=0.4, color=colors_ipsi)
ax1.set_ylim(0, ymax)
ax1.set_ylabel('Total number of cells', labelpad=10, fontsize=10)
ax1.set_title('Ipsilateral Hemisphere', fontsize=10, x=0.5, y=1.05)
ax1.bar_label(ax1.containers[0], fontsize=8.5, color='#494949', padding=3)

# Contralateral Hemisphere Plot
ax2.bar(x, y_contra, width=0.4, color=colors_contra)
ax2.set_ylim(0, ymax)
ax2.set_title('Contralateral Hemisphere', fontsize=10, x=0.5, y=1.05)
ax2.bar_label(ax2.containers[0], fontsize=8.5, color='#494949', padding=3)
ax2.set_ylabel('Total number of cells', labelpad=10, fontsize=10)

plt.tight_layout(pad=2)
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_regions_parts.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_{channel}_PFC_regions_parts.svg')

#%%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data for plotting
x = ['PL', 'ILA', 'MOs', 'ACAd', 'ACAv', 'ORBm', 'ORBl', 'ORBvl', 'AId', 'AIv', 'FRP']
y_ipsi = [PL_ipsi, ILA_ipsi, MOs_ipsi, ACAd_ipsi, ACAv_ipsi, ORBm_ipsi, ORBl_ipsi, ORBvl_ipsi, AId_ipsi, AIv_ipsi, FRP_ipsi]
y_contra = [PL_contra, ILA_contra, MOs_contra, ACAd_contra, ACAv_contra, ORBm_contra, ORBl_contra, ORBvl_contra, AId_contra, AIv_contra, FRP_contra]

# Combine all counts to determine scaling range
all_counts = y_ipsi + y_contra
min_count = min(all_counts)
max_count = max(all_counts)

# Debugging: Print min/max values
print(f"Min count: {min_count}, Max count: {max_count}")

# Colors
colors_ipsi = "blue"
colors_contra = "red"

# Scaling Dot Sizes
min_dot_size = 10   # Reduced minimum dot size
max_dot_size = 100  # Reduced maximum dot size
scale_dot = lambda count: min_dot_size + ((count - min_count) / (max_count - min_count)) * (max_dot_size - min_dot_size)

# Debugging: Print scaled dot sizes
scaled_ipsi = [scale_dot(count) for count in y_ipsi]
scaled_contra = [scale_dot(count) for count in y_contra]
print("Scaled dot sizes (ipsilateral):", scaled_ipsi)
print("Scaled dot sizes (contralateral):", scaled_contra)

# Plot settings
jitter = 0.3  # Increased horizontal separation for ipsilateral/contralateral

# Initialize the plot
plt.figure(figsize=(12, 6), dpi=300)
plt.title(f'#{animal_id}  {genotype}   inj: {inj_region}', fontsize=14, fontweight='bold')

# Plot ipsilateral data
for i, count in enumerate(y_ipsi):
    plt.scatter(
        i - jitter,  # Horizontal position (with jitter)
        0,  # Vertical position (all dots at the same level)
        s=scale_dot(count),  # Scale the size of the dots
        color=colors_ipsi,
        alpha=0.7,
        label='Ipsilateral Hemisphere' if i == 0 else None,
    )

# Plot contralateral data
for i, count in enumerate(y_contra):
    plt.scatter(
        i + jitter,  # Horizontal position (with jitter)
        0,  # Vertical position (all dots at the same level)
        s=scale_dot(count),  # Scale the size of the dots
        color=colors_contra,
        alpha=0.7,
        label='Contralateral Hemisphere' if i == 0 else None,
    )

# Formatting
plt.xticks(ticks=np.arange(len(x)), labels=x, rotation=45, fontsize=10)
plt.ylabel('Total Number of Cells', fontsize=12)
plt.xlabel('Brain Regions', fontsize=12)
plt.legend(title="Hemisphere", loc="upper right", fontsize=10)
plt.tight_layout()

# Show plot
sns.despine()
plt.show()
