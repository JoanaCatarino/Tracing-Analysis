# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:16:56 2024

@author: JoanaCatarino

co-localized cells analysis

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Select the animal to plot
animal_id = 848602

# Select channels to plot
channel_1 = "cy3"

channel_2 = "green"

channel_co = "dapi"

# Import prep data for a single animal for channel cy3 and green
data = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/df_it_cells.csv')

# Import data for dapi channel that has the co-expression data
data_co = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/data_prep/'f'{animal_id}_{channel_co}_data.csv')

# Import data for each channel and for co_expression
data_cy3 = data[(data["animal_id"] == animal_id) & (data["channel"] == channel_1)]

data_green = data[(data["animal_id"] == animal_id) & (data["channel"] == channel_2)]

# Get injection sites for each channel
inj_region_1 = data.loc[(data['animal_id'] == animal_id) & (data['channel'] == channel_1), 'region'].values[0]

inj_region_2 = data.loc[(data['animal_id'] == animal_id) & (data['channel'] == channel_2), 'region'].values[0]

# Get genotype
genotype = data.loc[data['animal_id'] == animal_id, 'genotype'].values[0]
#%%

# Get relevant data to plot 

PFC_cy3 = data_cy3["PFC_cells"].values[0]

PFC_green = data_green["PFC_cells"].values[0]


# Dapi channel = co-expressing cells

PL = len(data_co[(data_co.region == 'Prelimbic area') & (data_co.layer == ' layer 5')]) 
ILA = len(data_co[(data_co.region == 'Infralimbic area') & (data_co.layer == ' layer 5')]) 
MOs = len(data_co[(data_co.region == 'Secondary motor area') & (data_co.layer == ' layer 5')]) 
ACA = len(data_co[(data_co.region == 'Anterior cingulate area') & (data_co.layer == ' layer 5')]) 
ORB = len(data_co[(data_co.region == 'Orbital area') & (data_co.layer == ' layer 5')]) 
AI = len(data_co[(data_co.region == 'Agranular insular area') & (data_co.layer == ' layer 5')]) 
FRP = len(data_co[(data_co.region == 'Frontal pole') & (data_co.layer == ' layer 5')])


PFC_co = PL + ILA + MOs + ACA + ORB + AI + FRP

#%%

# Plot total cells

labels = ["cy3", "green", "co\n expressed"]
values = [PFC_cy3, PFC_green, PFC_co]
ymax = (max(values) + 1000)
color = ['#623B5A', '#08605F', "#F19C79"] 

fig, ax = plt.subplots(1,1, figsize=(4,5), dpi= 500)
ax.bar(labels, values, width=0.4, color=color)
ax.set(ylim=(0,ymax))
ax.set_title(f'#{animal_id}  {genotype}   inj: {inj_region_1} - {inj_region_2}', fontsize=11, fontweight='bold', x=0.4 , y=1.05)
ax.bar_label(ax.containers[0], fontsize=8.5, color='#494949', padding=3)
ax.margins(x=0.2)
plt.ylabel('Total number of cells', fontsize=10, labelpad=8)
plt.tight_layout()
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_co-expression_PFC_cells.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_co_expression_PFC_cells.svg')

#%%

# Plot PFC Cells across subregions

# Get colors for different subregions

region_colors = {'PL': '#B88A9F', #(RGB: 184, 138, 159)
                 'ILA': '#B0A5C4', #(RGB: 176, 165, 196)
                 'MOs': '#94D0CE', #(RGB: 148, 208, 206)
                 'ACA': '#C2DDB7', #(RGB: 194, 221, 183)
                 'ORB': '#EDD6A6', #(RGB: 237, 214, 166)
                 'AI': '#FFBDA2', #(RGB: 255, 189, 162)
                 'FRP': '#F5A0AF', #(RGB: 245, 160, 175)
                 }

# Plot
labels = ('PL', 'ILA', 'MOs', 'ACA', 'ORB', 'AI', 'FRP')
values = (PL, ILA, MOs, ACA, ORB, AI, FRP)
ymax = (max(values) + 200)
color = [region_colors[region] for region in labels]

fig, ax = plt.subplots(1,1, figsize=(5,4), dpi= 500)
ax.bar(labels, values, width=0.6, color=color)
ax.set(ylim=(0,ymax))
ax.set_title(f'#{animal_id}  {genotype}   inj: {inj_region_1} - {inj_region_2}', fontsize=11, fontweight='bold', x=0.4 , y=1.05)
ax.bar_label(ax.containers[0], fontsize=8.5, color='#494949', padding=3)
ax.margins(x=0.05)
plt.ylabel('Total number of cells', fontsize=10, labelpad=8)
plt.tight_layout()
sns.despine()

plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_co-expression_PFC_regions_gruped.png')
plt.savefig('Z:/dmclab/Joana/tracing/Analysis/plots/'f'{animal_id}/'f'{animal_id}_co-expression_PFC_regions_grouped.svg')
