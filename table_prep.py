# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 18:12:28 2024

@author: JoanaCatarino

File to generate data frames with total number of cells 

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import emoji
import os

# Select the animal to plot
animal_id = 857302

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

# Select slice AP (start, middle or end)
slice_ap = animal_info.loc[animal_info['animal'] == animal_id, 'slice'].values[0] 

#%%

# Get data to pupulate new dataframes for plotting

# Total cells
cells = len(data_p)

# Total cells ipsi and contra hemispheres
hemisphere_counts = data_p['hemisphere'].value_counts()

ipsi_cells = hemisphere_counts.get(inj_hemisph,0) # Same hemisphere as the injection
contra_cells = hemisphere_counts.sum() - ipsi_cells   # Opposite hemisphere to the injection

# Cell in PFC subregions for each hemisphere
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

# Total cells in PFC for each hemisphere
PFC_ipsi = PL_ipsi + ILA_ipsi + MOs_ipsi + ACA_ipsi + ORB_ipsi + AI_ipsi + FRP_ipsi
PFC_contra = PL_contra + ILA_contra + MOs_contra + ACA_contra + ORB_contra + AI_contra + FRP_contra

# Total cells in PFC
PFC_all = PFC_ipsi + PFC_contra

# Total cells for each PFC regions (both hemispheres together)
PL = PL_ipsi + PL_contra
ILA = ILA_ipsi + ILA_contra
MOs = MOs_ipsi + MOs_contra
ACA = ACA_ipsi + ACA_contra
ORB = ORB_ipsi + ORB_contra 
AI = AI_ipsi + AI_contra
FRP = FRP_ipsi + FRP_contra

# Proportion of cells 
pp_PL = round((PL/cells)*100,1)
pp_ILA = round((ILA/cells)*100,1)
pp_MOs = round((MOs/cells)*100,1)
pp_ACA = round((ACA/cells)*100,1)
pp_ORB = round((ORB/cells)*100,1)
pp_AI = round((AI/cells)*100,1)
pp_FRP = round((FRP/cells)*100,1)

# Proportion of cells per hemisphere
pp_PL_ipsi = round((PL_ipsi/ipsi_cells)*100,1)
pp_ILA_ipsi = round((ILA_ipsi/ipsi_cells)*100,1)
pp_MOs_ipsi = round((MOs_ipsi/ipsi_cells)*100,1)
pp_ACA_ipsi = round((ACA_ipsi/ipsi_cells)*100,1)
pp_ORB_ipsi = round((ORB_ipsi/ipsi_cells)*100,1)
pp_AI_ipsi = round((AI_ipsi/ipsi_cells)*100,1)
pp_FRP_ipsi = round((FRP_ipsi/ipsi_cells)*100,1)

pp_PL_contra = round((PL_contra/contra_cells)*100,1)
pp_ILA_contra = round((ILA_contra/contra_cells)*100,1)
pp_MOs_contra = round((MOs_contra/contra_cells)*100,1)
pp_ACA_contra = round((ACA_contra/contra_cells)*100,1)
pp_ORB_contra = round((ORB_contra/contra_cells)*100,1)
pp_AI_contra = round((AI_contra/contra_cells)*100,1)
pp_FRP_contra = round((FRP_contra/contra_cells)*100,1)
#%%

# General dataframe for plotting

# Select columns to include in the table

columns = ['animal_id', 'sex', 'genotype', 'channel', 'region', 
           'slice', 'total_cells', 'PFC_cells', 'PL', 
           'ILA', 'MOs','ACA', 'ORB', 'AI', 'FRP',
           'ipsi_cells', 'contra_cells', 'PFC_ipsi',
           'PFC_contra', 'PL_ipsi', 'ILA_ipsi', 'MOs_ipsi', 
           'ACA_ipsi', 'ORB_ipsi', 'AI_ipsi', 'FRP_ipsi', 
           'PL_contra', 'ILA_contra', 'MOs_contra', 'ACA_contra', 
           'ORB_contra', 'AI_contra', 'FRP_contra','pp_PL', 
           'pp_ILA', 'pp_MOs', 'pp_ACA', 'pp_ORB', 'pp_AI', 'pp_FRP',
           'pp_PL_ipsi','pp_ILA_ipsi', 'pp_MOs_ipsi', 'pp_ACA_ipsi', 'pp_ORB_ipsi', 'pp_AI_ipsi', 
           'pp_FRP_ipsi', 'pp_PL_contra', 'pp_ILA_contra', 'pp_MOs_contra', 'pp_ACA_contra', 'pp_ORB_contra',
           'pp_AI_contra', 'pp_FRP_contra']

# Load existing DataFrame or create a new one

file_path = 'Z:/dmclab/Joana/tracing/Analysis/df_it_cells.csv'

# Add data to the dataframe
new_row = {
    'animal_id': animal_id,
    'sex': sex,
    'genotype': genotype,
    'channel': channel,
    'region': inj_region,
    'slice': slice_ap,
    'total_cells': cells,
    'PFC_cells': PFC_all,
    'PL': PL,
    'ILA': ILA,
    'MOs': MOs,
    'ACA': ACA,
    'ORB': ORB,
    'AI': AI,
    'FRP': FRP,
    'ipsi_cells': ipsi_cells,
    'contra_cells': contra_cells,
    'PFC_ipsi': PFC_ipsi,
    'PFC_contra': PFC_contra,
    'PL_ipsi': PL_ipsi,
    'ILA_ipsi': ILA_ipsi,
    'MOs_ipsi': MOs_ipsi,
    'ACA_ipsi': ACA_ipsi,
    'ORB_ipsi': ORB_ipsi,
    'AI_ipsi': AI_ipsi,
    'FRP_ipsi':FRP_ipsi,
    'PL_contra': PL_contra,
    'ILA_contra': ILA_contra,
    'MOs_contra': MOs_contra,
    'ACA_contra': ACA_contra, 
    'ORB_contra': ORB_contra,
    'AI_contra': AI_contra,
    'FRP_contra': FRP_contra,
    'pp_PL': pp_PL,
    'pp_ILA': pp_ILA,
    'pp_MOs': pp_MOs,
    'pp_ACA': pp_ACA,
    'pp_ORB': pp_ORB,
    'pp_AI': pp_AI,
    'pp_FRP': pp_FRP,
    'pp_PL_ipsi': pp_PL_ipsi,
    'pp_ILA_ipsi': pp_ILA_ipsi,
    'pp_MOs_ipsi': pp_MOs_ipsi,
    'pp_ACA_ipsi': pp_ACA_ipsi,
    'pp_ORB_ipsi': pp_ORB_ipsi,
    'pp_AI_ipsi': pp_AI_ipsi, 
    'pp_FRP_ipsi': pp_FRP_ipsi,
    'pp_PL_contra': pp_PL_contra,
    'pp_ILA_contra': pp_ILA_contra,
    'pp_MOs_contra': pp_MOs_contra,
    'pp_ACA_contra': pp_ACA_contra,
    'pp_ORB_contra': pp_ORB_contra,
    'pp_AI_contra': pp_AI_contra,
    'pp_FRP_contra': pp_FRP_contra}

# Check if the file exists
if os.path.exists(file_path):
    # Load the existing data
    df_it_cells = pd.read_csv(file_path)
else:
    # Create a new DataFrame if the file doesn't exist
    df_it_cells = pd.DataFrame(columns=new_row.keys())

# Append the new row to the DataFrame
df_it_cells = pd.concat([df_it_cells, pd.DataFrame([new_row])], ignore_index=True)

# Save the updated DataFrame back to the file
df_it_cells.to_csv(file_path, index=False)

print(emoji.emojize('DONE :star-struck:\U0001F42D'))
