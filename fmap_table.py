# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:02:15 2024

@author: JoanaCatarino

Get new table with info for flatmap and concatenate all animals
"""
import pandas as pd
import os

# Select data paths
data_p_path = 'Z:/dmclab/Joana/tracing/Analysis/data_prep/'
animal_info_path = 'Z:/dmclab/Joana/tracing/Analysis/animal_info.csv'
fm_it_path = 'Z:/dmclab/Joana/tracing/Analysis/fm_it.csv'


# Load final table if it exists or create a new empty table
if os.path.isfile(fm_it_path):
    try:
        fm_it = pd.read_csv(fm_it_path)
    except Exception as e:
        print(f"Error loading final table: {e}")
        fm_it = pd.DataFrame(columns=['animal_id', 'channel'])  # Initialize empty DataFrame
else:
    fm_it = pd.DataFrame(columns=['animal_id', 'channel'])  # Initialize empty DataFrame

# Load animal info
try:
    animal_info = pd.read_csv(animal_info_path)
except Exception as e:
    raise FileNotFoundError(f"Animal info file not found: {e}")


# Load animal info
animal_info = pd.read_csv(animal_info_path)


# Process all files in the folder
for filename in os.listdir(data_p_path):
    if filename.endswith('_data.csv'):
        # Extract animal_id and channel from the filename
        parts = filename.split('_')
        animal_id = int(parts[0]) # Extract animal_id as the first part
        channel = parts[1] # Extract channel as the second part

        # Skip files with channel 'dapi' - this has the co-localized cells and needs to be analysed differently
        if channel.lower() == 'dapi':
            continue
        
        # Check if the animal_id and channel already exist in the final table
        if ((fm_it['animal_id'] == animal_id) & (fm_it['channel'] == channel)).any():
            continue


        # Import prep data for the selected animal
        data_p = pd.read_csv(os.path.join(data_p_path, filename))
    
    
        # Remove unwanted columns
        data_p = data_p.drop(columns=['Unnamed: 0','ap_mm', 'dv_mm', 'ml_mm'])
        
        
        # Remove unwanted lines (we only want to keep cells in layer 5 and PFC)
            # PFC regions to include: PL, ILA, MOs, ACA, ORB, AI, FRP
        data_p = data_p[data_p['layer'] == ' layer 5']
        
        data_p = data_p[data_p['region'].isin(['Prelimbic area', 'Infralimbic area', 'Secondary motor area',
                                               'Anterior cingulate area', 'Orbital area', 'Agranular insular area',
                                               'Frontal pole'])]

        # Add experimental info to the table
        
        # Import table with animal info
        animal_info = pd.read_csv(animal_info_path)
        
        # Select injection channel
        inj_channel = channel
        data_p['channel'] = inj_channel
        
        # Select genotype + add to table
        genotype = animal_info.loc[animal_info['animal'] == animal_id, 'genotype'].values[0]
        data_p['genotype'] = genotype
        
        # Select sex + add to table
        sex = animal_info.loc[animal_info['animal'] == animal_id, 'sex'].values[0]
        data_p['sex'] = sex
        
        # Select which injected region we are going to analyze + add to table
        inj_region = animal_info.loc[(animal_info['animal'] == animal_id) & (animal_info['channel_cells'] == channel), 'region'].values[0]
        data_p['inj_region'] = inj_region
        
        # Selected injected hemisphere + add to table
        inj_hemisph = animal_info.loc[(animal_info['animal'] == animal_id) & (animal_info['region'] == inj_region),'hemisphere'].values[0]
        data_p['inj_hemisphere'] = inj_hemisph
        
        # Select injection volume + add to table
        inj_volume = animal_info.loc[(animal_info['animal'] == animal_id) & (animal_info['region'] == inj_region),'volume'].values[0]
        data_p['inj_volume'] = inj_volume
            
        # Select slice AP (start, middle or end) + add to table
        slice_ap = animal_info.loc[animal_info['animal'] == animal_id, 'slice'].values[0] 
        data_p['slice'] = slice_ap
        
        # Reorganize the columns within the data frame
        data_p = data_p.loc[:,['animal_id','genotype','sex','channel', 'slice', 'inj_region', 'inj_hemisphere', 'inj_volume', 'acronym', 'region', 'part', 'layer', 'hemisphere', 'ap_coords',
                                 'dv_coords','ml_coords','structure_id']]


        # Append the new data to the final table
        fm_it = pd.concat([fm_it, data_p], ignore_index = True)
        
# Save the updated table
fm_it.to_csv(fm_it_path, index = False)
print('DONE!')



