# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:06:20 2024

@author: JoanaCatarino
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import emoji

# Select the animal to pre-process
animal_id = 693754

# Import data for that animal
data = pd.read_csv('Z:/dmclab/Joana/tracing/Analysis/data_raw/'f'{animal_id}_cells.csv')

# Remove the column 'Unnamed' because it is not giving any extra information
data = data.drop(columns=['Unnamed: 0'])

# Create a new data frame that excludes the cells that were found outside the brain slice = root 
data = data[~(data['acronym'] == 'root')]
data = data.reset_index(drop=True) # Reset index

# Check if there are any cells located the middle of the two hemispheres and remove it from the dataset
mid = data[data['ml_mm'] == 0]  
data = data[~(data['ml_mm'] == 0)]

# Check which data is on each hemisphere 
right_hemisphere = data[data['ml_mm'] > 0]
left_hemisphere = data[data['ml_mm'] < 0]

# Create a column showing which data is on each hemisphere
data['hemisphere'] = np.where(data['ml_mm'] > 0, 'right', 'left')

# To create a column with only the region in which the cell is
region=[] # create

for name in data['name'].values:
    region.append(name.split(',')[0])
#print(region)

data['region'] = region # Add column that only includes regions to the dataframe

# To create a column with the part of the region where the cell is
part=[]

for name in data['name'].values:
    #print(name.split(','))
    if len(name.split(',')) == 3:
        if name.split(',')[1].endswith('part'):
            part.append(name.split(',')[1])
        else:part.append(None)

    elif len(name.split(',')) == 2:
        if name.split(',')[-1].endswith('part'): # remove this to work a bit better
            part.append(name.split(',')[-1])
        else:
            part.append(None)
    else:
        part.append(None)      

data['part'] = part

# To create a column with the layer in which the cell is
layer=[]
 
for name in data['name'].values:
    #print(name.split(','))
    if len(name.split(',')) == 3:
        if name.split(',')[-1].startswith(' layer'):
            layer.append(name.split(',')[-1])
        else:layer.append(None)

    elif len(name.split(',')) == 2:
        if name.split(',')[-1].startswith(' layer') or name.split(',')[-1].startswith(' Layer') : # remove this to work a bit better
            layer.append(name.split(',')[-1])
        else:
            layer.append(None)
    else:
        layer.append(None)
        
data['layer'] = layer

# Remove the initial column that had all the info about region, part and layer together 
data = data.drop(columns=['name'])

# Remove the column with slide information  
data = data.drop(columns=['section_name'])

# Insert column with animal id
data.insert(0, 'animal_id', animal_id)

# Reorganize the columns within the data frame
data = data.loc[:,['animal_id','acronym','region','part','layer','hemisphere','structure_id','ap_mm','dv_mm','ml_mm','ap_coords',
                         'dv_coords','ml_coords']]


# Save this pre-processed table without more experiment info 
data.to_csv('Z:/dmclab/Joana/tracing/Analysis/data_prep/'f'{animal_id}_data.csv')

print(emoji.emojize('DONE :star-struck:\U0001F42D'))
