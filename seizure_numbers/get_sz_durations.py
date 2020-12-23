import json
import os
import numpy as np


# Get average duration (and std) of each type of seizures 
    
durations = dict.fromkeys(['bckg','fnsz', 'gnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'], []) 

list_files = sorted([f for f in os.listdir('.') if f.endswith('.txt')])

for file in list_files:
    
    with open(file, 'r') as fh:
        counts = json.load(fh)
        
    for sz_type in counts.keys():
        durations[sz_type] = durations[sz_type] + counts[sz_type][1] 
        

del durations['bckg']        
for sz_type in durations.keys():
    print('{} | min: {} | Q1: {}'.format(sz_type, np.amin(durations[sz_type]), np.quantile(durations[sz_type], 0.25)))