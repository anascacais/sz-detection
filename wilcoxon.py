import json
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

f1_off = np.array([])
f1_on = np.array([])

for montage in ['0103']:

    with open('report_{}_offline.json'.format(montage), 'r') as json_file:
        offline_report = json.load(json_file)
        
    with open('report_{}_online.json'.format(montage), 'r') as json_file:
        online_report = json.load(json_file)
    
    for sz_type in offline_report.keys():
        f1_off = np.append(f1_off, offline_report[sz_type]['macro avg']['f1-score'])
        
    for sz_type in online_report.keys():
        f1_on = np.append(f1_on, online_report[sz_type]['macro avg']['f1-score'])
    
stats.wilcoxon(f1_off, f1_on, mode='exact', alternative='less')    

#%% 

f1_02 = np.array([])
f1_0103 = np.array([])

montage = '0103'

with open('report_{}_offline.json'.format(montage), 'r') as json_file:
    offline_report = json.load(json_file)
    
with open('report_{}_online.json'.format(montage), 'r') as json_file:
    online_report = json.load(json_file)

for sz_type in offline_report.keys():
    f1_0103 = np.append(f1_0103, offline_report[sz_type]['weighted avg']['f1-score'])
    
for sz_type in online_report.keys():
    f1_0103 = np.append(f1_0103, online_report[sz_type]['weighted avg']['f1-score'])
    
montage = '02'

with open('report_{}_offline.json'.format(montage), 'r') as json_file:
    offline_report = json.load(json_file)
    
with open('report_{}_online.json'.format(montage), 'r') as json_file:
    online_report = json.load(json_file)

for sz_type in offline_report.keys():
    f1_02 = np.append(f1_02, offline_report[sz_type]['weighted avg']['f1-score'])
    
for sz_type in online_report.keys():
    f1_02 = np.append(f1_02, online_report[sz_type]['weighted avg']['f1-score'])
      


