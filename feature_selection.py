import subprocess
import os
import numpy as np
import pandas as pd

# Get datasets for each seizure type

drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0] 
    
feats_file = 'window_50'
montage = '0103'

labels = {8.: 'fnsz', 9.: 'gnsz', 10.: 'spsz', 11.: 'cpsz', 
          12.: 'absz', 13.: 'tnsz', 15.: 'tcsz', 17.: 'mysz'}

datasets = []
datasets_test = []
datasets_sz = []
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_fnsz_train.npy'.format(drive, montage, feats_file))]
    datasets_test += [np.load('{}\\TUH\\features\\{}\\{}_fnsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['fnsz']
except:
    print('fnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_gnsz_train.npy'.format(drive, montage, feats_file))]
    datasets_test += [np.load('{}\\TUH\\features\\{}\\{}_gnsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['gnsz']
except:
    print('gnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_spsz_train.npy'.format(drive, montage, feats_file))]
    datasets_test += [np.load('{}\\TUH\\features\\{}\\{}_spsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['spsz']
except:
    print('spsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_cpsz_train.npy'.format(drive, montage, feats_file))]
    datasets_test += [np.load('{}\\TUH\\features\\{}\\{}_cpsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['cpsz']
except:
    print('cpsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_absz_train.npy'.format(drive, montage, feats_file))]
    datasets_test += [np.load('{}\\TUH\\features\\{}\\{}_absz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['absz']
except:
    print('absz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_tnsz_train.npy'.format(drive, montage, feats_file))]
    datasets_test += [np.load('{}\\TUH\\features\\{}\\{}_tnsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['tnsz']
except:
    print('tnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_tcsz_train.npy'.format(drive, montage, feats_file))]
    datasets_test += [np.load('{}\\TUH\\features\\{}\\{}_tcsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['tcsz']
except:
    print('tcsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_mysz_train.npy'.format(drive, montage, feats_file))]
    datasets_test += [np.load('{}\\TUH\\features\\{}\\{}_mysz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['mysz']
except:
    print('mysz file does not exist')
    

#%% Get number of samples of bckg and sz

for dataset in datasets:
    print(np.unique(dataset[:,1], return_counts=True))    
    
#%% Original feature set 

if feats_file == 'window_025':
    feature_list = ['sampEn', 'FD', 'HE', 
                    'REcD2', 'REcD3', 'REcA3', 
                    'MEANcA3', 'MEANcD3', 'MEANcD2', 
                    'STDcA3', 'STDcD3', 'STDcD2', 
                    'KURTcA3', 'KURTcD3', 'KURTcD2', 
                    'SKEWcA3', 'SKEWcD3', 'SKEWcD2']
    
elif feats_file == 'window_05':
    feature_list = ['sampEn', 'FD', 'HE', 
                    'REcD2', 'REcD3', 'REcD4', 'REcA4', 
                    'MEANcA4', 'MEANcD4', 'MEANcD3', 'MEANcD2', 
                    'STDcA4', 'STDcD4', 'STDcD3', 'STDcD2', 
                    'KURTcA4', 'KURTcD4', 'KURTcD3', 'KURTcD2', 
                    'SKEWcA4', 'SKEWcD4', 'SKEWcD3', 'SKEWcD2']
    
else:
    feature_list = ['sampEn', 'FD', 'HE', 
                    'REcD2', 'REcD3', 'REcD4', 'REcD5', 'REcA5', 
                    'MEANcA5', 'MEANcD5', 'MEANcD4', 'MEANcD3', 'MEANcD2', 
                    'STDcA5', 'STDcD5', 'STDcD4', 'STDcD3', 'STDcD2', 
                    'KURTcA5', 'KURTcD5', 'KURTcD4', 'KURTcD3', 'KURTcD2', 
                    'SKEWcA5', 'SKEWcD5', 'SKEWcD4', 'SKEWcD3', 'SKEWcD2']

    
#%% Set threshold for the overlap coefficients based on quantiles

thr = 0.5
    
overlap = pd.read_csv('histograms\\overlap.csv', index_col=0)

target_feats = dict.fromkeys(overlap.index, None)

for sz_type in overlap.index:
    
    q1 = np.quantile(overlap.loc[sz_type], thr)
    print('{} | Q1: {}'.format(sz_type, q1))
    
    target_feats[sz_type] = list(overlap.columns[overlap.loc[sz_type] <= q1])
    
    
#%% Select target features and save new subsets
    
for i,dataset in enumerate(datasets):
    
    target = list(target_feats[datasets_sz[i]])
    target_index = [0, 1] + [feature_list.index(f)+2 for f in target]
    
    subset = dataset[:, target_index]
    
    subset_test = datasets_test[i][:, target_index]
    
    np.save('{}\\TUH\\features\\{}\\subsets\\{}_{}_train'.format(drive, montage, feats_file, sz_type), subset)
    np.save('{}\\TUH\\features\\{}\\subsets\\{}_{}_test'.format(drive, montage, feats_file, sz_type), subset_test)
   
    