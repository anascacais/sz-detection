from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import subprocess
import os
import pickle


# Get datasets for each seizure type

drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0] 
    
feats_file = 'window_50'
montage = '0103'

labels = {8.: 'fnsz', 9.: 'gnsz', 10.: 'spsz', 11.: 'cpsz', 
          12.: 'absz', 13.: 'tnsz', 15.: 'tcsz', 17.: 'mysz'}

datasets = []
datasets_sz = []
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_fnsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['fnsz']
except:
    print('fnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_gnsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['gnsz']
except:
    print('gnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_spsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['spsz']
except:
    print('spsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_cpsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['cpsz']
except:
    print('cpsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_absz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['absz']
except:
    print('absz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_tnsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['tnsz']
except:
    print('tnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_tcsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['tcsz']
except:
    print('tcsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_mysz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['mysz']
except:
    print('mysz file does not exist')


#%% Outlier detection and removal
    
for i,dataset in enumerate(datasets):
    
    thr_const = 3
    
    bg_data = dataset[np.reshape(np.argwhere(dataset[:,1] == 6.), (-1,)), 2:]
    sz_data = dataset[np.reshape(np.argwhere(dataset[:,1] == list(labels.keys())[list(labels.values()).index(datasets_sz[i])]), (-1,)), 2:]    
    
    q1_bg = np.quantile(bg_data, 0.25, axis=0)
    q3_bg = np.quantile(bg_data, 0.75, axis=0)
    
    q1_sz = np.quantile(sz_data, 0.25, axis=0)
    q3_sz = np.quantile(sz_data, 0.75, axis=0)
    
    dn_thr_bg = q1_bg - thr_const * (q3_bg - q1_bg)
    up_thr_bg = q3_bg + thr_const * (q3_bg - q1_bg)
    
    dn_thr_sz = q1_sz - thr_const * (q3_sz - q1_sz)
    up_thr_sz = q3_sz + thr_const * (q3_sz - q1_sz)
    
    bg_data = dataset[np.reshape(np.argwhere((bg_data >= dn_thr_bg).all(axis=1) & (bg_data <= up_thr_bg).all(axis=1)), (-1,)), :]   
    sz_data = dataset[np.reshape(np.argwhere((sz_data >= dn_thr_sz).all(axis=1) & (sz_data <= up_thr_sz).all(axis=1)), (-1,)), :]   

    datasets[i] = np.append(bg_data, sz_data, axis=0)
 
    
#%% Feature scaling
    
for i,dataset in enumerate(datasets):
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataset[:,2:])
    
    datasets[i] = np.append(dataset[:,:2], scaled, axis=1)
    
    pickle.dump(scaler, open('classifiers\\scaler_{}_{}'.format(datasets_sz[i], feats_file.split('_')[1]), 'wb'))
    

#%% Offline training
    
classification = {}

for i,dataset in enumerate(datasets):
    
    X = dataset[:,2:]
    y = dataset[:,1]
    
    # Classifier grid search
    #param_grid = {'C': [100], 'gamma': [1], 'degree': [1, 2, 3, 4, 5], 'kernel': ['poly','rbf']}
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001],'kernel': ['linear', 'poly', 'sigmoid','rbf']}
    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)  
    
    grid.fit(X, y)
    classification[datasets_sz[i]]['best params'] = grid.best_params_ 
    classification[datasets_sz[i]]['train results'] = grid.cv_results_

    model = grid.best_estimator_
    pickle.dump(model, open('classifiers\\svm_{}_{}'.format(datasets_sz[i], feats_file.split('_')[1]), 'wb'))
    

with open('classification.json', 'w') as f:
    json.dump(classification, f)
    
    
