from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler
import numpy as np
import sys
import getopt
import json
import subprocess
import os
import pickle


# Get datasets for each seizure type
def get_datasets(drive, montage, feats_file):
    
    print('--- getting train subsets ---')
    
    datasets = []
    datasets_sz = []
    try:
        datasets += [np.load('{}\\TUH\\features\\{}\\feature_subsets\\{}_fnsz_train.npy'.format(drive, montage, feats_file))]
        datasets_sz += ['fnsz']
    except:
        print('fnsz file does not exist')
    try:
        datasets += [np.load('{}\\TUH\\features\\{}\\feature_subsets\\{}_gnsz_train.npy'.format(drive, montage, feats_file))]
        datasets_sz += ['gnsz']
    except:
        print('gnsz file does not exist')
    try:
        datasets += [np.load('{}\\TUH\\features\\{}\\feature_subsets\\{}_spsz_train.npy'.format(drive, montage, feats_file))]
        datasets_sz += ['spsz']
    except:
        print('spsz file does not exist')
    try:
        datasets += [np.load('{}\\TUH\\features\\{}\\feature_subsets\\{}_cpsz_train.npy'.format(drive, montage, feats_file))]
        datasets_sz += ['cpsz']
    except:
        print('cpsz file does not exist')
    try:
        datasets += [np.load('{}\\TUH\\features\\{}\\feature_subsets\\{}_absz_train.npy'.format(drive, montage, feats_file))]
        datasets_sz += ['absz']
    except:
        print('absz file does not exist')
    try:
        datasets += [np.load('{}\\TUH\\features\\{}\\feature_subsets\\{}_tnsz_train.npy'.format(drive, montage, feats_file))]
        datasets_sz += ['tnsz']
    except:
        print('tnsz file does not exist')
    try:
        datasets += [np.load('{}\\TUH\\features\\{}\\feature_subsets\\{}_tcsz_train.npy'.format(drive, montage, feats_file))]
        datasets_sz += ['tcsz']
    except:
        print('tcsz file does not exist')
    try:
        datasets += [np.load('{}\\TUH\\features\\{}\\feature_subsets\\{}_mysz_train.npy'.format(drive, montage, feats_file))]
        datasets_sz += ['mysz']
    except:
        print('mysz file does not exist')

    return datasets, datasets_sz

#%% Outlier detection and removal

def remove_outliers(datasets, datasets_sz, montage, labels, scale=3):
    
    for i,dataset in enumerate(datasets):
        
        print('-- removing outliers from {} -- '.format(datasets_sz[i]))

        bg_data_out = dataset[np.reshape(np.argwhere(dataset[:,1] == 6.), (-1,)), :]
        sz_data_out = dataset[np.reshape(np.argwhere(dataset[:,1] == list(labels.keys())[list(labels.values()).index(datasets_sz[i])]), (-1,)), :]
        
        q1_bg = np.quantile(bg_data_out[:, 2:], 0.25, axis=0)
        q3_bg = np.quantile(bg_data_out[:, 2:], 0.75, axis=0)
        q1_sz = np.quantile(sz_data_out[:, 2:], 0.25, axis=0)
        q3_sz = np.quantile(sz_data_out[:, 2:], 0.75, axis=0)
        
        dn_thr_bg = q1_bg - scale * (q3_bg - q1_bg)
        up_thr_bg = q3_bg + scale * (q3_bg - q1_bg)
        dn_thr_sz = q1_sz - scale * (q3_sz - q1_sz)
        up_thr_sz = q3_sz + scale * (q3_sz - q1_sz)
        
        bg_data =  bg_data_out[np.reshape(np.argwhere((bg_data_out[:, 2:] >= dn_thr_bg).all(axis=1) & (bg_data_out[:, 2:] <= up_thr_bg).all(axis=1)), (-1,)), :]
        sz_data = sz_data_out[np.reshape(np.argwhere((sz_data_out[:, 2:] >= dn_thr_sz).all(axis=1) & (sz_data_out[:, 2:] <= up_thr_sz).all(axis=1)), (-1,)), :]
                
        datasets[i] = np.append(bg_data, sz_data, axis=0)
        
        nbefore = np.unique(dataset[:,1], return_counts=True)[1]
        nafter = np.unique(datasets[i][:,1], return_counts=True)[1]
        
        print('-- -- bckg: {} | {} -- sz: {} | {} -- --'.format(nbefore[0], nafter[0], nbefore[1],  nafter[1]))
        
    return datasets
 
    
#%% Feature scaling

def scale_features(datasets, datasets_sz, montage, feats_file):    
    
    for i,dataset in enumerate(datasets):
        
        print('- scaling outliers from {} -'.format(datasets_sz[i]))
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(dataset[:,2:])
        
        datasets[i] = np.append(dataset[:,:2], scaled, axis=1)
        
        pickle.dump(scaler, open('classifiers\\scaler_{}_{}_{}'.format(montage, datasets_sz[i], feats_file.split('_')[1]), 'wb'))
        
    return datasets

#%% Balancing

def balance_dataset(datasets, datasets_sz):
    
    #cc = ClusterCentroids(sampling_strategy='majority', random_state=42)
    cc = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    res_datasets = []
    
    for i,dataset in enumerate(datasets):
        
        print('balancing {} subset'.format(datasets_sz[i]))
        
        X = dataset[:,2:]
        y = dataset[:,1]
        
        print(np.unique(y, return_counts=True))
        
        X_res, y_res = cc.fit_resample(X, y)
        
        res_datasets += [[X_res, y_res]] 
        
        ntotal = np.unique(y_res, return_counts=True)
        
        print('{} | bckg: {} | sz: {}'.format(datasets_sz[i], ntotal[1][0], ntotal[1][1]))
        
    return res_datasets
    

#%% Offline training

def offline_training(datasets, datasets_sz, montage, feats_file):    
    
    for i,dataset in enumerate(datasets):
        
        print('  training classifier for {}'.format(datasets_sz[i]))
        
        X = dataset[0]
        y = dataset[1]
        
        # Classifier grid search
        param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}
        grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)  
        
        grid.fit(X, y)
        model = grid.best_estimator_
        pickle.dump(model, open('classifiers\\svm_{}_{}_{}'.format(montage, datasets_sz[i], feats_file.split('_')[1]), 'wb'))
        
        classification = {}
        best_params = {}
        for p in grid.best_params_.keys():
            best_params[p] = str(grid.best_params_[p])      
        classification['best params'] = best_params
        
        cv_results = {}
        for r in grid.cv_results_.keys():
            cv_results[r] = str(grid.cv_results_[r])
        classification['train results'] = cv_results

        with open('training_{}_{}.json'.format(montage, datasets_sz[i]), 'w') as f:
            json.dump(classification, f)
        
    return classification
    
#%%

def main(argv):
    
    montage = '0103'
    epoch = '10'
    
    try:
        opts, args = getopt.getopt(argv, 'hm:e:',["montage=","epoch="])
    except getopt.GetoptError:
      sys.exit(2)
      
    for opt, arg in opts:
        if opt == '-h':
            print('classifier_training.py -m <montage> -e <epoch_length>')
            sys.exit()
        elif opt in ("-m", "--montage"):
            montage = arg
        elif opt in ("-e", "--epoch"):
            epoch = arg
        
    drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
    drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0] 
        
    labels = {8.: 'fnsz', 9.: 'gnsz', 10.: 'spsz', 11.: 'cpsz', 
              12.: 'absz', 13.: 'tnsz', 15.: 'tcsz', 17.: 'mysz'}


    feats_file = 'window_' + epoch

    datasets, datasets_sz = get_datasets(drive, montage, feats_file)
    datasets = remove_outliers(datasets, datasets_sz, montage, labels, scale=3)
    datasets = scale_features(datasets, datasets_sz, montage, feats_file)
    datasets = balance_dataset(datasets, datasets_sz)
    classification = offline_training(datasets, datasets_sz, montage, feats_file)

if __name__ == '__main__':
              
    main(sys.argv[1:])   
    
    
        
