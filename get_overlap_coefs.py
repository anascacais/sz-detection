import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

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
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_fnsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['fnsz']
except:
    print('fnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_gnsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['gnsz']
except:
    print('gnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_spsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['spsz']
except:
    print('spsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_cpsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['cpsz']
except:
    print('cpsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_absz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['absz']
except:
    print('absz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_tnsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['tnsz']
except:
    print('tnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_tcsz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['tcsz']
except:
    print('tcsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\{}_mysz_train.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['mysz']
except:
    print('mysz file does not exist')

#%%
    
if feats_file == 'window_025':
    feature_list = ['class', 'sampEn', 'FD', 'HE', 
                    'REcD2', 'REcD3', 'REcA3', 
                    'MEANcA3', 'MEANcD3', 'MEANcD2', 
                    'STDcA3', 'STDcD3', 'STDcD2', 
                    'KURTcA3', 'KURTcD3', 'KURTcD2', 
                    'SKEWcA3', 'SKEWcD3', 'SKEWcD2']
    re_list = ['class', 'REcD2', 'REcD3', 'REcA3']
    mean_list = ['class', 'MEANcA3', 'MEANcD3', 'MEANcD2']
    std_list = ['class', 'STDcA3', 'STDcD3', 'STDcD2']
    kurt_list = ['class', 'KURTcA3', 'KURTcD3', 'KURTcD2']
    skew_list = ['class', 'SKEWcA3', 'SKEWcD3', 'SKEWcD2']
    
elif feats_file == 'window_05':
    feature_list = ['class', 'sampEn', 'FD', 'HE', 
                    'REcD2', 'REcD3', 'REcD4', 'REcA4', 
                    'MEANcA4', 'MEANcD4', 'MEANcD3', 'MEANcD2', 
                    'STDcA4', 'STDcD4', 'STDcD3', 'STDcD2', 
                    'KURTcA4', 'KURTcD4', 'KURTcD3', 'KURTcD2', 
                    'SKEWcA4', 'SKEWcD4', 'SKEWcD3', 'SKEWcD2']
    re_list = ['class', 'REcD2', 'REcD3', 'REcD4', 'REcA4']
    mean_list = ['class', 'MEANcA4', 'MEANcD4', 'MEANcD3', 'MEANcD2']
    std_list = ['class', 'STDcA4', 'STDcD4', 'STDcD3', 'STDcD2']
    kurt_list = ['class', 'KURTcA4', 'KURTcD4', 'KURTcD3', 'KURTcD2']
    skew_list = ['class', 'SKEWcA4', 'SKEWcD4', 'SKEWcD3', 'SKEWcD2']
    
else:
    feature_list = ['class', 'sampEn', 'FD', 'HE', 
                    'REcD2', 'REcD3', 'REcD4', 'REcD5', 'REcA5', 
                    'MEANcA5', 'MEANcD5', 'MEANcD4', 'MEANcD3', 'MEANcD2', 
                    'STDcA5', 'STDcD5', 'STDcD4', 'STDcD3', 'STDcD2', 
                    'KURTcA5', 'KURTcD5', 'KURTcD4', 'KURTcD3', 'KURTcD2', 
                    'SKEWcA5', 'SKEWcD5', 'SKEWcD4', 'SKEWcD3', 'SKEWcD2']
    re_list = ['class', 'REcD2', 'REcD3', 'REcD4', 'REcD5', 'REcA5']
    mean_list = ['class', 'MEANcA5', 'MEANcD5', 'MEANcD4', 'MEANcD3', 'MEANcD2']
    std_list = ['class', 'STDcA5', 'STDcD5', 'STDcD4', 'STDcD3', 'STDcD2']    
    kurt_list = ['class', 'KURTcA5', 'KURTcD5', 'KURTcD4', 'KURTcD3', 'KURTcD2']
    skew_list = ['class', 'SKEWcA5', 'SKEWcD5', 'SKEWcD4', 'SKEWcD3', 'SKEWcD2']


#%% Compute normalized histograms and multiply them to get a coefficient of overlap

overlap = pd.DataFrame(np.zeros((len(datasets_sz), len(feature_list[1:]))), index=datasets_sz, columns=feature_list[1:])

for i,dataset in enumerate(datasets):
    print('---- {} ----'.format(datasets_sz[i]))  
    
    bg_data = dataset[np.reshape(np.argwhere(dataset[:,1] == 6.), (-1,)), :]
    sz_data = dataset[np.reshape(np.argwhere(dataset[:,1] == list(labels.keys())[list(labels.values()).index(datasets_sz[i])]), (-1,)), :]
    
    for j in np.arange(2, dataset.shape[1]):
        print('-- feature: {} --'.format(feature_list[j-1]))
        
        max_range = np.amax([np.amax(bg_data[:,j]), np.amax(sz_data[:,j])])
        min_range = np.amin([np.amin(bg_data[:,j]), np.amin(sz_data[:,j])])
    
        _, bins_bg, _ = plt.hist(bg_data[:,j], density=True, bins='auto', alpha=0.7, range=[min_range, max_range])
        _, bins_sz, _ = plt.hist(sz_data[:,j], density=True, bins='auto', alpha=0.7, range=[min_range, max_range])
        plt.close()
        
        bins = np.amin([bins_bg.size, bins_sz.size])
        
        plt.figure()
        hist_bg, bins_r, _ = plt.hist(bg_data[:,j], density=True, alpha=0.7, range=[min_range, max_range], bins=bins)
        hist_sz, _, _ = plt.hist(sz_data[:,j], density=True, alpha=0.7, range=[min_range, max_range], bins=bins)
        plt.legend(['bckg', datasets_sz[i]])
        plt.xlabel(feature_list[j-1])
        plt.savefig('histograms\\{}_{}_{}'.format(datasets_sz[i], feature_list[j-1], feats_file.split('_')[1]))
        plt.close()
        
        bins_w = [bins_r[n]-bins_r[n-1] for n in range(1, len(bins_r))]
        
        area_bg = np.array([hist_bg[n] * bins_w[n] for n in range(len(hist_bg))])
        area_sz = np.array([hist_sz[n] * bins_w[n] for n in range(len(hist_sz))])
        
        overlap[feature_list[j-1]][datasets_sz[i]] = np.sum(np.sqrt(area_bg * area_sz))
        print(overlap[feature_list[j-1]][datasets_sz[i]])

overlap.to_csv('histograms\\overlap.csv', index=True, columns=feature_list[1:])

plt.figure()
sb.heatmap(overlap, 
       xticklabels=overlap.columns.values, 
       yticklabels=overlap.index.values, 
       cmap="YlGnBu",
       vmin=0,
       vmax=1,
       annot=False)
plt.savefig('histograms\\overlap_{}'.format(feats_file.split('_')[1]))


