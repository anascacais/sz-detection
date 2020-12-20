import subprocess
import os
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

#%% Plot univariate distributions of each feature

for i,dataset in enumerate(datasets):
    
    bg_data = dataset[np.reshape(np.argwhere(dataset[:,1] == 6.), (-1,)), :]
    sz_data = dataset[np.reshape(np.argwhere(dataset[:,1] == list(labels.keys())[list(labels.values()).index(datasets_sz[i])]), (-1,)), :]
    
    for j in np.arange(2, dataset.shape[1]):
    
        plt.figure()
        sb.kdeplot(bg_data[:,j], fill=True)
        sb.kdeplot(sz_data[:,j], fill=True)
        plt.xlabel(feature_list[j-1])
        plt.legend(['bckg', datasets_sz[i]])
        
        plt.savefig('uni_distributions\\{}_{}'.format(datasets_sz[i], feats_file.split('_')[1]))
        plt.close()
        
        
#%% For each seizure type, compute pairwise relationships between the features (only with train dataset)
    
# for i,dataset in enumerate(datasets):
    
#     df = pd.DataFrame(data=dataset[:,1:], index=dataset[:,1], columns=feature_list)
    
#     df1 = df[['class', 'sampEn', 'FD', 'HE']]
#     pairplot(df1, hue='class')
#     plt.savefig('pairplots\\{}_{}_nonlinear.png'.format(datasets_sz[i], feats_file.split('_')[1]))
#     plt.close()
    
#     df1 = df[re_list]
#     pairplot(df1, hue='class')
#     plt.savefig('pairplots\\{}_{}_energies.png'.format(datasets_sz[i], feats_file.split('_')[1]))
#     plt.close()
    
#     df1 = df[mean_list]
#     pairplot(df1, hue='class')
#     plt.savefig('pairplots\\{}_{}_means.png'.format(datasets_sz[i], feats_file.split('_')[1]))
#     plt.close()
    
#     df1 = df[std_list]
#     pairplot(df1, hue='class')
#     plt.savefig('pairplots\\{}_{}_std.png'.format(datasets_sz[i], feats_file.split('_')[1]))
#     plt.close()
    
#     df1 = df[kurt_list]
#     pairplot(df1, hue='class')
#     plt.savefig('pairplots\\{}_{}_kurt.png'.format(datasets_sz[i], feats_file.split('_')[1]))
#     plt.close()
    
#     df1 = df[skew_list]
#     pairplot(df1, hue='class')
#     plt.savefig('pairplots\\{}_{}_skew.png'.format(datasets_sz[i], feats_file.split('_')[1]))
#     plt.close()

