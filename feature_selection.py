import subprocess
import os
import numpy as np
import pandas as pd
import getopt
import sys

# Get datasets for each seizure type
def get_datasets(drive, montage, feats_file):

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

    return datasets, datasets_test, datasets_sz, feature_list

#%% Set threshold for the overlap coefficients based on quantiles

def select_features(datasets, datasets_test, datasets_sz, feature_list, drive, montage, feats_file):
     
    overlap = pd.read_csv('histograms\\overlap_{}_{}.csv'.format(montage, feats_file.split('_')[1]), index_col=0)
    
    target_feats = dict.fromkeys(overlap.index, None)
    
    for sz_type in overlap.index:
        thr = np.mean(overlap.loc[sz_type])
        print('{} | mean: {}'.format(sz_type, np.mean(overlap.loc[sz_type])))
        #print('       thr: {}'.format(thr))
        
        target_feats[sz_type] = list(overlap.columns[overlap.loc[sz_type] <= thr])
        print('       feats: {}'.format(len(target_feats[sz_type])))
    
    # Select target features and save new subsets
    for i,dataset in enumerate(datasets):
        
        target = list(target_feats[datasets_sz[i]])
        target_index = [0, 1] + [feature_list.index(f)+2 for f in target]
        
        subset = dataset[:, target_index]
        
        subset_test = datasets_test[i][:, target_index]
        
        np.save('{}\\TUH\\features\\{}\\feature_subsets\\{}_{}_train'.format(drive, montage, feats_file, datasets_sz[i]), subset)
        np.save('{}\\TUH\\features\\{}\\feature_subsets\\{}_{}_test'.format(drive, montage, feats_file, datasets_sz[i]), subset_test)
   
    
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
            print('get_overlap_coefs.py -m <montage> -e <epoch_length>')
            sys.exit()
        elif opt in ("-m", "--montage"):
            montage = arg
        elif opt in ("-e", "--epoch"):
            epoch = arg
        
    drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
    drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0] 
        
    feats_file = 'window_' + epoch

    datasets, datasets_test, datasets_sz, feature_list = get_datasets(drive, montage, feats_file)
    
    select_features(datasets, datasets_test, datasets_sz, feature_list, drive, montage, feats_file)
            
    

if __name__ == '__main__':
              
    main(sys.argv[1:])    
    