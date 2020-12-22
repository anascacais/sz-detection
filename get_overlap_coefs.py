import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import sys 
import getopt


# Get datasets for each seizure type
def get_datasets(drive, montage, feats_file):

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


    if feats_file == 'window_025':
        feature_list = ['class', 'sampEn', 'FD', 'HE', 
                        'REcD2', 'REcD3', 'REcA3', 
                        'MEANcA3', 'MEANcD3', 'MEANcD2', 
                        'STDcA3', 'STDcD3', 'STDcD2', 
                        'KURTcA3', 'KURTcD3', 'KURTcD2', 
                        'SKEWcA3', 'SKEWcD3', 'SKEWcD2']
        # re_list = ['class', 'REcD2', 'REcD3', 'REcA3']
        # mean_list = ['class', 'MEANcA3', 'MEANcD3', 'MEANcD2']
        # std_list = ['class', 'STDcA3', 'STDcD3', 'STDcD2']
        # kurt_list = ['class', 'KURTcA3', 'KURTcD3', 'KURTcD2']
        # skew_list = ['class', 'SKEWcA3', 'SKEWcD3', 'SKEWcD2']
        
    elif feats_file == 'window_05':
        feature_list = ['class', 'sampEn', 'FD', 'HE', 
                        'REcD2', 'REcD3', 'REcD4', 'REcA4', 
                        'MEANcA4', 'MEANcD4', 'MEANcD3', 'MEANcD2', 
                        'STDcA4', 'STDcD4', 'STDcD3', 'STDcD2', 
                        'KURTcA4', 'KURTcD4', 'KURTcD3', 'KURTcD2', 
                        'SKEWcA4', 'SKEWcD4', 'SKEWcD3', 'SKEWcD2']
        # re_list = ['class', 'REcD2', 'REcD3', 'REcD4', 'REcA4']
        # mean_list = ['class', 'MEANcA4', 'MEANcD4', 'MEANcD3', 'MEANcD2']
        # std_list = ['class', 'STDcA4', 'STDcD4', 'STDcD3', 'STDcD2']
        # kurt_list = ['class', 'KURTcA4', 'KURTcD4', 'KURTcD3', 'KURTcD2']
        # skew_list = ['class', 'SKEWcA4', 'SKEWcD4', 'SKEWcD3', 'SKEWcD2']
        
    else:
        feature_list = ['class', 'sampEn', 'FD', 'HE', 
                        'REcD2', 'REcD3', 'REcD4', 'REcD5', 'REcA5', 
                        'MEANcA5', 'MEANcD5', 'MEANcD4', 'MEANcD3', 'MEANcD2', 
                        'STDcA5', 'STDcD5', 'STDcD4', 'STDcD3', 'STDcD2', 
                        'KURTcA5', 'KURTcD5', 'KURTcD4', 'KURTcD3', 'KURTcD2', 
                        'SKEWcA5', 'SKEWcD5', 'SKEWcD4', 'SKEWcD3', 'SKEWcD2']
        # re_list = ['class', 'REcD2', 'REcD3', 'REcD4', 'REcD5', 'REcA5']
        # mean_list = ['class', 'MEANcA5', 'MEANcD5', 'MEANcD4', 'MEANcD3', 'MEANcD2']
        # std_list = ['class', 'STDcA5', 'STDcD5', 'STDcD4', 'STDcD3', 'STDcD2']    
        # kurt_list = ['class', 'KURTcA5', 'KURTcD5', 'KURTcD4', 'KURTcD3', 'KURTcD2']
        # skew_list = ['class', 'SKEWcA5', 'SKEWcD5', 'SKEWcD4', 'SKEWcD3', 'SKEWcD2']

    return datasets, datasets_sz, feature_list
    
    
#%% Compute normalized histograms and multiply them to get a coefficient of overlap
def get_nbins(bg_data, sz_data):
    
    iqr = np.quantile(bg_data, 0.75) - np.quantile(bg_data, 0.25)
    bw = 2 * (iqr / np.sqrt(bg_data.size))
    datmin, datmax = np.amin(bg_data), np.amax(bg_data)
    datrng = datmax - datmin
    nbins_bg = int((datrng / bw) + 1)
    
    iqr = np.quantile(sz_data, 0.75) - np.quantile(sz_data, 0.25)
    bw = 2 * (iqr / np.sqrt(sz_data.size))
    datmin, datmax = np.amin(sz_data), np.amax(sz_data)
    datrng = datmax - datmin
    nbins_sz = int((datrng / bw) + 1)
    
    min_b = min([nbins_bg, nbins_sz])
    
    print('fd bins: {}'.format(min_b))
    
    if min_b > 500: 
        return 500
    else:
        return min_b

def remove_outliers(dataset, thr_const=3):
    
    data = dataset[:,2:]
    
    q1 = np.quantile(data, 0.25, axis=0)
    q3 = np.quantile(data, 0.75, axis=0)

    dn_thr = q1 - thr_const * (q3 - q1)
    up_thr = q3 + thr_const * (q3 - q1)
    
    dataset = dataset[np.reshape(np.argwhere((data >= dn_thr).all(axis=1) & (data <= up_thr).all(axis=1)), (-1,)), :]
    
    return dataset

def get_overlap(datasets, datasets_sz, feature_list, montage, labels, feats_file):
    
    overlap = pd.DataFrame(np.zeros((len(datasets_sz), len(feature_list[1:]))), index=datasets_sz, columns=feature_list[1:])
    
    for i,dataset in enumerate(datasets):
        print('---- {} ----'.format(datasets_sz[i]))  
        
        bg_data_out = dataset[np.reshape(np.argwhere(dataset[:,1] == 6.), (-1,)), :]
        sz_data_out = dataset[np.reshape(np.argwhere(dataset[:,1] == list(labels.keys())[list(labels.values()).index(datasets_sz[i])]), (-1,)), :]
        
        bg_data = remove_outliers(bg_data_out)
        sz_data = remove_outliers(sz_data_out)
        
        for j in np.arange(2, dataset.shape[1]):
            print('-- feature: {} --'.format(feature_list[j-1]))
            
            max_range = np.amax([np.amax(bg_data[:,j]), np.amax(sz_data[:,j])])
            min_range = np.amin([np.amin(bg_data[:,j]), np.amin(sz_data[:,j])])
        
            nbins = get_nbins(bg_data[:,j], sz_data[:,j])           
            
            plt.figure()
            hist_bg, bins_r, _ = plt.hist(bg_data[:,j], density=True, alpha=0.7, range=[min_range, max_range], bins=nbins)
            hist_sz, _, _ = plt.hist(sz_data[:,j], density=True, alpha=0.7, range=[min_range, max_range], bins=nbins)
            plt.legend(['bckg', datasets_sz[i]])
            plt.xlabel(feature_list[j-1])
            plt.tight_layout()
            plt.savefig('histograms\\{}_{}_{}_{}'.format(montage, datasets_sz[i], feature_list[j-1], feats_file.split('_')[1]))
            plt.close()
            
            bw = [bins_r[n]-bins_r[n-1] for n in range(1, len(bins_r))]
            
            area_bg = np.array([hist_bg[n] * bw[n] for n in range(len(hist_bg))])
            area_sz = np.array([hist_sz[n] * bw[n] for n in range(len(hist_sz))])
            
            overlap[feature_list[j-1]][datasets_sz[i]] = np.sum(np.sqrt(area_bg * area_sz))
            print(overlap[feature_list[j-1]][datasets_sz[i]])
    
    overlap.to_csv('histograms\\overlap_{}_{}.csv'.format(montage, feats_file.split('_')[1]), index=True, columns=feature_list[1:])
    
    plt.figure()
    sb.heatmap(overlap, 
           xticklabels=overlap.columns.values, 
           yticklabels=overlap.index.values, 
           cmap="YlGnBu",
           vmin=0,
           vmax=1,
           annot=False)
    plt.tight_layout()
    plt.savefig('histograms\\overlap_{}_{}'.format(montage, feats_file.split('_')[1]))

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
    
    labels = {8.: 'fnsz', 9.: 'gnsz', 10.: 'spsz', 11.: 'cpsz', 
              12.: 'absz', 13.: 'tnsz', 15.: 'tcsz', 17.: 'mysz'}

    datasets, datasets_sz, feature_list = get_datasets(drive, montage, feats_file)
    
    get_overlap(datasets, datasets_sz, feature_list, montage, labels, feats_file)
            
    

if __name__ == '__main__':
              
    main(sys.argv[1:])      
    