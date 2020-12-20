from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import subprocess
import json
import pickle
import os
import numpy as np

def classify(X, model, t_constraint=0, sz_type=None, window=None): # simulating online classification
    
    y = np.array([])
    counter = 0
    bg_counter = 0
    
    for i in np.arange(X.shape[0]):
        y = np.append(y, model.predict(X[i,:]))
    
        if y[-1] == 1.: 
            counter += 1
            y[-counter:] = 1.
            bg_counter = 0
        else:
            if bg_counter == 0:
                bg_counter += 1
            else:
                bg_counter += 1
                if counter < t_constraint:
                    y[-counter-bg_counter:] = 0.
                counter = 0
                
    return y


    # y = model.predict(X)
    
    # diffs = np.diff(y) != 0
    # split_ind = np.nonzero(diffs)[0] + 1
    # y_split = np.split(y, split_ind)
    
    # for i,y_s in enumerate(y_split):
    #     if y_s[0] == 1. and len(y_s) < t_constraint:
    #         y_split[i][:] = 0. # this is enough to change the original y
    
    # return y

#%% Get datasets for each seizure type

drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0] 
    
feats_file = 'window_50'
montage = '0103'

labels = {8.: 'fnsz', 9.: 'gnsz', 10.: 'spsz', 11.: 'cpsz', 
          12.: 'absz', 13.: 'tnsz', 15.: 'tcsz', 17.: 'mysz'}

datasets = []
datasets_sz = []
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_fnsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['fnsz']
except:
    print('fnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_gnsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['gnsz']
except:
    print('gnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_spsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['spsz']
except:
    print('spsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_cpsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['cpsz']
except:
    print('cpsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_absz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['absz']
except:
    print('absz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_tnsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['tnsz']
except:
    print('tnsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_tcsz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['tcsz']
except:
    print('tcsz file does not exist')
try:
    datasets += [np.load('{}\\TUH\\features\\{}\\subsets\\{}_mysz_test.npy'.format(drive, montage, feats_file))]
    datasets_sz += ['mysz']
except:
    print('mysz file does not exist')


#%% Get number of samples of bckg and sz

for dataset in datasets:
    print(np.unique(dataset[:,1], return_counts=True))    
    
        
#%% Feature scaling

for i,dataset in enumerate(datasets):

    scaler = pickle.load(open('classifiers\\scaler_{}_{}'.format(datasets_sz[i], feats_file.split('_')[1]), 'rb'))    
    scaled = scaler.transform(dataset[:,2:])
    
    datasets[i] = np.append(dataset[:,:2], scaled, axis=1)
    
    
#%% Split files with other type of seizures 

t_constraint = 0

YT = dict.fromkeys(datasets_sz, np.array([]))
YP = dict.fromkeys(datasets_sz, np.array([]))

yt = dict.fromkeys(datasets_sz, [])
yp = dict.fromkeys(datasets_sz, [])
    
for i,dataset in enumerate(datasets):
    
    model = pickle.load(open('classifiers\\svm_{}_{}'.format(datasets_sz[i], feats_file.split('_')[1]), 'rb'))
    
    events = [e for e in list(np.unique(dataset[:,1])) if e != 6. and e!= list(labels.keys())[list(labels.values()).index(datasets_sz[i])]]
    
    dataset = np.append(np.reshape(np.arange(dataset.shape[0]), (-1,1)), dataset, axis=1)
        
    for event in events: # if there are other types of sz in the file
        dataset = np.delete(dataset, np.reshape(np.argwhere(dataset[:,2] == event), (-1,)), axis=0)
        
        
    list_files = np.unique(dataset[:,1])
    
    for file in list_files:
        
        data = dataset[np.reshape(np.argwhere(dataset[:,1] == file), (-1,)), :]
        diffs = np.diff(data[:,0]) != 1
        split_ind = np.nonzero(diffs)[0] + 1
        
        data = np.split(data, split_ind, axis=0)
        
        for recording in data:
            
            Xrec = recording[:,3:]
            yrec = recording[:,2]
            
            ypred = classify(Xrec, model, sz_type=datasets_sz[i], window=feats_file.split('_')[1])
            
            # used for the final performance report
            if YT[datasets_sz[i]].size == 0:
                YT[datasets_sz[i]] = yrec
                YP[datasets_sz[i]] = ypred
            else:
                YT[datasets_sz[i]] = np.append(YT[datasets_sz[i]], yrec)
                YP[datasets_sz[i]] = np.append(YP[datasets_sz[i]], ypred)
                
            # used for plotting the individual file classifications
            yrec = np.where(yrec==6., 0., yrec)
            yrec = np.where(yrec==list(labels.keys())[list(labels.values()).index(datasets_sz[i])], 1., yrec)
            
            ypred = np.where(ypred==6., 0., ypred)
            ypred = np.where(ypred==list(labels.keys())[list(labels.values()).index(datasets_sz[i])], 1., ypred)
            
            yt[datasets_sz[i]] += [[yrec]]
            yp[datasets_sz[i]] += [[ypred]]
                

with open('classification.json', 'r') as openfile: 
    classification = json.load(openfile) 
              
for i in np.arange(len(datasets)):
    report = classification_report(YT[datasets_sz[i]], YP[datasets_sz[i]], labels=np.unique(YT[datasets_sz[i]]), target_names=['bckg', datasets_sz[i]])
    print('--- {} ---'.format(datasets_sz[i]))
    print(report)    
    report = classification_report(YT[datasets_sz[i]], YP[datasets_sz[i]], labels=np.unique(YT[datasets_sz[i]]), target_names=['bckg', datasets_sz[i]], output_dict=True)
    
    classification[datasets_sz[i]]['report'] = report
    
    for recording in np.arange(len(yt[datasets_sz[i]])):
        plt.figure()
        plt.plot(yt[datasets_sz[i]][recording])
        plt.plot(yp[datasets_sz[i]][recording])
        plt.legend(['bckg', datasets_sz[i]])
    
with open('classification.json', 'w') as f:
    json.dump(classification, f)
    
    