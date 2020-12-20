import numpy as np
import itertools
import subprocess
import os

def get_best_combo(counts, target, comb_n, sz_type):

    min_diff = np.inf
    
    for n in comb_n:
        print('--- trying comb of {} sessions ---'.format(n))
        
        combinations = itertools.combinations(counts.keys(), n)
        n_combo = sum(1 for _ in combinations)
        print('{} combinations'.format(n_combo))
        
        for session_comb in itertools.combinations(counts.keys(), n):
            try: 
                nbckg0 = counts[session_comb[0]][6.0]
            except:
                nbckg0 = 0
            try:
                nbckg1 = counts[session_comb[1]][6.0]
            except:
                nbckg1 = 0
            try: 
                nsz0 = counts[session_comb[0]][sz_type]
            except:
                nsz0 = 0
            try: 
                nsz1 = counts[session_comb[1]][sz_type]
            except:
                nsz1 = 0
            
            sub = abs(target - (nbckg0+nbckg1)) + abs(target - (nsz0+nsz1))
            if sub < min_diff:
                min_diff = sub
                best_combo = session_comb
                samples = [nbckg0+nbckg1, nsz0+nsz1]
                
        print('{}: {}'.format(best_combo, samples))
                
    print('{}: {} | target: {}'.format(best_combo, samples, target))
    return {'sessions': [str(st) for st in best_combo], 'target': target, 'nsamples': samples}

#%% Clean data + Get session combination that yields the closest number of test samples to the desired one (in both background and seizure samples) 

drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0] 

montage = '0103'    
feats_file = 'window_10'

data = np.load('{}\\TUH\\features\\{}\\{}.npy'.format(drive, montage, feats_file))

data = data[~np.isnan(data).any(axis=1)]    
data = data[~np.isinf(data).any(axis=1)] 

labels = {8.: 'fnsz', 9.: 'gnsz', 10.: 'spsz', 11.: 'cpsz', 
          12.: 'absz', 13.: 'tnsz', 15.: 'tcsz', 17.: 'mysz'}
    
ntotal = np.unique(data[:,1], return_counts=True)
best_combo = {}

for sz_type in labels.keys():
    
    print('seizure {}'.format(labels[sz_type]))
    try:
        target = float(ntotal[1][np.argwhere(ntotal[0]==sz_type)]*0.2)
        print(target)
        
        counts = {}
        
        list_sessions = np.unique(np.array([f[:-3] for f in map(str, map(int, data[:,0]))]))
        for i,session in enumerate(list_sessions):
            
            print('session {} out of {}'.format(i, len(list_sessions)))
            
            index = np.reshape(np.argwhere(np.array([f[:-3] for f in map(str, map(int, data[:,0]))]) == session), (-1,))
            samples = data[index, :] 
            
            if sz_type in samples[:,1]:
            
                bg_index = np.reshape(np.argwhere(samples[:,1] == 6.), (-1,))
                sz_index = np.reshape(np.argwhere(samples[:,1] == sz_type), (-1,))
                samples = np.append(samples[bg_index,:], samples[sz_index,:], axis=0)
                
                unique_counts = np.unique(samples[:,1], return_counts=True)
                
                try:
                    if len(unique_counts[0]) == 1:
                        counts[str(session)] = {unique_counts[0][0]: unique_counts[1][0]}
                    else:
                        counts[str(session)] = {unique_counts[0][0]: unique_counts[1][0], unique_counts[0][1]: unique_counts[1][1]}
                except:
                    print('---- session {} has no bckg or {} ----'.format(session, labels[sz_type]))
            
            else:
                print('---- session {} has no {} ----'.format(session, labels[sz_type]))
                
        # get sessions combination that yields the closest number of test samples to the desired one (in both background and seizure samples) 
        best_combo[labels[sz_type]] = get_best_combo(counts, target, [1, 2], sz_type)
        
    except:
        print('---- seizure {} not in dataset ----'.format(labels[sz_type]))
                
            
#%% Split dataset into train and test + create separate datasets for each sz_type            

for sz_type in best_combo.keys():
        
    print('---- {} ----'.format(sz_type))
    train = np.array([])
    test = np.array([])
    
    if best_combo[sz_type]['nsamples'][0] != 0 and best_combo[sz_type]['nsamples'][1] != 0:
        
        test_sessions = best_combo[sz_type]['sessions']
        for session in test_sessions:
            
            # get test samples from the specified sessions
            test_index = np.reshape(np.argwhere(np.array([f[:-3] for f in map(str, map(int, data[:,0]))]) == session), (-1,))
            test_samples = data[test_index, :] 
            
            if test.size == 0:
                test = test_samples
            else:
                test = np.append(test, test_samples, axis=0)
            
            # get train samples by deleting the training sessions (but only the bckg and sz_type samples) - order is not important for this one
            train_samples = np.delete(data, test_index, axis=0)
            bg_index = np.reshape(np.argwhere(train_samples[:,1] == 6.), (-1,))
            
            if train.size == 0:
                train = train_samples[bg_index, :]
            else:
                train = np.append(train, train_samples[bg_index, :], axis=0)
            
            sz_type_index = list(labels.keys())[list(labels.values()).index(sz_type)]
            sz_index = np.reshape(np.argwhere(train_samples[:,1] == sz_type_index), (-1,))
            train = np.append(train, train_samples[sz_index, :], axis=0)
        
        print(np.unique(train[:,1], return_counts=True))
        print(np.unique(test[:,1], return_counts=True))
            
        np.save('{}\\TUH\\features\\0103\\{}_{}_test'.format(drive, feats_file, sz_type), test)
        np.save('{}\\TUH\\features\\0103\\{}_{}_train'.format(drive, feats_file, sz_type), train)
        
    else:
        print('-- best combo is not so good --')

        