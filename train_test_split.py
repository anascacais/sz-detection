import numpy as np
import itertools
import subprocess
import getopt
import sys
import os
import json

def get_best_combo(counts, target, comb_n, sz_type, comb_type):

    min_diff = np.inf
    
    for n in comb_n:
        print('    trying comb of {} {} '.format(n, comb_type))
        
        combinations = itertools.combinations(counts.keys(), n)
        n_combo = sum(1 for _ in combinations)
        print('      {} combinations'.format(n_combo))
        
        for comb in itertools.combinations(counts.keys(), n):
            
            nbckg = 0
            nsz = 0
            
            for session in comb:
                try: 
                    nbckg += counts[session][6.]
                except:
                    nbckg += 0
                    
                try: 
                    nsz += counts[session][sz_type]
                except:
                    nbckg += 0
            
            # 4*target = target treino
            sub = abs(target - nbckg) + abs(target - nsz) + abs(4*target - (5*target - nsz))
                
            if sub < min_diff:
                min_diff = sub
                best_combo = comb
                samples = [nbckg, nsz]
                
        print('      {}: {}'.format(best_combo, samples))
                
    print('    {}: {} | target: {}'.format(best_combo, samples, target))
    return {comb_type: [str(st) for st in best_combo], 'target': target, 'nsamples': samples}


def get_best_combo_json(target, comb_n, sz_type, montage, labels, epoch='10', comb_type='sessions'):
    
    with open('session_counts_{}_{}.json'.format(montage, epoch)) as json_file:
        counts_file = json.load(json_file)
    
    counts = counts_file[labels[sz_type]]
    min_diff = np.inf
    
    for n in comb_n:
        print('    trying comb of {} {}'.format(n, comb_type))
        
        combinations = itertools.combinations(counts.keys(), n)
        n_combo = sum(1 for _ in combinations)
        print('      {} combinations'.format(n_combo))
        
        for comb in itertools.combinations(counts.keys(), n):
            
            nbckg = 0
            nsz = 0
            
            for session in comb:
                try: 
                    nbckg += counts[session]['6']
                except:
                    nbckg += 0
                    
                try: 
                    nsz += counts[session][str(int(sz_type))]
                except:
                    nbckg += 0
            
            # 4*target = target treino
            sub = abs(target - nbckg) + abs(target - nsz) + abs(4*target - (5*target - nsz))
                
            if sub < min_diff:
                min_diff = sub
                best_combo = comb
                samples = [nbckg, nsz]
                
        print('      {}: {}'.format(best_combo, samples))
                
    print('    {}: {} | target: {}'.format(best_combo, samples, target))
    return {comb_type: [str(st) for st in best_combo], 'target': target, 'nsamples': samples}

                
#%% Clean data + Get session combination that yields the closest number of test samples to the desired one (in both background and seizure samples) 

def get_session_combo(drive, montage, feats_file, labels, from_count_files=False):

    data = np.load('{}\\TUH\\features\\{}\\{}.npy'.format(drive, montage, feats_file))
    
    data = data[~np.isnan(data).any(axis=1)]    
    data = data[~np.isinf(data).any(axis=1)] 
    
    ntotal = np.unique(data[:,1], return_counts=True)
    best_combo = {}
    
    if not from_count_files:
        counts = dict.fromkeys([labels[n] for n in ntotal[0] if n!=6.], None)
        list_sessions = np.unique(np.array([f[:-3] for f in map(str, map(int, data[:,0]))]))
        
        for i,session in enumerate(list_sessions):
            print('session {} out of {}'.format(i+1, len(list_sessions)))
            
            index = np.reshape(np.argwhere(np.array([f[:-3] for f in map(str, map(int, data[:,0]))]) == session), (-1,))
            samples = data[index, :]
                
            for sz_type in labels.keys(): 
            #for sz_type in [12.]:    
                try:
                    if sz_type in samples[:,1] or 6. in samples[:,1]:
                        
                        bg_index = np.reshape(np.argwhere(samples[:,1] == 6.), (-1,))
                        sz_index = np.reshape(np.argwhere(samples[:,1] == sz_type), (-1,))
                        samples_sz = np.append(samples[bg_index,:], samples[sz_index,:], axis=0)
                        
                        unique_counts = np.unique(samples_sz[:,1], return_counts=True)
                        
                        if counts[labels[sz_type]] == None: counts[labels[sz_type]] = {}
                        if len(unique_counts[0]) == 1:
                            counts[labels[sz_type]][str(session)] = {int(unique_counts[0][0]): int(unique_counts[1][0])}
                        else:
                            counts[labels[sz_type]][str(session)] = {int(unique_counts[0][0]): int(unique_counts[1][0]), int(unique_counts[0][1]): int(unique_counts[1][1])}
                                    
                    else:
                        print('---- session {} has no bckg or {} ----'.format(session, labels[sz_type]))
                
                except:
                    print('---- seizure {} not in dataset ----'.format(labels[sz_type]))
                       
                    
        with open('session_counts_{}_{}.json'.format(montage, feats_file.split('_')[1]), 'w') as f:
            json.dump(counts, f)         
    
        for sz_type in labels.keys(): 
        #for sz_type in [12.]: 
            try:
                print('-- {} --'.format(labels[sz_type]))
                target = float(ntotal[1][np.argwhere(ntotal[0]==sz_type)]*0.2)
                best_combo[labels[sz_type]] = get_best_combo(counts[labels[sz_type]], target, [1, 2], sz_type, 'sessions')       
            except:
                print('---- seizure {} not in dataset ----'.format(labels[sz_type]))
                
    else:
        for sz_type in labels.keys(): 
        #for sz_type in [12.]: 
            try:
                print('-- {} --'.format(labels[sz_type]))
                target = float(ntotal[1][np.argwhere(ntotal[0]==sz_type)]*0.2)
                #print(target)
            except:
                print('---- seizure {} not in dataset ----'.format(labels[sz_type]))
            try:
                best_combo[labels[sz_type]] =  get_best_combo_json(target, [1, 2], sz_type, montage, labels) 
            except Exception as e:
                print(e)
                    
    return data, best_combo
                
#%% Get file combination that yields the closest number of test samples to the desired one (if sessions didn't work)

def get_file_combo(data, sz_type):
    
    counts = {}
    
    data_sz = data[np.reshape(np.argwhere(data[:,1] == sz_type), (-1,)), :]        
    list_files = np.unique(data_sz[:,0])
    
    print('   {} files'.format(len(list_files)))
    for i,file in enumerate(list_files):
        
        index = np.reshape(np.argwhere(data[:,0] == file), (-1,))
        samples = data[index, :]
        
        bg_index = np.reshape(np.argwhere(samples[:,1] == 6.), (-1,))
        sz_index = np.reshape(np.argwhere(samples[:,1] == sz_type), (-1,))
        samples_sz = np.append(samples[bg_index,:], samples[sz_index,:], axis=0)
        
        unique_counts = np.unique(samples_sz[:,1], return_counts=True)
        
        if len(unique_counts[0]) == 1:
            counts[file] = {int(unique_counts[0][0]): int(unique_counts[1][0])}
        else:
            counts[file] = {int(unique_counts[0][0]): int(unique_counts[1][0]), int(unique_counts[0][1]): int(unique_counts[1][1])}
                        
    ntotal = np.unique(data[:,1], return_counts=True)
    target = float(ntotal[1][np.argwhere(ntotal[0]==sz_type)]*0.2)

    best_combo = get_best_combo(counts, target, [1, 2, 3], sz_type, 'files')     
    
    return best_combo

#%% Split dataset into train and test + create separate datasets for each sz_type            

def get_split_by_sessions(test_sessions, data, sz_type, labels):
    
    test_index = np.array([])
        
    for session in test_sessions:
        
        # get test samples from the specified sessions 
        aux = np.reshape(np.argwhere(np.array([f[:-3] for f in map(str, map(int, data[:,0]))]) == session), (-1,))
        if test_index.size == 0:
            test_index = aux
        else:
            test_index = np.append(test_index, aux)
            
    test = data[test_index, :] 
    
    # get train samples by deleting the training sessions (but only the bckg and sz_type samples) - order is not important for this one
    train_samples = np.delete(data, test_index, axis=0)
    
    bg_index = np.reshape(np.argwhere(train_samples[:,1] == 6.), (-1,))
    train = train_samples[bg_index, :]
    
    sz_type_index = list(labels.keys())[list(labels.values()).index(sz_type)]
    sz_index = np.reshape(np.argwhere(train_samples[:,1] == sz_type_index), (-1,))
    train = np.append(train, train_samples[sz_index, :], axis=0)    
        
    return train, test
    

def get_split_by_files(test_files, data, sz_type, labels):
    
    test_index = np.array([])
    
    for file in test_files:
    
        # get test samples from the specified sessions 
        aux = np.reshape(np.argwhere(data[:,0] == float(file)), (-1,))
        if test_index.size == 0:
            test_index = aux
        else:
            test_index = np.append(test_index, aux)
            
    test = data[test_index, :] 
    
    # get train samples by deleting the training sessions (but only the bckg and sz_type samples) - order is not important for this one
    train_samples = np.delete(data, test_index, axis=0)
    
    bg_index = np.reshape(np.argwhere(train_samples[:,1] == 6.), (-1,))
    train = train_samples[bg_index, :]
    
    sz_type_index = list(labels.keys())[list(labels.values()).index(sz_type)]
    sz_index = np.reshape(np.argwhere(train_samples[:,1] == sz_type_index), (-1,))
    train = np.append(train, train_samples[sz_index, :], axis=0)    
    
    return train, test
    


def train_test_split(data, best_combo, drive, montage, feats_file, labels):

    for sz_type in best_combo.keys():
    #for sz_type in ['absz']:        
        print('---- {} ----'.format(sz_type))
        
        ratio_sessions = best_combo[sz_type]['nsamples'][1] / best_combo[sz_type]['nsamples'][0]
        print('   sessions sz/bckg: {}'.format(ratio_sessions))
        
        if ratio_sessions >= 2/3:
            
            test_sessions = best_combo[sz_type]['sessions']
            train, test = get_split_by_sessions(test_sessions, data, sz_type, labels)
            
        else:
            print('     session combo was not so good, trying file combo')
            
            try:
                best_file_combo = get_file_combo(data, list(labels.keys())[list(labels.values()).index(sz_type)])
                ratio_files = best_file_combo['nsamples'][1] / best_file_combo['nsamples'][0]
                
                if ratio_files > ratio_sessions:
                
                    print('   files sz/bckg: {}'.format(ratio_files))
                    
                    if ratio_files < 2/3: print('   -- neither combo was good --')
                        
                    test_files = best_file_combo['files']
                    train, test = get_split_by_files(test_files, data, sz_type, labels)
                    
                else:
                    print('   -- session combo was better --')
                    
                    test_sessions = best_combo[sz_type]['sessions']
                    train, test = get_split_by_sessions(test_sessions, data, sz_type, labels)
                        
            except Exception as e:
                print(e)
        
        try:
            if len(np.unique(train[:,1], return_counts=True)[0]) != 2 or len(np.unique(test[:,1], return_counts=True)[0]) != 2:
                print('   not enough samples in train or test')
            else:
                np.save('{}\\TUH\\features\\{}\\{}_{}_test'.format(drive, montage, feats_file, sz_type), test)
                np.save('{}\\TUH\\features\\{}\\{}_{}_train'.format(drive, montage, feats_file, sz_type), train)
            
        except Exception as e:
            print(e)
            
        
            
            
#%%

def main(argv):
    
    montage = '02'
    epoch = '10'
    from_files = True
    
    try:
        opts, args = getopt.getopt(argv, 'hm:e:f:',["montage=","epoch=", "from_files="])
    except getopt.GetoptError:
      sys.exit(2)
      
    for opt, arg in opts:
        if opt == '-h':
            print('train_test_split.py -m <montage> -e <epoch_length> -f <from_files>')
            sys.exit()
        elif opt in ("-m", "--montage"):
            montage = arg
        elif opt in ("-e", "--epoch"):
            epoch = arg
        elif opt in ("-f", "--from_files"):
            from_files = arg
        
    drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
    drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0] 
        
    feats_file = 'window_' + epoch
    
    labels = {8.: 'fnsz', 9.: 'gnsz', 10.: 'spsz', 11.: 'cpsz', 
              12.: 'absz', 13.: 'tnsz', 15.: 'tcsz', 17.: 'mysz'}


    data, best_combo = get_session_combo(drive=drive, montage=montage, feats_file=feats_file, labels=labels, from_count_files=from_files)
    train_test_split(data=data, best_combo=best_combo, drive=drive, montage=montage, feats_file=feats_file, labels=labels)
    

if __name__ == '__main__':
              
    main(sys.argv[1:])         
    