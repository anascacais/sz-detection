import numpy as np
import subprocess
import os

#%% Separate bckg and sz based on seizure type

def main():
    drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
    drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0] 
    
    montage = '0103'    
    feats_file = 'window_50'
    
    data = np.load('{}\\TUH\\features\\{}\\{}.npy'.format(drive, montage, feats_file))
    
    labels = {8.: 'fnsz', 9.: 'gnsz', 10.: 'spsz', 11.: 'cpsz', 
              12.: 'absz', 13.: 'tnsz', 15.: 'tcsz', 17.: 'mysz'}
    
    for sz_type in labels.keys():
        
        print('getting samples for seizure type --{}'.format(labels[sz_type]))
        
        file_name = feats_file + '_{}'.format(labels[sz_type])
        try:
            samples = np.load('{}\\TUH\\features\\0103\\{}.npy'.format(drive, file_name))
            sessions_done = [f[:-3] for f in map(str, map(int, np.unique(samples[:,0])))]
    
        except:
            samples = np.array([])
            sessions_done = []
        
        sz_index = np.reshape(np.argwhere(data[:,1] == sz_type), (-1,))
        sz_samples = data[sz_index,:]
        
        for file in sz_samples[:,0]:
            
            session = str(int(file))[:-3]
            print('-- session {} --'.format(session))
            if session not in sessions_done:
                bg_index = np.reshape(np.argwhere(np.array([f[:-3] for f in map(str, map(int, data[:,0]))]) == session), (-1,))
                bg_samples = data[bg_index,:]
                print(np.unique(bg_samples[:,1]))
                bg_index2 = np.reshape(np.argwhere(bg_samples[:,1] == 6.), (-1,))
                bg_samples2 = bg_samples[bg_index2]
                print(np.unique(bg_samples2[:,1]))
                
                # add to samples
                if samples.size == 0:
                    samples = sz_samples
                else:
                    samples = np.append(samples, sz_samples, axis=0)
                
                samples = np.append(samples, bg_samples2, axis=0)
                
                # add session to sessions_done
                sessions_done += [session]
            
            else:  
                print('---- session already done ----')
            
            # save after each session
            np.save('{}\\TUH\\features\\0103\\{}'.format(drive, file_name), samples)
            
# %% 
if __name__ == '__main__':

    main()
