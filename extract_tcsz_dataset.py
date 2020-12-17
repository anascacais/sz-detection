
from feature_extr import get_features
import subprocess
import numpy as np
import os
import json
import _biosppy.biosppy as bsp
import pickle


def szInSession(annotations):
    
    sz_starts = []
    sz_ends = []
    
    eventsInSession = [list(annotations[ses].keys()) for ses in annotations.keys()]
    keys = [key for key in annotations.keys()]
    
    for i, file in enumerate(eventsInSession):
        file = list(filter(lambda a: a != 'bckg', file))
        file = list(filter(lambda a: a != 'file_end', file))
    
        for key in file:
            sz_starts += [annotations[keys[i]][key][k] for k in range(0, len(annotations[keys[i]][key]), 2)]
            sz_ends += [annotations[keys[i]][key][k] for k in range(1, len(annotations[keys[i]][key]), 2)]
        
    return sz_starts, sz_ends

    

def main(seg_window, feat_types=['non_linear', 'dwt'], seg_overlap=0, fs=250): 
    
    
    drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
    drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0]
        
    montages = ['0103', '02']
    
    for montage in montages:
        
        npy_data_dir = '{}\\TUH\\data_npy\\{}'.format(drive, montage)
        save_dir = '{}\\TUH\\features\\{}'.format(drive, montage)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #list_files = sorted([f for f in os.listdir(npy_data_dir) if f.endswith('.npy')])
        with open ('tcsz_files', 'rb') as fp:
            list_files = pickle.load(fp) 
        
        # create dict with possible labels
        labels = {'(null)': 0, 'spsw': 1, 'gped': 2, 'pled': 3, 'eybl': 4, 'artf': 5, 
                  'bckg': 6, 'seiz': 7, 'fnsz': 8, 'gnsz': 9, 'spsz': 10, 'cpsz': 11, 
                  'absz': 12, 'tnsz': 13, 'cnsz': 14, 'tcsz': 15, 'atsz': 16, 'mysz': 17, 
                  'nesz': 18, 'intr': 19, 'slow': 20, 'eyem': 21, 'chew': 22, 'shiv': 23, 
                  'musc': 24, 'elpp': 25, 'elst': 26, 'calb': 27}         
        
            
        for i,file in enumerate(list_files): 
            
            print('\t Loading file -- {}: {}/{}'.format(file, i+1, len(list_files)))
    
            file_name = file[:-4]
            file_path = os.path.join(npy_data_dir, file)  
            unfiltered = np.load(file_path)
            filt = bsp.signals.eeg.eeg(signal=np.reshape(unfiltered, (-1,1)), sampling_rate=250., show=False)
            recording = filt['filtered'][:,0]
            list_file_name = file_name.split('_')
            patient = list_file_name[0]
            session = list_file_name[1]
            int_file_name = int(list_file_name[0][3:] + list_file_name[1][1:] + list_file_name[2][1:])
            
            try:
                features = np.load(os.path.join(save_dir, 'window_'+''.join(str(seg_window).split('.'))+'.npy'))
            except:
                features = np.array([])
                
            if len(features) > 0 and int_file_name in features[:, 0]:
                print('\t \t This file is name already in features!')
                continue
        
            with open(os.path.join(npy_data_dir, patient+'_'+session)+'_annotations.txt', 'r') as annot_file:
                annotations = json.load(annot_file)
            
            list_sessions = [list(annotations[ses].keys()) for ses in annotations.keys()]
            if 'tcsz' not in [event for ses in list_sessions for event in ses]:
                print('---- no tcsz in session ----')
            else:
                sz_starts, sz_ends = szInSession(annotations)
            
                try:
                    annotations = annotations[file_name]
                except:
                    print('\t \t No annotations for ' + file_name)
                    annotations = {}
                    
                if annotations != {}:
                    
                    if 'file_end' in annotations:
                        del annotations['file_end']
                    
                    nb_intervals = int(sum([len(annotations[a])/2 for a in annotations]))
                    for n in range(nb_intervals):
            
                        annot_type = min(annotations.keys(), key=(lambda k: annotations[k]))
                        
                        print('\t \t \t extracting features for annotation ' + annot_type + '...' )
                        print('\t \t \t from sample ' + str(int(annotations[annot_type][0]*fs)) + ' to ' + str(int(annotations[annot_type][1]*fs)))
                        
                        segment = recording[int(annotations[annot_type][0]*fs):int(annotations[annot_type][1]*fs)]
                        
                        if len(segment) >= fs*seg_window:
                            feature_vector = get_features(signal=segment, feat_types=feat_types,
                                                          sampling_rate=fs, seg_window=seg_window,
                                                          seg_overlap=seg_overlap)
                        
    
                            # Add timeAFTsz and time2sz if applicable
                            # if annot_type == 'bckg':
                            #     bg = annotations['bckg'][0:2]
                            #     bg_time = bg[1]-bg[0]
                            #     nsamples = int(bg_time/(seg_window-seg_overlap))
                                
                            #     if len(sz_starts) == 0:
                            #         new_col = np.empty((feature_vector.shape[0], 2))
                            #         new_col[:] = np.nan
                            #         feature_vector = np.hstack((new_col, feature_vector))
                                    
                            #     else:
                            #         if sorted(sz_starts + [bg[0]]).index(bg[0]) == 0: #there's only a sz after
                                        
                            #             # timeAFTsz column
                            #             new_col = np.empty((feature_vector.shape[0], 1))
                            #             new_col[:] = np.nan
                            #             feature_vector = np.hstack((new_col, feature_vector))
                                        
                            #             # time2sz column
                            #             time2sz = sz_starts[0] - bg[0] - seg_window
                            #             new_col = np.empty((feature_vector.shape[0], 1))
                            #             new_col[:] = np.nan
                                    
                            #             for k in range(nsamples):
                            #                 # Add time2sz to first sample
                            #                 new_col[k] = time2sz
                            #                 # Set time for next segment
                            #                 time2sz = time2sz - (seg_window - seg_overlap)
                            #             feature_vector = np.hstack((new_col, feature_vector))
                                        
                            #         elif sorted(sz_starts + [bg[0]]).index(bg[0]) == len(sz_starts): #there's only a sz before
                                        
                            #             # timeAFTsz column
                            #             timeAFTsz = bg[0] - sz_ends[0] + seg_window
                            #             new_col = np.empty((feature_vector.shape[0], 1))
                            #             new_col[:] = np.nan
                                        
                            #             for k in range(nsamples):
                            #                 new_col[k] = timeAFTsz
                            #                 timeAFTsz = timeAFTsz + (seg_window - seg_overlap)
                            #             feature_vector = np.hstack((new_col, feature_vector))
                                        
                            #             # time2sz column
                            #             new_col = np.empty((feature_vector.shape[0], 1))
                            #             new_col[:] = np.nan
                            #             feature_vector = np.hstack((new_col, feature_vector))
                                        
                            #         else: #there's a sz before and after
                                        
                            #             # timeAFTsz column
                            #             ind = sorted(sz_ends + [bg[1]]).index(bg[1])
                            #             timeAFTsz = bg[0] - sz_ends[ind-1] + seg_window
                            #             new_col = np.empty((feature_vector.shape[0], 1))
                            #             new_col[:] = np.nan
                                    
                            #             for k in range(nsamples):
                            #                 new_col[k] = timeAFTsz
                            #                 timeAFTsz = timeAFTsz + (seg_window - seg_overlap)
                            #             feature_vector = np.hstack((new_col, feature_vector))
                                        
                            #             # time2sz column
                            #             ind = sorted(sz_starts + [bg[0]]).index(bg[0])
                            #             time2sz = sz_starts[ind+1] - bg[0] - seg_window
                            #             new_col = np.empty((feature_vector.shape[0], 1))
                            #             new_col[:] = np.nan
                                    
                            #             for k in range(nsamples):
                            #                 new_col[k] = time2sz
                            #                 time2sz = time2sz - (seg_window - seg_overlap)
                            #             feature_vector = np.hstack((new_col, feature_vector))
                                        
                            # else:
                            #     sz = annotations[annot_type][0:2]
                            #     sz_time = sz[1]-sz[0]
                            #     nsamples = int(sz_time/(seg_window-seg_overlap))
            
                                
                            #     new_col = np.empty((feature_vector.shape[0], 2))
                            #     new_col[:] = np.nan
                            #     feature_vector = np.hstack((new_col, feature_vector))
                        
                        else:
                            print('\t \t \t Segment length is not enough = {} samples'.format(len(segment)))
                            feature_vector = None        
                                
                                
                        # remove interval from annotations
                        annotations[annot_type] = annotations[annot_type][2:]
                        if len(annotations[annot_type]) == 0:
                            del annotations[annot_type]       
                        
                        # add feature vector of segment to features array
                        if feature_vector is not None:
                            # Add label as column
                            feature_vector = np.vstack(
                                (np.ones([1, feature_vector.shape[0]]) * labels[annot_type], feature_vector.T)).T
                            # add file name as column
                            feature_vector = np.vstack(
                                (np.ones([1, feature_vector.shape[0]]) * int_file_name, feature_vector.T)).T
                            
                            if features.size == 0:
                                features = feature_vector
                            else:
                                features = np.concatenate((features, feature_vector), axis=0)
                        
                            np.save(os.path.join(save_dir, 'tcsz_window_'+''.join(str(seg_window).split('.'))), features)
                        
# %%                            
if __name__ == '__main__':
    
    # durations = [0.25, 0.5, 1., 2., 3., 5.]
    # for duration in durations:
    #     main(seg_window=duration)            
    main(seg_window=1.)                