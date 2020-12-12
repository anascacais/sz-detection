import pyedflib as pyedf
import pandas as pd
import os
import json
import rereferencing as rr
import datetime
import numpy as np

def load_patient_data(patient, patient_dir, patient_dir_res, save_dir):

        
    for session in sorted([s for s in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, s))]):
        session_dir = os.path.join(patient_dir, session)
        session_dir_res = os.path.join(patient_dir_res, session)
        session = os.path.basename(session_dir).split('_')[0]
        
        # Get existing annotations
        if os.path.exists(os.path.join(save_dir, patient+'_'+session)+'_annotations.txt'):
            with open(os.path.join(save_dir, patient+'_'+session)+'_annotations.txt', 'r') as annot_file:
                annotations = json.load(annot_file)
        else:
            annotations = dict()
            
            
        list_files = sorted([f for f in os.listdir(session_dir) if f.endswith('.edf')])

        for file in list_files:  
            file_path = os.path.join(session_dir_res, file)
            
            # Check if npy file already exists
            file_name = file[:-4]
            print('\t Loading file -- {}'.format(file_name))
            if not os.path.exists(os.path.join(save_dir, file_name+'.npy')):
                try:
                    edf = pyedf.EdfReader(file_path)
                except:
                    print('\t \t This file could not be open')
                    continue
                
                chn_labels = edf.getSignalLabels()
                
                # Transform signal to DataFrame; each column is a channel with time as index
                df = pd.DataFrame(columns=None)
                
                for chn in enumerate(chn_labels): #(0, 'EEG FP1-REF')       
                    # Select only EEG channels
                    if 'EEG' in chn_labels[chn[0]]:
                        signal = edf.readSignal(chn[0])
                        df[chn_labels[chn[0]]] = signal
                        
                # get tcp_montage and get chosen channels
                tcp_montage = patient_dir.split(os.sep)[-3]
                rr_array = rr.rr(df, tcp_montage)
                np.save(os.path.join(save_dir, file_name), rr_array)
                print('\t \t .npy file saved successfully!')
                
                # Save annotations for file
                file_start_time = edf.getStartdatetime()
                file_duration = datetime.timedelta(seconds=edf.getFileDuration())
                edf.close()
                hasAnnot = True
                
            else:
                print('\t \t .npy file already exists!')
                hasAnnot = False
                
            #### Annotations
            if file_name not in annotations:
                # Make annotations files (try for multi-class first)
                if not hasAnnot:
                    try:
                        edf = pyedf.EdfReader(file_path)
                    except:
                        print('\t \t This file could not be open')
                        continue
                    
                    file_start_time = edf.getStartdatetime()
                    file_duration = datetime.timedelta(seconds=edf.getFileDuration())
                    edf.close()
                
                try:
                    with open (os.path.join(session_dir, file_name+'.tse'), 'rt') as file:    
                        og_annot = file.read()
                        og_annot = og_annot.split('\n')
                        og_annot = list(filter(None, og_annot))
                except:
                    with open (os.path.join(session_dir, file_name+'.tse_bi'), 'rt') as file:
                        og_annot = file.read()
                        og_annot = og_annot.split('\n')
                        og_annot = list(filter(None, og_annot))
                

                if len(annotations) == 0: #first file in session
                    start = 0.
                else:
                    prev_file = sorted(annotations.keys())[-1]
                    prev_file_end = annotations[prev_file]['file_end']
                    distance = file_start_time - datetime.datetime.strptime(prev_file_end, '%Y/%m/%d, %H:%M:%S')
                    del annotations[prev_file]['file_end']
                    start = max(list(annotations[prev_file].values()))[-1] + distance.total_seconds()
                
                annotations[file_name] = {}
                file_end = file_start_time + file_duration
                annotations[file_name]['file_end'] = file_end.strftime("%Y/%m/%d, %H:%M:%S")
                
                for i in range(1, len(og_annot)):
                    event_annot = og_annot[i].split(' ')
                    event_start = start + float(event_annot[0])
                    event_end = start + float(event_annot[1])
                    
                    try:
                        annot_old = annotations[file_name][event_annot[2]]
                    except KeyError:
                        annot_old = []

                    annot_old.extend([event_start, event_end])
                    annotations[file_name][event_annot[2]] = annot_old

                hasAnnot = False
                
        # Save annotations in json file  
        with open(os.path.join(save_dir, patient+'_'+session)+'_annotations.txt', 'w') as file:
            json.dump(annotations, file)
        
        print('\t \t Annotations file created for session -- {}!'.format(session))
            
            
            
            
            
            