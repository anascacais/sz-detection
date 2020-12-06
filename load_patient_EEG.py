
import pyedflib as pyedf
import pandas as pd
import os
import json
import rereferencing_resampling as rr
import datetime


def load_patient_data(patient_dir, save_dir):
    
    # patient_dir is directory of a patient and the root level has the sessions
    
    # Create annotations file for each session: as a dictionary  
    # key: file name
    # value: dict with events e.g.: ['bckg':[start, end], 'seiz': [start, end]]
    
    patient = os.path.basename(patient_dir)
    list_sessions = sorted([s for s in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, s))])
    
    for session in list_sessions:
        session_dir = os.path.join(patient_dir, session)
        session = session.split('_')[0]
        
        if os.path.exists(os.path.join(save_dir, patient+'_'+session)+'_annotations.txt'):
            with open(os.path.join(save_dir, patient+'_'+session)+'_annotations.txt', 'r') as annot_file:
                annotations = json.load(annot_file)
        else:
            annotations = dict()
        
        
        list_files = sorted([f for f in os.listdir(session_dir) if f.endswith('.edf')])
    
        for file in list_files:  
            file_path = os.path.join(session_dir, file)
            
            # Check if csv file already exists
            file_name = file[:-4]
            print('\t Loading file -- {}'.format(file_name))
            
            if not os.path.exists(os.path.join(save_dir, file_name+'.csv')):
                try:
                    edf = pyedf.EdfReader(file_path)
                except:
                    print('\t \t This file could not be open')
                    continue
                
                chn_labels = edf.getSignalLabels()
                
                # Transform signal in DataFrame; each column is a channel with time as index
                df = pd.DataFrame(columns=None)
                
                for chn in enumerate(chn_labels): #(0, 'EEG FP1-REF')       
                    # Select only EEG channels
                    if 'EEG' in chn_labels[chn[0]]:
                        signal = edf.readSignal(chn[0])
                        df[chn_labels[chn[0]]] = signal
                    
                fs = edf.getSampleFrequency(chn_labels.index(list(df)[0]))
                
                tcp_montage = patient_dir.split(os.sep)[-3]
                df = rr.rr_resampling(df, fs, tcp_montage)
                df.to_csv(os.path.join(save_dir, file_name+'.csv'), index=None, header=True)
                print('\t \t .csv file saved successfully!')
                
                # Save annotations for file
                file_start_time = edf.getStartdatetime()
                file_duration = datetime.timedelta(seconds=edf.getFileDuration())
                edf.close()
                hasAnnot = True
                
            else:
                print('\t \t .csv file already exists!')
                hasAnnot = False
                
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
        


def load_patient_data_eval(patient_dir, save_dir):

    # patient_dir is directory of a patient and the root level has the sessions

    # Create annotations file for the patient: as a dictionary
    # key: file name
    # value: dict with events e.g.: ['bckg':[start, end], 'seiz': [start, end]]

    list_sessions = [s + '\\' for s in os.listdir(patient_dir) if os.path.isdir(patient_dir + s)]

    for session in list_sessions:

        list_files = [f for f in os.listdir(patient_dir + session) if f.endswith('.edf')]

        for file in list_files:

            # Check if file already exists
            file_name = file[:-4]
            if not os.path.exists(save_dir + file_name + '.csv'):

                file_path = patient_dir + session + file

                try:
                    edf = pyedf.EdfReader(file_path)
                except:
                    print('This file could not be open')
                    edf = []

                chn_labels = edf.getSignalLabels()

                # Transform signal in DataFrame; each column is a channel with time as index
                df_temp = pd.DataFrame(columns=None)

                for chn in enumerate(chn_labels):  # (0, 'EEG FP1-REF')
                    # Select only EEG channels
                    if 'EEG' in chn_labels[chn[0]]:
                        signal_temp = edf.readSignal(chn[0])
                        df_temp[chn_labels[chn[0]]] = signal_temp

                fs = edf.getSampleFrequency(chn_labels.index(list(df_temp)[0]))

                edf.close()

                tcp_montage = patient_dir.split(os.sep)[5]

                df_temp = rr.rr_resampling(df_temp, fs, tcp_montage)
                df_temp.to_csv(save_dir + file_name + '.csv', index=None, header=True)
                print('Success! ')

