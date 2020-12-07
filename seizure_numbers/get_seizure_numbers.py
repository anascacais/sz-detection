import json
import os
import subprocess


def get_annotations(file_path):
    
    with open(file_path, 'rt') as file:    
        annotations = file.read()
        annotations = annotations.split('\n')
        annotations = list(filter(None, annotations))[1:] # returns annotations as a list of strings
                            
    return [annot.split(' ') for annot in annotations] # transforms list of strings into list of lists 

def get_seizure_numbers(drive, montages, dataset):
    
    for montage in montages:
        print('                  --- {} ---'.format(montage))
        groups_dir = '{}\\tuszepi\\edf\\{}\\{}'.format(drive, dataset, montage)
        
        for group in sorted([g for g in os.listdir(groups_dir) if os.path.isdir(os.path.join(groups_dir, g))]):
            group_dir = os.path.join(groups_dir, group)
            
            
            for patient in sorted([f for f in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, f))]):
                #print('-- patient {} --'.format(patient))
                patient_dir = os.path.join(group_dir, patient)
                
                for session in sorted([s for s in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, s))]):
                    session_dir = os.path.join(patient_dir, session)
                    
                    try:
                        with open(montage+'_'+dataset+'_seizure_numbers.txt') as json_file:
                            seizures = json.load(json_file)
                    except:
                        seizures = {}
                    
                    tse_files = [f for f in os.listdir(session_dir) if f.endswith('.tse')]
                    
                    for tse_file in tse_files:
                        
                        #print('    --- in file {} ---'.format(os.path.basename(tse_file)))
                        
                        try: 
                            annotations = get_annotations(os.path.join(session_dir, tse_file)) 
                                
                            for annot in annotations:
                                sz_duration = float(annot[1]) - float(annot[0])
                                try: 
                                    print('{}: {}'.format(annot[2],seizures[annot[2]]))
                                    seizures[annot[2]] = [seizures[annot[2]][0]+1, seizures[annot[2]][1]+sz_duration]
                                except:
                                    seizures[annot[2]] = [1, sz_duration]
                                
                        except Exception as e:
                            print(e)
                            print('\n')
                            print('    --- in file {} ---'.format(os.path.basename(tse_file)))
                            pass
                    
                    with open(montage+'_'+dataset+'_seizure_numbers.txt', 'w') as outfile: # save updated file after each session
                        json.dump(seizures, outfile)
                    

def main():
    
    drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
    drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0]
    
    montages = ['01_tcp_ar', '02_tcp_le', '03_tcp_ar_a']
    dataset = 'dev'
    
    get_seizure_numbers(drive, montages, dataset)

if __name__ == '__main__':

    main()