import subprocess
import os
import json

#%% Verify if, for each session, there's oly one type of crisis

isTrue = True 
whenIsNotValid = []

montages = ['0103', '02']

for montage in montages:
    
        drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
        drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0]
        
        annot_dir = '{}\\TUH\\data_npy\\{}'.format(drive, montage)
        
        for annot_file in sorted([f for f in os.listdir(annot_dir) if f.endswith('.txt')]):
            
            print('in session {}'.format(annot_file[:-16]))
            
            with open(os.path.join(annot_dir, annot_file), 'r') as afile:
                annotations = json.load(afile)
                
                annots = []
                for record in annotations:
                    annots += list(annotations[record].keys())
                
                
            if len(set(annots)) > 3: # gets unique occurances in list
                isTrue = False
                whenIsNotValid += [list(set(annots))]
                print('---- not valid ----')
                print('          {}'.format(set(annots)))

print(isTrue)

