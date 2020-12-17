import os
import subprocess
import json
import pickle

drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0]

montages = ['0103', '02']

# saves the files whose sessions include tcsz
tcsz_files = []

for montage in montages:
    npy_data_dir = '{}\\TUH\\data_npy\\{}'.format(drive, montage)
    
    list_files = sorted([f for f in os.listdir(npy_data_dir) if f.endswith('.npy')])
    
    annotation_files = sorted([f for f in os.listdir(npy_data_dir) if f.endswith('.txt')])
    
    for annotation_file in annotation_files:
        
        session = annotation_file[:-16]
        
        with open(os.path.join(npy_data_dir, annotation_file), 'r') as annot_file:
            annotations = json.load(annot_file)
        
        list_rec = [list(annotations[ses].keys()) for ses in annotations.keys()]
        if 'tcsz' in [event for ses in list_rec for event in ses]:
            tcsz_files += [f for f in list_files if session in f]
             
    
with open('tcsz_files', 'wb') as fp:
    pickle.dump(tcsz_files, fp)
