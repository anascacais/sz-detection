from load_patient_data import load_patient_data
import subprocess
import json
import os


def main():
    
    datasets = ['train', 'dev']
    
    for dataset in datasets:
    
        drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
        drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0]
        
        montages = ['01_tcp_ar', '02_tcp_le', '03_tcp_ar_a']
    
        for montage in montages:
            
            if montage in ['01_tcp_ar', '03_tcp_ar_a']:
                m = '0103'
            else:
                m = '02'
                
            groups_dir = '{}\\tuszepi\\edf\\{}\\{}'.format(drive, dataset, montage)
            groups_dir_res = '{}\\tuszepi\\edf_resampled\\{}\\{}'.format(drive, dataset, montage)
            
            load_edf_data(groups_dir, groups_dir_res, save_dir='{}\\TUH\\data_npy\\{}'.format(drive, m))


# %% 
def load_edf_data(groups_dir, groups_dir_res, save_dir):

    for group in sorted([g for g in os.listdir(groups_dir) if os.path.isdir(os.path.join(groups_dir, g))]):
        print('-- group {} --'.format(group))
        group_dir = os.path.join(groups_dir, group)
        group_dir_res = os.path.join(groups_dir_res, group)
        
        for patient in sorted([f for f in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, f))]):
            #print('-- patient {} --'.format(patient))
            patient_dir = os.path.join(group_dir, patient)
            patient_dir_res = os.path.join(group_dir_res, patient)
                    
            load_patient_data(patient, patient_dir, patient_dir_res, save_dir=save_dir)

            
def check_files():
    
    drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
    drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0]
    
    montages = ['0103', '02']
    
    for montage in montages:
            
        check_dir = '{}\\TUH\\data_npy\\{}'.format(drive, montage)

        list_check_dir = os.listdir(check_dir)
        list_annotations = [lcd for lcd in list_check_dir if 'annotations' in lcd]
        list_npy = [lcd for lcd in list_check_dir if lcd.endswith('.npy')]
        sorted(list_annotations)
        for la in list_annotations:
            annotation = json.load(open(check_dir + os.sep + la, 'rb'))
            npy_files = list(annotation.keys())
            for npy in npy_files:
                if npy + '.npy' in list_npy:
                    list_npy.remove(npy + '.npy')
                else:
                    print('file not here')
    
        print(list_npy)


# %% 
if __name__ == '__main__':

    main()
    
    check_files()