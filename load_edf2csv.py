from load_patient_data import load_patient_data
import subprocess
import json
import os


def main():
    
    montage = '01_tcp_ar'
    dataset = 'train'
    
    drives = ['%s:' % d for d in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if os.path.exists('%s:' % d)]
    drive = [d for d in drives if subprocess.check_output(["cmd","/c vol "+d]).decode().split("\r\n")[0].split(" ").pop()=='Passport'][0]
    
    groups_dir = '{}\\tuszepi\\edf\\{}\\{}'.format(drive, dataset, montage)
    groups_dir_res = '{}\\tuszepi\\edf_resampled\\{}\\{}'.format(drive, dataset, montage)
    
    load_edf_data(groups_dir, groups_dir_res, save_dir='{}\\TUH\\data_csv'.format(drive))
    #print(check_files())


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

            
def check_files(check_dir):

    #check_dir = 'E:\\TUH\\v1.5.1\\ref_data_csv\\eval\\01_tcp_ar'

    list_check_dir = os.listdir(check_dir)
    list_annotations = [lcd for lcd in list_check_dir if 'tse_annotations' in lcd]
    list_csv = [lcd for lcd in list_check_dir if lcd.endswith('.csv')]
    sorted(list_annotations)
    for la in list_annotations:
        annotation = json.load(open(check_dir + os.sep + la, 'rb'))
        csv_files = list(annotation.keys())
        for csv in csv_files:
            if csv+'.csv' in list_csv:
                list_csv.remove(csv+'.csv')
            else:
                print('file not here')

    return(list_csv)


# %% 
if __name__ == '__main__':

    main()
