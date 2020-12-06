from load_patient_EEG import load_patient_data
#from load_patient_EEG import load_patient_data_eval
import json
import os


# %% For TUH database

montage = '01_tcp_ar'
dataset = 'train'

def load_edf_data(groups_dir, save_dir):

    for group in sorted([g for g in os.listdir(groups_dir) if os.path.isdir(os.path.join(groups_dir, g))]):
        group_dir = os.path.join(groups_dir, group)
        
        for patient in sorted([f for f in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, f))]):
            patient_dir = os.path.join(group_dir, patient)
            
            print('Loading patient -- ' + str(patient))
            
            if 'eval' in save_dir:
                load_patient_data_eval(patient_dir, save_dir=save_dir)
            else:
                load_patient_data(patient_dir, save_dir=save_dir)


def check_files():

    check_dir = 'E:\\TUH\\v1.5.1\\ref_data_csv\\eval\\01_tcp_ar'

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

load_edf_data(groups_dir='D:\\tuszepi\\edf\\{}\\{}'.format(dataset, montage), save_dir='D:\\TUH\\data_csv')
#print(check_files())

