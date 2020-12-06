import numpy as np
import os

#%% Join different features (by channel)
path = 'TUH\\v1.5.1\\features\\train\\01_tcp_ar\\full_dataset\\'  
window = 10
overlap = 0

feat_type = np.unique([ol[:-14] for ol in os.listdir(path) if ol.endswith('.npy')])
feat_type = np.delete(feat_type, np.argwhere(feat_type == 'signal_stats'))

ss = np.load(path + 'signal_stats'
            + '_' + str(window) + '_' + str(overlap) + '_'
            + str(window) + '_' + str(overlap) + '.npy')

sm = np.load(path + 'spectral_moments'
            + '_' + str(window) + '_' + str(overlap) + '_'
            + str(window) + '_' + str(overlap) + '.npy')

nl = np.load(path + 'non_linear'
            + '_' + str(window) + '_' + str(overlap) + '_'
            + str(window) + '_' + str(overlap) + '.npy')

full_data = ss[:, :4]
ss = ss[:, 4:]
sm = sm[:, 4:]
nl = nl[:, 4:]

nb_feats = [int(ss.shape[1]/22), int(sm.shape[1]/22), int(nl.shape[1]/22)]    
for chn in range(22):
    full_data = np.hstack([full_data, ss[:, chn*nb_feats[0] : chn*nb_feats[0] + nb_feats[0]]])
    full_data = np.hstack([full_data, sm[:, chn*nb_feats[1] : chn*nb_feats[1] + nb_feats[1]]])
    aux = nl[:, chn*nb_feats[2] + nb_feats[2] - 1]
    full_data = np.hstack([full_data, np.reshape(aux, (aux.shape[0],1))])
    
full_data = full_data[~np.isnan(full_data[:,4:]).any(axis=1), :]
full_data = full_data[~np.isinf(full_data[:,4:]).any(axis=1), :]   

np.save(path + 'full_dataset', full_data)

#%% Channel selection
path = 'TUH\\v1.5.1\\features\\train\\01_tcp_ar\\full_dataset\\'  

xy = np.load(path + 'full_dataset.npy')

channels = ['FP1-F7', 'FP2-F8', 'F7-T3', 'F8-T4','T3-T5', 'T4-T6',
                'T5-O1', 'T6-O2','A1-T3', 'T4-A2', 'T3-C3', 'C4-T4',
                'C3-CZ', 'CZ-C4', 'FP1-F3', 'FP2-F4','F3-C3', 'F4-C4',
                'C3-P3', 'C4-P4', 'P3-O1', 'P4-O2']
nb_feats = 4
wanted = ['FP2-F8', 'F7-T3', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3', 
          'T4-A2', 'T3-C3', 'C3-CZ', 'CZ-C4', 'FP1-F3', 'C3-P3',
          'C4-P4', 'P3-O1', 'P4-O2']

chn_indx = [channels.index(chn) for chn in wanted]
chn_indx.sort()

new = xy[:, :4]
xy = xy[:, 4:]


for chn in chn_indx:
    new = np.hstack([new, xy[:, chn*nb_feats : chn*nb_feats + nb_feats]])

np.save(path + 'full_dataset4', new)

# full_dataset2: 'C3-P3', 'F8-T4', 'FP1-F3', 'FP1-F7'
# full_dataset3: 'FP1-F7', 'F8-T4', 'C3-P3', 'FP1-F3', 'F7-T3', 'T3-T5', 'T3-C3', 'T5-O1', 'P3-O1', 'T6-O2', 'P4-O2'
# full_dataset4: 'FP2-F8', 'F7-T3', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3',  'T4-A2', 'T3-C3', 'C3-CZ', 'CZ-C4', 'FP1-F3', 'C3-P3', 'C4-P4', 'P3-01', 'P4-O2'