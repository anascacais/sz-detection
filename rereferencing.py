import numpy as np


# %% Rereferencing to bipolar montage

# pairs = [['FP1', 'F7'], ['FP2', 'F8'], ['F7', 'T3'], ['F8', 'T4'],
#          ['T3', 'T5'], ['T4', 'T6'], ['T5', 'O1'], ['T6', 'O2'],
#          ['T3', 'C3'], ['C4', 'T4'],
#          ['C3', 'CZ'], ['CZ', 'C4'], ['FP1', 'F3'], ['FP2', 'F4'],
#          ['F3', 'C3'], ['F4', 'C4'], ['C3', 'P3'], ['C4', 'P4'],
#          ['P3', 'O1'], ['P4', 'O2']]

pairs = [['FP1', 'FP2']]

def rr(recordings, tcp_montage):

    for pair in pairs:
        if tcp_montage[3:9] == 'tcp_ar':
            rr_recordings = np.array(recordings['EEG '+pair[0]+'-REF'] - recordings['EEG '+pair[1]+'-REF'])

        elif tcp_montage[3:9] == 'tcp_le':
            rr_recordings = np.array(recordings['EEG '+pair[0]+'-LE'] - recordings['EEG '+pair[1]+'-LE'])
        else:
            print('Channels '+pair[0]+' and/or '+pair[1]+' do not exist.')
            
    # if tcp_montage[3:9] == 'tcp_ar':        
    #     rr_recordings['FP1-REF'] = recordings['EEG FP1-REF']
    #     rr_recordings['FP2-REF'] = recordings['EEG FP2-REF']
    # else:
    #     rr_recordings['FP1-REF'] = recordings['EEG FP1-LE']
    #     rr_recordings['FP2-LE'] = recordings['EEG FP2-LE']

    return rr_recordings

