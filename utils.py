
import numpy as np
from collections import Counter
from scipy import stats, signal
import time
import pywt


def hurst_exponent(list_values):
    R = abs(np.max(list_values) - np.mean(list_values) - \
        abs(np.min(list_values)-np.mean(list_values)))


    return np.log10(R/np.std(list_values))/np.log10(len(list_values))


def non_linear_features(list_values):
    list_values = list_values.values #transform df into array
    #
    # #Sample Entropy duration ~ 0.25s
    # #sae = univariate.samp_entropy(list_values, 2, 0.2 * np.std(list_values))
    #
    # #Spectral Entropy duration ~0.0s
    # spe = univariate.spectral_entropy(list_values, 250.)
    #
    # # Perm Entropy
    # pme = e.entropy.perm_entropy(list_values)
    #
    # # SVD Entropy duration ~0.0s
    # sve = univariate.svd_entropy(list_values)
    #
    # # Approximate Entropy duration ~0.03s
    # #ape = e.entropy._app_samp_entropy(list_values, order=2)
    #
    # # Higuchi Fractal Dimension duration ~0.4s
    # #hfd_ = e.fractal._higuchi_fd(list_values, 10)
    #
    # #PFD duration ~ 0.009s
    # pfd_ = univariate.pfd(list_values)
    #
    # #Hjorth duration ~0.002s returns activity, morbidity, complexity
    # hj_a, hj_m, hj_c = univariate.hjorth(list_values)

    # Hurst Exponent Entropy
    he = hurst_exponent(list_values)
    #uni_he = univariate.hurst(list_values)

    return [he]

def spectral_moments(list_values, sampling_rate):
    freqs, psd = signal.welch(x=list_values, fs=sampling_rate, nperseg=1024, return_onesided=True)

    centroid = np.sum(freqs * psd) / np.sum(psd)

    var_coeff = np.sum(np.square(freqs - centroid) * psd) / np.sum(psd)

    temp = (freqs - centroid) / np.sqrt(var_coeff)
    spec_skew = np.sum(np.power(temp, 3 * np.ones(temp.shape)) * psd) / np.sum(psd)

    return [spec_skew]


def signal_stats(list_values):
    mean = np.nanmean(list_values)
    #std = np.nanstd(list_values)
    #kurt = stats.kurtosis(list_values)
    skew = stats.skew(list_values)
    #rms = np.nanmean(np.sqrt(list_values ** 2))

    # Hjorth parameters:
    # activity, complexity, morbidity = univariate.hjorth(signal)

    return [mean, skew]

def dwt_spectral(signal):
    dwt = []
    
    #sub-band extraction f = [0, 4], [4, 8], [8, 16], [16, 31], [31, 62.5], [62.5, 125] Hz
    coeffs = pywt.wavedec(signal, 'db4', level=5)
    
    energies = np.array([])
    for coeff in coeffs: # energy of each level
        energies = np.append(energies, np.sqrt(np.sum(np.array(coeff)**2))/len(coeff))
        
    tot_energy = np.sum(energies)
    rwenergy = list(energies / tot_energy) # relative wavelet energy
    dwt += rwenergy
    
    # entropies = np.array([])
    # for coeff in coeffs:
    #     entropies = np.append(entropies, calculate_entropy(coeff))
    
    # tot_entropy = np.sum(entropies)
    # rwentropy = list(entropies / tot_entropy) # relative wavelet entropy
    # dwt += rwentropy
    
    return dwt

def pearson(signal):
    #chns = signal.columns
    #chn_pairs = combinations(chns, 2)
    feature_vector = np.array([])
    chn_pairs = []
    
    for pair in chn_pairs:
        s1 = signal[pair[0]]
        s2 = signal[pair[1]]
        feature_vector = np.append(feature_vector, stats.pearsonr(s1, s2))
    
    return feature_vector


def compute_seg_feat(segment=None,
                     sampling_rate=250,
                     feat_type=None,
                     feat_window=10, 
                     feat_overlap=0):
    
    """ Computes features specified by feat_type, based on a sliding 
    window approach, and returns the resulting features placed consecutively in 
    a single feature vector.
    
    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    feat_type : string
        Features to extract; 'dwt' , 'signal stats', 'spectral moments'.
    feat_window : int, float
        Size of window to compute features, in seconds.
    feat_overlap : int, float, optional
        Time to overlap between windows; defaults to 0.
        
    Returns
    -------
    feature_vector : array
        Feature vector of segment, with features from consecutive windows
        placed consecutively.
    """
    if feat_type == 'dwt':
        feature_vector = dwt_spectral(segment)
            
    if feat_type == 'signal_stats':
        feature_vector = signal_stats(segment) # array with 8 features per segment

    if feat_type == 'non_linear':
        feature_vector = non_linear_features(segment)

    if feat_type == 'spectral_moments':
        feature_vector = spectral_moments(segment, sampling_rate) # array with 3 features per segment
        
    return np.array(feature_vector)

def get_features(signal=None,
                 sampling_rate=250,
                 feat_type=None,
                 seg_window=10,
                 seg_overlap=0,
                 feat_window=10,
                 feat_overlap=0):
    
    """ Segments signal into epochs with length seg_window, based on a sliding 
    window approach. The feature vectors of channels of the signal are placed 
    consecutively in a single feature vector, in the order 
    
    Parameters
    ----------
    signal : array
        Input signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    feat_type : string
        Features to extract; 'dwt' , 'signal stats', 'spectral moments'.
    seg_window : int, float
        Size of window to segment signal and extract feature vector.
    seg_overlap : int, float, optional    
        Time to overlap between segments; defaults to 0.
    feat_window : int, float
        Size of window to compute features, in seconds.
    feat_overlap : int, float, optional
        Time to overlap between segments; defaults to 0.
        
    Returns
    -------
    features : array
    
    """
    
    features = np.array([])
    
    channels = ['FP1-F7', 'FP2-F8', 'F7-T3', 'F8-T4','T3-T5', 'T4-T6', 
                'T5-O1', 'T6-O2', 'T3-C3', 'C4-T4',
                'C3-CZ', 'CZ-C4', 'FP1-F3', 'FP2-F4','F3-C3', 'F4-C4', 
                'C3-P3', 'C4-P4', 'P3-O1', 'P4-O2']
    
    window = seg_window*sampling_rate
    overlap = seg_overlap*sampling_rate
    
    isample = 0
    # Each window corresponds to the analysis of a segment with size=window
    start_time = time.time()
    while isample+window <= len(signal):
        #print('Progression inside signal ' + str(np.round(100*isample/len(signal),2)) + '%')

        segment = signal[isample:isample+window]
        feat_segment = np.array([])
        
        # Gets feature vector for each channel (in the assigned order)
        for chn in channels:
            signal_chn = segment[chn]
            # Compute feature vector for segment of channel chn
            feat_chn = compute_seg_feat(segment=signal_chn, sampling_rate=sampling_rate,
                                        feat_type=feat_type, feat_window=feat_window, 
                                        feat_overlap=feat_overlap)
            feat_chn = np.array(feat_chn)
            feat_chn = feat_chn.reshape([1, feat_chn.shape[0]])
            
            if feat_segment.size == 0:
                feat_segment = feat_chn
            else:
                # Add chn feature vector along the main segment feature vector
                feat_segment = np.concatenate((feat_segment, feat_chn), axis=1)
                
        if feat_type == 'pearson':
            feat_segment = pearson(segment)
        
        feat_segment = np.array(feat_segment)
        if features.size == 0:
            features = feat_segment
        else:
            # Add feature vector of whole segment to the next row of features
            features = np.concatenate((features, feat_segment), axis=0)
        
        isample += window - overlap
    print("--- %s seconds for all signal ---" % (time.time() - start_time))
    return features

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy = stats.entropy(probabilities)
    return entropy
 
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]
 
def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]


