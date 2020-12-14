
import numpy as np
from collections import Counter
from scipy import stats
import pywt
from entropy import sample_entropy, higuchi_fd


def hurst_exponent(list_values):
    R = abs(np.max(list_values) - np.mean(list_values) - \
        abs(np.min(list_values)-np.mean(list_values)))

    return np.log10(R/np.std(list_values))/np.log10(len(list_values))


def non_linear_features(x):
    
    #Sample Entropy duration ~ 0.25s
    try:
        sae = sample_entropy(x, order=2, metric='chebyshev')
    except Exception as e:
        print(e)
        sae = np.nan
    
    # Higuchi Fractal Dimension duration ~0.4s
    try: 
        hfd_ = higuchi_fd(x, kmax=10)
    except Exception as e:
        print(e)
        hfd_ = np.nan

    # Hurst Exponent Entropy
    try:
        he = hurst_exponent(x)
    except Exception as e:
        print(e)
        he = np.nan

    return [sae, hfd_, he]


def signal_stats(list_values):
    try:
        mean = np.nanmean(list_values)
    except Exception as e:
        print(e)
        mean = np.nan
    try:
        std = np.nanstd(list_values)
    except Exception as e:
        print(e)
        std = np.nan
    try:
        kurt = stats.kurtosis(list_values)
    except Exception as e:
        print(e)
        kurt = np.nan
    try:
        skew = stats.skew(list_values)
    except Exception as e:
        print(e)
        skew = np.nan

    return [mean, std, skew, kurt]

def dwt(signal):
    
    w = pywt.Wavelet('db4')
    max_lvl = pywt.dwt_max_level(len(signal), w.dec_len)
    
    if max_lvl > 5:
        dec_lvl = 5
    else:
        dec_lvl = max_lvl
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.title('original')
    # plt.plot(signal)
        
    try:
        energies = np.array([])
        for i in np.arange(2, dec_lvl+1):
            
            D = pywt.downcoef('d', signal, w, level=i)
            x = pywt.upcoef('d', D, w, level=i)
            energies = np.append(energies, np.dot(x.T, x)) # energy of sub-band
            
            # plt.figure()
            # plt.title('reconstruction - coeff D{}'.format(i))
            # plt.plot(x)
            
            if i == dec_lvl:
                A = pywt.downcoef('a', signal, w, level=i)
                x = pywt.upcoef('a', A, w, level=i)
                energies = np.append(energies, np.dot(x.T, x))
                
                # plt.figure()
                # plt.title('reconstruction - coeff A{}'.format(i))
                # plt.plot(x)
        
        tot_energy = np.sum(energies)
        rwenergy = list(energies / tot_energy) # relative wavelet energy
        dwt = rwenergy
        
    except Exception as e:
        print(e)
        if max_lvl == 3:
            rwenergy = np.empty((3,))*np.nan
        elif max_lvl == 4:
            rwenergy = np.empty((4,))*np.nan
        else:
            rwenergy = np.empty((5,))*np.nan
        dwt = rwenergy
    
    try:
        #sub-band extraction f = [0, 3.91], [3.91, 7.812], [7.812, 15.625], [15.625, 31.25], [31.25, 62.5], [62.5, 125] Hz
        coeffs = pywt.wavedec(signal, 'db4', level=dec_lvl) #[cA5, cD5, cD4, cD3, cD2, cD1], n=dec_lvl
        
        
        means = np.array([])
        stds = np.array([])
        kurts = np.array([])
        skews = np.array([])
        
        for i in np.arange(len(coeffs)-1):
            
            ss = signal_stats(coeffs[i]) 
            means = np.append(means, ss[0])
            stds = np.append(stds, ss[1])
            kurts = np.append(kurts, ss[2])
            skews = np.append(skews, ss[3])
            
        dwt = np.append(dwt, means)
        dwt = np.append(dwt, stds)
        dwt = np.append(dwt, kurts)
        dwt = np.append(dwt, skews)
        
    except Exception as e:
        print(e)
        if max_lvl == 3:
            dwt = np.append(dwt, np.empty((12,))*np.nan)
        elif max_lvl == 4:
            dwt = np.append(dwt, np.empty((16,))*np.nan)
        else:
            dwt = np.append(dwt, np.empty((20,))*np.nan)
    
    return dwt


def compute_seg_feat(segment=None,
                     sampling_rate=250,
                     feat_types=None,
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
    feat_types : list
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
    
    feature_vector = np.array([]) 
    
    if 'dwt' in feat_types:
        f = dwt(segment)
        if feature_vector.size == 0:
            feature_vector = f
        else:
            feature_vector = np.append(feature_vector, f)
            
    if 'signal_stats' in feat_types:
        f = signal_stats(segment)
        if feature_vector.size == 0:
            feature_vector = f
        else:
            feature_vector = np.append(feature_vector, f)

    if 'non_linear' in feat_types:
        f = non_linear_features(segment)
        if feature_vector.size == 0:
            feature_vector = f
        else:
            feature_vector = np.append(feature_vector, f)

    return np.array(feature_vector)

def get_features(signal=None,
                 sampling_rate=250,
                 feat_types=None,
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
    feat_types : list
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
    
    window = int(seg_window*sampling_rate)
    overlap = seg_overlap*sampling_rate
    
    isample = 0
    
    # Each window corresponds to the analysis of a segment with size=window
    while isample+window <= len(signal):
        print('Progression inside signal ' + str(np.round(100*isample/len(signal),2)) + '%')

        segment = signal[isample:isample+window]
        
        # Compute feature vector for segment
        feat_segment = compute_seg_feat(segment=segment, sampling_rate=sampling_rate,
                                        feat_types=feat_types, feat_window=feat_window, 
                                        feat_overlap=feat_overlap)
        
        feat_segment = np.reshape(np.array(feat_segment), (1,-1))
        
        if features.size == 0:
            features = feat_segment
        else:
            # Add feature vector of whole segment to the next row of features
            features = np.concatenate((features, feat_segment), axis=0)
        
        isample += window - overlap

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


