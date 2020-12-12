import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import _biosppy.biosppy as bsp

data = pd.read_csv('00000002_s001_t000.csv')

plt.figure(figsize=(5, 4))
y1 = data['FP1-FP2'][0:int(250*36.8868)]
t = np.arange(len(y1))/float(250)
plt.plot(t, y1)
plt.title('Background')
plt.xlabel('Time [s]')
plt.ylabel('EEG')
plt.tight_layout()

plt.figure(figsize=(5, 4))
y2 = data['FP1-FP2'][int(250*36.8868):int(237.2101*250)]
t = np.arange(len(y2))/float(250)
plt.plot(t, y2)
plt.title('Complex partial seizure')
plt.xlabel('Time [s]')
plt.ylabel('EEG')
plt.tight_layout()

#import biosppy as bsp

l = np.vstack((y1, y2[:len(y1)])).transpose()
res = bsp.signals.eeg.eeg(signal=l, sampling_rate=250., show=False)

freqs, psd = signal.periodogram(res['filtered'][:,0], fs=250)
plt.figure(figsize=(5, 4))
plt.plot(freqs, psd)
plt.title('PSD: Background')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()

freqs, psd = signal.periodogram(data['FP1-FP2'][int(250*36.8868):int(237.2101*250)], fs=250)
plt.figure(figsize=(5, 4))
plt.plot(freqs, psd)
plt.title('PSD: Complex partial seizure')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()

import _biosppy.biosppy.signals.tools as st

cutoff = 0.8
b, a = st.get_filter(ftype='butter',
                        band='highpass',
                        order=8,
                        frequency=cutoff,
                        sampling_rate=250.)
w, h = signal.freqz(b,a)

fig = plt.figure()
plt.title('Highpass Butterworth | order 8 | cutoff {}'.format(cutoff))
ax1 = fig.add_subplot(111)
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')
ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
plt.plot(w, angles, 'g')
plt.ylabel('Angle (radians)', color='g')
plt.grid()
plt.axis('tight')
plt.show()


cutoff = 48
b, a = st.get_filter(ftype='butter',
                         band='lowpass',
                         order=16,
                         frequency=cutoff,
                         sampling_rate=250.)
w, h = signal.freqz(b,a)

fig = plt.figure()
plt.title('Lowpass Butterworth | order 16 | cutoff {}'.format(cutoff))
ax1 = fig.add_subplot(111)
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')
ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
plt.plot(w, angles, 'g')
plt.ylabel('Angle (radians)', color='g')
plt.grid()
plt.axis('tight')
plt.show()