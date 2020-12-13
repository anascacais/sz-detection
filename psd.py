import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import _biosppy.biosppy as bsp
import _biosppy.biosppy.signals.tools as st

data = np.load('D:\\TUH\\data_npy\\0103\\00000006_s004_t000.npy')

res = bsp.signals.eeg.eeg(signal=np.reshape(data, (-1,1)), sampling_rate=250., show=False)

freqs, psd = signal.periodogram(data, fs=250)
plt.figure(figsize=(5, 4))
plt.plot(freqs[:int(np.argwhere(freqs==65))], psd[:int(np.argwhere(freqs==65))])
plt.title('PSD: Original Signal')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()


freqs, psd = signal.periodogram(res['filtered'][:,0], fs=250)
plt.figure(figsize=(5, 4))
plt.plot(freqs[:int(np.argwhere(freqs==65))], psd[:int(np.argwhere(freqs==65))])
plt.title('PSD: Filtered Signal')
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.tight_layout()


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