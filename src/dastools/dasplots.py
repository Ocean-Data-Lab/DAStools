"""dasplots.py
Collection of functions for plots of DAS data.
Quentin Goestchel, 2024
"""

import numpy as np
import matplotlib.pyplot as plt 
from dastools import dasfuncs

def plot_trace(data, metadata, channels=[2500,7500,15000,30000]):

    time = np.arange(metadata['ns']) / metadata['fs']
    tmax = time.max()  # metadata['ns'] / metadata['fs']

    plt.figure(figsize=(8,4))
    for chan in channels: 
        tr = data[chan,:].astype(float) # get the data at each channel
        tr -= np.mean(tr) # remove the mean
        tr *= metadata['scale_factor'] # convert to strain
        plt.plot(time, tr)

    plt.xlabel('Time [s]')
    plt.ylabel('Strain [-]')
    plt.xlim(0,tmax)
    plt.grid()
    plt.draw()
    return


def plot_psd(data, metadata, channels=[2500,7500,15000,30000]):

    freqs1 = np.fft.rfftfreq(metadata['ns'], 1/metadata['fs'])
    fmax = freqs1.max()
    for chan in channels:
        tr = data[chan,:].astype(float) # get the data at each channel
        tr -= np.mean(tr) # remove the mean
        tr *= metadata['scale_factor'] # convert to strain
        ftr1 = 20 * np.log10(2 * abs(np.fft.rfft(tr * np.hamming(metadata['ns']))) / metadata['ns'])
        t, freqs, ftr = dasfuncs.compute_psd(tr, 5, int(metadata["fs"] * 5), 0.1, metadata['fs'])
        plt.semilogx(freqs, np.mean(ftr, axis=0))
        plt.semilogx(freqs1, ftr1)
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [-]')
    plt.xlim(0.01, fmax)
    plt.grid()
    plt.draw()
    return