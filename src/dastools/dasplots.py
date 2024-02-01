"""dasplots.py
Collection of functions for plots of DAS data.
Quentin Goestchel, 2024
"""

import numpy as np
import matplotlib.pyplot as plt 
from dastools import dasfuncs

def plot_trace(data, metadata, channels=[2500,7500,15000,30000]):
    """Plot the strain time series of channels given in optional argument

    Parameters
    ----------
    data : array-like
        array in h5 format
    metadata : dict
        filled with the measurements metadata
    channels : list, optional
        list of channel indexes, by default [2500,7500,15000,30000]
    """    

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
    """Plot the PSD of signals at channels given in optional argument

    Parameters
    ----------
    data : array-like
        data in h5 format
    metadata : dict
        filled with the measurements metadata
    channels : list, optional
        list of channel indexes, by default [2500,7500,15000,30000]
    """    

    freqs1 = np.fft.rfftfreq(metadata['ns'], 1/metadata['fs'])
    fmax = freqs1.max()
    for chan in channels:
        tr = data[chan,:].astype(float) # get the data at each channel
        tr -= np.mean(tr) # remove the mean
        tr *= metadata['scale_factor'] # convert to strain
        ftr1 = 10 * np.log10(abs(np.fft.rfft(tr * np.hamming(metadata['ns']))) **2 / (metadata['ns'] * metadata['fs']))
        t, freqs, ftr = dasfuncs.compute_psd(tr, 10, 2**10, 0.4, metadata['fs'])
        plt.semilogx(freqs1, ftr1,label='discrete PSD')
        plt.semilogx(freqs,np.mean(ftr, axis=0),label='Welch method') # np.mean(ftr, axis=0)
        plt.legend()
    
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dB.Hz$^{-1}$]')
    plt.xlim(0.01, fmax)
    plt.grid()
    plt.draw()
    return


def plot_waterfall(data,dist,timestamp,vmin,vmax,xmin,xmax,xint,df_loc=None,add_bathymetry=False, cmap='RdBu'):
    fig, ax = plt.subplots(figsize=(12,10))
    # Version for an xarray:
    # np.abs(np.log10(np.abs(data.T))).plot(robust=True, cmap='Greys_r',norm = LogNorm(vmin = vmin, vmax=vmax), add_colorbar=False)
    plt.imshow(data.T,extent=[min(dist)*1e-3,max(dist)*1e-3,min(timestamp),max(timestamp)],aspect='auto',\
             origin='lower',cmap=cmap,vmin=vmin, vmax=vmax)
    plt.xlabel('Distance, km')
    plt.ylabel('Time, s')
    # ax.set_xticks(np.arange(0,70000,5000))
    # plt.xlim(dist[0],dist[-1])
    # plt.ylim(timestamp[0],timestamp[-1]) 
    # plt.gca().invert_xaxis()
    
    if add_bathymetry == 'True':
        ax2 = fig.add_axes([0.125, 0.9, 0.775, 0.1])
        #plt.gca().xaxis.tick_up()
        ax2.xaxis.tick_top()
        plt.plot(np.arange(int(xmin/dx),int(xmax/dx),xint),df_loc['Depth'][int(xmin/dx):int(xmax/dx):xint], color='black', linewidth = 3, zorder = 90, label = 'Bathymetry')
        plt.xlim([int(xmin/dx), int(xmax/dx)])
        plt.grid(True)
        plt.ylim([np.min(df_loc['Depth'][int(xmin/dx):int(xmax/dx):xint]), np.max(df_loc['Depth'][int(xmin/dx):int(xmax/dx):xint])])
        plt.xticks([])
        plt.title("Bathymetry")
    return