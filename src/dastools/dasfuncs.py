"""dasfuncs.py
Collection of functions for signal processing of DAS data built on numpy and scipy methods.
Shima Abadi, 2023
"""

import numpy as np
from numpy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import wget
from matplotlib.colors import LogNorm
from datetime import datetime
import geopy.distance
from pyproj import Proj
from scipy import signal
from scipy.signal import butter, sosfilt, tukey, detrend, filtfilt
from scipy.ndimage import gaussian_filter


def compute_psd(data_sel,avg_time,L_fft,overlap,fs):
    """Compute Power Spectral Density (PSD) using the Welch method.

    Parameters
    ----------
    data_sel : array-like
        Input signal data.
    avg_time : float
        Length of the segments for averaging in seconds.
    L_fft : int
        Length of the FFT used for each segment.
    overlap : float
        Overlap between segments (fraction of segment length).
    fs : float
        Sampling frequency of the signal.

    Returns
    -------
    t : list
        Time values corresponding to each PSD estimate.
    f : array
        Frequency values.
    psd : list of arrays
        Power Spectral Density estimates for each time segment.

    Notes
    -----
    The function uses the Welch method to estimate the PSD. It divides the input signal into overlapping
    segments, applies a windowing function (Hann window by default), computes the FFT for each segment, 
    and averages the resulting power spectral densities. The function returns time values (t), frequency values (f), and the corresponding PSD estimates (psd).

    The frequency values (f) and PSD estimates (psd) can be used to visualize the time-dependent frequency content of the signal using plots such as spectrograms or line plots.
    """    
    t = []
    psd = []
    ns = len(data_sel)
    for n in range(int(ns/(fs*avg_time))-1):
        # Use the Welch method on each time segments
        f, Pxx = signal.welch(x=data_sel.data[n * int(fs * avg_time) : (n + 1) * int(fs * avg_time)], fs=fs, window='hann' , nperseg=L_fft, noverlap=int(L_fft * overlap), nfft=L_fft,average='mean')
        # Convert the PSD in dB/Hz
        Pxx = 10 * np.log10(Pxx)
        psd.append(Pxx)
        t.append(n*avg_time)
    return t, f, psd


def beamf(data,dist,f,fs,fmin,fmax):
    '''
    beamf - beamform data

    Parameters
    ----------
    data : array
        data to beamform
    dist : float
        distance between sensors
    f : array
        frequency for beamforming
    fs : float
        sampling frequency
    fmin : float
        minimum frequency for beamforming
    fmax : float
        maximum frequency for beamforming

    Returns
    -------
    theta : array
        angle of arrival
    B : array
        beamformed data
    theta_max : float
        maximum angle of arrival

    '''
    c = 1500     
    nx = len(data)
    ns = len(data[0])
    nfft = ns
    signalfft = np.fft.fft(data,nfft,axis = 1)
    tou = np.zeros((nx,1))
    B = []
    theta = np.linspace(-np.pi/2, np.pi/2, num = 180)
    for theta_temp in theta:
        w = []
        BB = []
        for f_ind in range(int(fmin*nfft/fs),int(fmax*nfft/fs)):
            tou = dist*np.sin(theta_temp)/c
            w = (np.exp(1j*2*np.pi*f[int(f_ind)]*tou[:]))
            BB.append(np.dot(signalfft[:,int(f_ind)],w))
        B.append(sum(np.square(np.abs(BB))))
    theta_max = theta[np.abs(B) == max(np.abs(B))][0]
    return theta,B,theta_max


def bp_filt(data,fs,fmin,fmax):
    b, a = sp.butter(8,[fmin/(fs/2),fmax/(fs/2)],'bp')
    tr_filt = sp.filtfilt(b,a,data,axis = 1)
    return tr_filt


def fk_filt(data,tint,fs,xint,dx,c_min,c_max):
    """fk_filt - perform fk filtering on an array of DAS data   

    Parameters
    ----------
    data : array-like
        array containing wave signal from DAS data
    tint : float
        decimation time interval between considered samples
    fs : float
        sampling frequency
    xint : float
        decimation space interval between considered samples
    dx : float
        spatial resolution
    c_min : float
        minimum phase speed for the pass-band filter in f-k domain
    c_max : float
        maximum phase speed for the pass-band filter in f-k domain

    Returns
    -------
    f : array-like
        vector of frequencies

    k : array-like
        vector of wavenumbers   
    g : array-like
        2D designed gaussian filter
    data_fft_g: array-like
        2D Fourier transformed data, filtered by g
    data_g.real: array-like
        Real value of spatiotemporal filtered data
    """    

    # Perform 2D Fourier Transform on the detrended input data
    data_fft = fft2(detrend(data))
    # Make freq and wavenum vectors
    nx = data_fft.shape[0]
    ns = data_fft.shape[1]
    f = fftshift(fftfreq(ns, d = tint/fs))
    k = fftshift(fftfreq(nx, d = xint*dx))
    ff,kk = np.meshgrid(f,k)

    #  Define a filter in the f-k domain
    # Soundwaves have f/k = c so f = k*c

    g = 1.0*((ff < kk*c_min) & (ff < -kk*c_min))
    g2 = 1.0*((ff < kk*c_max) & (ff < -kk*c_max))

    # Symmetrize the filter
    g += np.fliplr(g)
    # g2 += np.fliplr(g2)
    # g -= g2 + np.fliplr(g2) # combine to have g = g - g2
    
    # Apply Gaussian filter to the f-k filter
    # Tuning the standard deviation of the filter can improve computational efficiency
    # g = gaussian_filter(g, 40)
    # epsilon = 0.0001
    # g = np.exp (-epsilon*( ff-kk*c)**2 )

    # Normalize the filter to values between 0 and 1
    g = (g - np.min(g)) / (np.max(g) - np.min(g))

    # Apply the filter to the 2D Fourier-transformed data
    data_fft_g = fftshift(data_fft) * g
    # Perform inverse Fourier Transform to obtain the filtered data in t-x domain
    data_g = ifft2(ifftshift(data_fft_g))
    
    # return f, k, g, data_fft_g, data_g.real
    return data_g.real


def array_geo(nx,dx,cable_name):
    if cable_name == 'north':
        fpath_base = './'
        tmp = np.genfromtxt(fpath_base+'north_DAS_latlondepth.txt')
        x1_pick = 1923
        x2_pick = 66278+1
    elif cable_name == 'south':
        fpath_base = './'
        tmp = np.genfromtxt(fpath_base+'south_DAS_latlondepth.txt')
        x1_pick = 1923
        x2_pick = 66278+30471

    c = np.arange(nx,dtype=int)
    c0 = int(x1_pick//dx)
    c1 = int(x2_pick//dx)        # channel numbers of beginning and end
    c = c[c0:c1+1]               # cut off beginning and ending channels
    nc = len(c)                  # new number of channels

    #du = tmp[:,0]
    lat = tmp[:,1]
    lon = tmp[:,2]
    depth = tmp[:,3]
    
    myProj = Proj("+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    x,y = myProj(lon,lat)

    # Calculate geographic distance along cable track 
    xd = np.diff(x)
    yd = np.diff(y)
    dd = np.sqrt(xd**2 + yd**2)
    u = np.cumsum(dd)
    u = np.hstack(([0],u))

    # Interpolate channel locations in x,y
    du = np.linspace(0,u.max(),nx)
    du = du[c0:c1]

    cable_loc = {'Lat': lat, 'Lon': lon, 'Dist': du,'Depth': depth,'First2Last_ChannelIndex': c}
    return cable_loc 


def get_metadata_optasense(fp1):
    fs = fp1['Acquisition']['Raw[0]'].attrs['OutputDataRate'] # sampling rate in Hz
    dx = fp1['Acquisition'].attrs['SpatialSamplingInterval'] # channel spacing in m
    ns = fp1['Acquisition']['Raw[0]']['RawDataTime'].attrs['Count']
    n = fp1['Acquisition']['Custom'].attrs['Fibre Refractive Index'] # refractive index
    GL = fp1['Acquisition'].attrs['GaugeLength'] # gauge length in m
    nx = fp1['Acquisition']['Raw[0]'].attrs['NumberOfLoci'] # number of channels
    scale_factor = (2*np.pi)/2**16 * (1550.12 * 1e-9)/(0.78 * 4 * np.pi * n * GL)

    meta_data = {'fs': fs, 'dx': dx, 'ns': ns,'n': n,'GL': GL, 'nx':nx , 'scale_factor': scale_factor}
    return meta_data


def axvlines(ax = None, xs = [0, 1], ymin=0, ymax=1, **kwargs):
    ax = ax or plt.gca()
    for x in xs:
        ax.axvline(x, ymin=ymin, ymax=ymax, **kwargs)


def scale(data, metadata):
    """Convert a data array from the OptaSENSE HDF5 int32 format to a numpy array of float32 strain values.

    Parameters
    ----------
    data : np.ndarray
        The input data array in OptaSENSE HDF5 int32 format.
    metadata : dict
        A dictionary containing metadata information, including the 'scale_factor'.

    Returns
    -------
    np.ndarray
        A numpy array of float32 strain values.
    """
    # TODO: adapt to xarrays/daskarray
    tr = data.astype(np.float64) # convert the data
    # remove the mean along the time dimension for each channels
    tr -= np.mean(tr, axis=1, keepdims=True)
    tr *= metadata['scale_factor'] # convert to strain
    return tr


def dl_file(url):
    """Download the file at the given url

    Parameters
    ----------
    url : string
        url location of the file

    Returns
    -------
    filepath : string
        local path destination of the file
    """    
    filename = url.split('/')[-1]
    filepath = os.path.join('data',filename)
    if os.path.exists(filepath) == True:
        print(f'{filename} already stored locally')
    else:
        # Create the data subfolder if it doesn't exist
        os.makedirs('data', exist_ok=True)
        wget.download(url, out='data', bar=wget.bar_adaptive)
    return filepath


# plt.rcParams.update({'font.size': 38})