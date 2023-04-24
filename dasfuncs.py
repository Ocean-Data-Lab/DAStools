"""dasfuncs.py
Collection of functions for signal processing of DAS data built on numpy and scipy methods.
Shima Abadi, 2023
"""

import numpy as np
from numpy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import scipy.signal as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from datetime import datetime
import geopy.distance
from pyproj import Proj
from scipy import signal
from scipy.signal import butter, sosfilt, tukey, detrend, filtfilt
from scipy.ndimage import gaussian_filter

def compute_psd(data_sel,avg_time,L_fft,overlap,fs):
    t = []
    psd = []
    ns = len(data_sel)
    for n in range(int(ns/(fs*avg_time))-1):
        f, Pxx = signal.welch(x=data_sel.data[n * int(fs * avg_time) : (n + 1) * int(fs * avg_time)], fs=fs, window='hann' , nperseg=L_fft, noverlap=int(L_fft * overlap), nfft=L_fft,average='mean')
        Pxx = 10 * np.log10(Pxx)
        psd.append(Pxx)
        t.append(n*avg_time)
    return t, f, psd

def beamf(data,dist,f,fs,fmin,fmax):
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

def bp_filt(data,tint,fs,fmin,fmax):
    b, a = sp.butter(8,[fmin/(fs/2),fmax/(fs/2)],'bp')
    tr_filt = sp.filtfilt(b,a,data.astype(float),axis = 1)
    return tr_filt

def fk_filt(data,tint,fs,xint,dx,c_min,c_max):
    data_fft = fft2(detrend(data))
    # Make freq and wavenum vectors
    nx = data_fft.shape[0]
    ns = data_fft.shape[1]
    f = fftshift(fftfreq(ns, d = tint/fs))
    k = fftshift(fftfreq(nx, d = xint*dx))
    ff,kk = np.meshgrid(f,k)

    # Soundwaves have f/k = c so f = k*c

    g = 1.0*((ff < kk*c_min) & (ff < -kk*c_min))
    g2 = 1.0*((ff < kk*c_max) & (ff < -kk*c_max))

    g = g + np.fliplr(g)
    g2 = g2 + np.fliplr(g2)
    g = g-g2
    g = gaussian_filter(g, 40)
    # epsilon = 0.0001
    # g = np.exp (-epsilon*( ff-kk*c)**2 )

    g = (g - np.min(g.flatten())) / (np.max(g.flatten()) - np.min(g.flatten()))
    g = g.astype('f')

    data_fft_g = fftshift(data_fft) * g
    data_g = ifft2(ifftshift(data_fft_g))
    
    return f,k,g,data_fft_g,data_g

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
        
def plot_waterfall(data,dist,timestamp,vmin,vmax,xmin,xmax,xint,df_loc,add_bathymetry):
    fig, ax = plt.subplots(figsize=(12,10))
    np.abs(np.log10(np.abs(data.T))).plot(robust=True, cmap='Greys_r',norm = LogNorm(vmin = vmin, vmax=vmax), add_colorbar=False)
    plt.xlabel('Distance, m')
    plt.ylabel('Time, s')
    ax.set_xticks(np.arange(0,70000,5000))
    plt.xlim(dist[0],dist[-1])
    plt.ylim(timestamp[0],timestamp[-1]) 
    plt.gca().invert_xaxis()
    
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

    plt.rcParams.update({'font.size': 38})