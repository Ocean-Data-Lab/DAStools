import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import signal, ndimage

def fk_filt_chunk(data,tint,fs,xint,dx,c_min,c_max):
    '''
    fk_filt_chunk - perform fk filtering on single chunk of DAS data

    Parameters
    ----------
    data : xr.DataArray
        DataArray containing single chunk
    tint : float
        interval in time between samples
    fs : float
        sampling frequency
    xint : float
        interval in space between samples
    dx : float
        distance between samples

    '''

    data_fft = np.fft.fft2(signal.detrend(data))
    
    # Make freq and wavenum vectors
    nx = data_fft.shape[0]
    ns = data_fft.shape[1]
    f = np.fft.fftshift(np.fft.fftfreq(ns, d = tint/fs))
    k = np.fft.fftshift(np.fft.fftfreq(nx, d = xint*dx))
    ff,kk = np.meshgrid(f,k)

    # Soundwaves have f/k = c so f = k*c

    g = 1.0*((ff < kk*c_min) & (ff < -kk*c_min))
    g2 = 1.0*((ff < kk*c_max) & (ff < -kk*c_max))

    g = g + np.fliplr(g)
    g2 = g2 + np.fliplr(g2)
    g = g-g2
    g = ndimage.gaussian_filter(g, 40)
    # epsilon = 0.0001
    # g = np.exp (-epsilon*( ff-kk*c)**2 )

    g = (g - np.min(g.flatten())) / (np.max(g.flatten()) - np.min(g.flatten()))
    g = g.astype('f')

    data_fft_g = np.fft.fftshift(data_fft) * g
    data_g = np.fft.ifft2(np.fft.ifftshift(data_fft_g))
    
    #return f,k,g,data_fft_g,data_g
    
    # construct new DataArray
    data_gx = xr.DataArray(data_g, dims=['distance','time'], coords=data.coords)
    return data_gx


def fk_filt(data,tint,fs,xint,dx,c_min,c_max):
    '''
    fk_filt - perform fk filtering on DAS data

    Parameters
    ----------
    data : xr.DataArray
        DataArray containing DAS data
    tint : float
        interval in time between samples
    fs : float
        sampling frequency
    xint : float
        interval in space between samples
    dx : float
        distance between samples

    '''
    kwargs = {'tint':tint, 'fs':fs, 'xint':xint, 'dx':dx, 'c_min':c_min, 'c_max':c_max}
    data_gx = data.map_blocks(fk_filt_chunk, kwargs=kwargs, template=data)
    return data_gx


# I think everything below this is implemented in xrsignal
def filtfilt(da, dim, **kwargs):
    '''
    filtfilt - this is an implentation of [scipy.signal.fitlfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)
    This will filter the DAS data in time for each chunk. This process maps chunks and will therefore have error at the end of chunks in time.

    By default, this does not compute, but generates the task graph

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing DAS data that you want to filter.
    dim : string
        dimension to filter in (should be dimension in da)
    **kwargs : various types
        passed to [scipy.signal.filtfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html)
        as per docs, ['x', 'b', and 'a'] are required

    Returns
    -------
    da_filt : xr.DataArray
        filtered data array in time. This does not compute the result, but just the task map in dask
    '''
    kwargs['dim']='time'

    da_filt = da.map_blocks(filtfilt_chunk, kwargs=kwargs, template=da)

    return da_filt

def filtfilt_chunk(da, dim='time', **kwargs):
    '''
    converts dataarray to numpy, sends it to signal.filtfilt and then reinhereits all coordinates

    Parameters
    ----------
    da : xr.DataArray
    dim : string
        dimension to filter over (should be dimension in da)
    **kwargs : various types
        passed to scipy.signal.filtfilt
    '''

    dim_axis = da.dims.index(dim)
    da_np = da.values
    da_filt = signal.filtfilt(x=da_np, axis=dim_axis, **kwargs)

    da_filtx = xr.DataArray(da_filt, dims=da.dims, coords=da.coords, name=da.name, attrs=da.attrs)

    return da_filtx

def spec(da):
    '''
    very quick implentation to calculate spectrogram
        PSD is calculated for every chunk
    
    Currently hardcoded for chunk size of 3000 in time
    Parameters
    ----------
    da : xr.DataArray
        das data to compute spectrogram for
    '''

    template = xr.DataArray(np.ones((int(da.sizes['time']/3000), 513)), dims=['time','frequency']).chunk({'time':1, 'frequency':513})
    return da.map_blocks(__spec_chunk, template=template)

def __spec_chunk(da):
    '''
    compute PSD for single chunk

    Currently hard coded to handle only a time dimension..
    '''
    f, Pxx = signal.welch(da.values, fs=200, nperseg=1024)

    return xr.DataArray(Pxx, dims='frequency', coords={'frequency':f})