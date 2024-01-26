'''
auto-generated from copilot and not yet tested
'''
import pytest
import numpy as np
import xarray as xr
from DAStools.tools import fk_filt_chunk, fk_filt, _energy_TimeDomain_chunk, energy_TimeDomain, filtfilt

def test_fk_filt_chunk():
    data = xr.DataArray(np.random.rand(10, 10), dims=['distance', 'time'])
    tint = 1.0
    fs = 1.0
    xint = 1.0
    dx = 1.0
    c_min = 1.0
    c_max = 2.0
    result = fk_filt_chunk(data, tint, fs, xint, dx, c_min, c_max)
    assert isinstance(result, xr.DataArray)
    assert result.shape == data.shape

def test_fk_filt():
    data = xr.DataArray(np.random.rand(10, 10), dims=['distance', 'time'])
    tint = 1.0
    fs = 1.0
    xint = 1.0
    dx = 1.0
    c_min = 1.0
    c_max = 2.0
    result = fk_filt(data, tint, fs, xint, dx, c_min, c_max)
    assert isinstance(result, xr.DataArray)
    assert result.shape == data.shape

def test__energy_TimeDomain_chunk():
    data = xr.DataArray(np.random.rand(10, 10), dims=['distance', 'time'])
    result = _energy_TimeDomain_chunk(data)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (10, 1)

def test_energy_TimeDomain():
    data = xr.DataArray(np.random.rand(10, 10), dims=['distance', 'time'])
    result = energy_TimeDomain(data)
    assert isinstance(result, xr.DataArray)
    assert result.shape == (10, 1)

def test_filtfilt():
    data = xr.DataArray(np.random.rand(10, 10), dims=['distance', 'time'])
    dim = 'time'
    kwargs = {'b': [1, -1], 'a': [1, -1]}
    result = filtfilt(data, dim, **kwargs)
    assert isinstance(result, xr.DataArray)
    assert result.shape == data.shape