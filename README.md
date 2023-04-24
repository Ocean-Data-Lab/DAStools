# DAStools
A group of tools and functions for analysis of DAS data.
This repository is under development and the main branch will likely be rapidly changing. Please feel free to submit PRs if you find anything that would be nice to include.

A related repository is John-Ragland/xrsignal, which implements various scipy.signal methods in a scalable, dask/xarray compatible environment.

## Contents
- dasfuncs.py
    - signal processing methods built on numpy and scipy methods
- das_chunk.py
    - implementations of dasfuncs and other functions on single chunks of data.
    - also includes associated functions that map these (using xarray.map_blocks), to an entire array