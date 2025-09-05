
#%%
import numpy as np
import xarray as xr
import pandas as pd
import sys, os, glob 


print(os.getcwd())


filepath_countrymask = os.path.join(
    '/data/brussel/vo/000/bvo00012/vsc10419/demographics4climate/data/country-masks/isipedia-countries/countrymasks_fractional_05deg.nc')


#%%

def load_countrymasks_fillcoasts(
    filepath_countrymask=filepath_countrymask,
    fillcoast=True,
    fix_smallislands=False,
    bbox=None,
    ):
    """
    Load countrymasks and fill coastal pixels so sum of fraction = 1 so coastal populations are not lost. 

    """

    # Open data 
    ds=xr.open_dataset(filepath_countrymask, chunks={"lat": 100, "lon": 100})
    da_countrymasks = ds.to_array()
    # clean
    strings = da_countrymasks['variable'].values
    cleaned_strings = [s[2:] if s.startswith('m_') else s for s in strings]
    da_countrymasks['variable'] = cleaned_strings
    # last variable is 'world', lose it 
    da_countrymasks = da_countrymasks.isel(variable=slice(0,225))
    # sum over all countries 
    countrymask_sum = da_countrymasks.sum(dim='variable')

    if fillcoast:
        # Part 2. Correct for coastal pixels 
        # where sum of fraction is less than 1, weighted multiplication for sum to equal one
        da_countrymasks_correct = xr.where(countrymask_sum < 1, da_countrymasks * (1 / countrymask_sum ), da_countrymasks)
        # small area sum = 2, correct for it 
        da_countrymasks_corr = xr.where(da_countrymasks_correct.sum(dim='variable') > 1, da_countrymasks_correct/da_countrymasks_correct.sum(dim='variable'), da_countrymasks_correct)
        da_countrymasks = da_countrymasks_corr

    if fix_smallislands:  
        #TODO change the lat indexing to be with coords!! 
        # Fix issue in Singapore pixel, assign fraction from IOSID to SGP 
        da_countrymasks.loc[dict(lat=da_countrymasks.lat[177], lon=da_countrymasks.lon[567], variable='SGP')] += da_countrymasks.loc[dict(lat=da_countrymasks.lat[177], lon=da_countrymasks.lon[567], variable='IOSID')].values
        da_countrymasks.loc[dict(lat=da_countrymasks.lat[177], lon=da_countrymasks.lon[567], variable='IOSID')] = 0
        # Fix it also in Mauritius 
        da_countrymasks.loc[dict(lat=da_countrymasks.lat[220], lon=da_countrymasks.lon[474], variable='MUS')] += da_countrymasks.loc[dict(lat=da_countrymasks.lat[220], lon=da_countrymasks.lon[474], variable='IOSID')].values
        da_countrymasks.loc[dict(lat=da_countrymasks.lat[220], lon=da_countrymasks.lon[474], variable='IOSID')] = 0
    

    return da_countrymasks.rename({'variable':'country'})

da_countrymasks = load_countrymasks_fillcoasts(
    filepath_countrymask=filepath_countrymask,
    fillcoast=True,
    fix_smallislands=True,
    bbox=None,
    )

da_countrymasks.to_netcdf('data/country-masks/isipedia-countries/countrymasks_fractional_05deg_filledcoasts.nc')