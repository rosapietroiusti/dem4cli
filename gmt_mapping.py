"""
Adapted from L. Grant 2025 Unprecedented lifetime exposure

Amaury Laridon / Rosa Pietroiusti 


To do
- add data for representative scenarios

"""
#%%

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import pickle as pk
from scipy import interpolate
import regionmask
import glob
import os
from copy import deepcopy as cp
import numpy as np
import dask
from dask.diagnostics import progress
import statsmodels.api as sm

from _settings import * 

#%%

#%%

def ar6_scen_grab(
    scens,
    df_GMT_all,
):
    """
    Filter represtative AR6 scenarios from whole AR6 scenario database. 

    Input:
        scens (dict):               defined in settings.py, thresholds to define representative scenarios filtered from AR6 explorer
                                              as 1.5, 2, NDCs, 3, 4 degrees
        df_GMT_all (df):                      all the AR6 pathways (903 pathways)

    Returns:
        df_GMT_lb, df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_30, df_GMT_40 (dfs):    representative pathways for lower bound, 1.5 degree, 2 degree, NDCs, 3 degrees, upper bound (~ 4 degrees)
                                                                                    lower bound is defined as being the scenario that is the coldest scenario on more years than any other scenario, 
                                                                                    the upper bound (~4 deg in 2100) is defined as the scenario that is hottest on more years,
                                                                                    1.5,2,3 deg are defined based on peak warming, i.e. the 1.5 scenario has peak warming held within the bounds 
                                                                                    defined in 'scens' 
                                                                                    All of these these are raw scenarios filtered from the database, not interpolated yet. 

    Notes:
        - NDCs fixed at 2.4 in settings.py - update this value? 
        - 1.5 refers to peak temperature, it is actually a 1.3 degree scenario in 2100
                                                                                    
    """
    
    # start with upper line toward 4 degrees
    # convert to bools based on row max to find column with most maxes via idxmax
    maxes = pd.concat(
        [df_GMT_all.loc[:,c]==df_GMT_all.max(axis=1) for c in df_GMT_all.columns],
        axis=1,
    )
    df_GMT_40 = df_GMT_all.loc[:,df_GMT_all.columns[maxes.sum(axis=0).idxmax()]]
    
    # second line, 3 degrees
    # get all lines between target (3) and lower bound (first criteria)
    df_GMT_30 = df_GMT_all[
        df_GMT_all.columns[(df_GMT_all.max(axis=0)<scens['3.0'][1])&(df_GMT_all.max(axis=0)>scens['3.0'][0])]
    ]  
    # dfbools is new df with bool cells for years where series in df_GMT_30 are below the 4 deg line
    # check if it remains before the 4 deg line to choose a run that is consistently below 4 deg
    dfbools=pd.concat(
        [df_GMT_30.loc[:,c]<=df_GMT_40.loc[:] for c in df_GMT_30.columns],
        axis=1,
    )
    if len(df_GMT_30[df_GMT_30.columns[dfbools.all()]].columns) == 0: # if there's no columns fully beneath upper line, grab least overlapping
        minfalsecol = df_GMT_30.columns[dfbools.sum(axis=0).idxmax()]
        df_GMT_30 = df_GMT_30.loc[:,minfalsecol]    
    else: # otherwise ( if there are more than one), get column with most max years in subset i.e. the hottest simulation still within constraint 
        maxes = pd.concat(
            [df_GMT_30.loc[:,c]==df_GMT_30.max(axis=1) for c in df_GMT_30[df_GMT_30.columns[dfbools.all()]].columns],
            axis=1,
        )
        maxes.columns = df_GMT_30[df_GMT_30.columns[dfbools.all()]].columns
        df_GMT_30 = df_GMT_30[df_GMT_30.columns[dfbools.all()]].loc[:,maxes.sum(axis=0).idxmax()]
        
    # third line, NDC, between the bounds defined in scens 
    df_GMT_NDC = df_GMT_all[
        df_GMT_all.columns[(df_GMT_all.max(axis=0)<scens['NDC'][1])&(df_GMT_all.max(axis=0)>scens['NDC'][0])]
    ]
    # check if it remains before the 3 deg line to choose a run that is consistently below 3 deg
    dfbools=pd.concat(
        [df_GMT_NDC.loc[:,c]<=df_GMT_30.loc[:] for c in df_GMT_NDC.columns],
        axis=1,
    )
    if len(df_GMT_NDC[df_GMT_NDC.columns[dfbools.all()]].columns) == 0: # if there's no columns fully beneath upper line, grab least overlapping
        minfalsecol = df_GMT_NDC.columns[dfbools.sum(axis=0).idxmax()]
        df_GMT_NDC = df_GMT_NDC.loc[:,minfalsecol]    
    else: # otherwise, get column with most max years in subset (i.e. the hottest simulation that is still below 3 deg and within constraints)
        maxes = pd.concat(
            [df_GMT_NDC.loc[:,c]==df_GMT_NDC.max(axis=1) for c in df_GMT_NDC[df_GMT_NDC.columns[dfbools.all()]].columns],
            axis=1,
        )
        maxes.columns = df_GMT_NDC[df_GMT_NDC.columns[dfbools.all()]].columns
        df_GMT_NDC = df_GMT_NDC[df_GMT_NDC.columns[dfbools.all()]].loc[:,maxes.sum(axis=0).idxmax()]

    # 2 degree scen - peak warming is close to 2 degrees
    df_GMT_20 = df_GMT_all[
        df_GMT_all.columns[(df_GMT_all.max(axis=0)<scens['2.0'][1])&(df_GMT_all.max(axis=0)>scens['2.0'][0])]
    ]
    dfbools=pd.concat(
        [df_GMT_20.loc[:,c]<=df_GMT_NDC.loc[:] for c in df_GMT_20.columns],
        axis=1,
    )
    if len(df_GMT_20[df_GMT_20.columns[dfbools.all()]].columns) == 0:
        minfalsecol = df_GMT_20.columns[dfbools.sum(axis=0).idxmax()]
        df_GMT_20 = df_GMT_20.loc[:,minfalsecol]
    else:    
        maxes = pd.concat(
            [df_GMT_20.loc[:,c]==df_GMT_20.max(axis=1) for c in df_GMT_20[df_GMT_20.columns[dfbools.all()]].columns],
            axis=1,
        )
        maxes.columns = df_GMT_20[df_GMT_20.columns[dfbools.all()]].columns
        df_GMT_20 = df_GMT_20[df_GMT_20.columns[dfbools.all()]].loc[:,maxes.sum(axis=0).idxmax()]    

    # 1.5 degree scen - peak warming is close to 1.5 degrees
    df_GMT_15 = df_GMT_all[
        df_GMT_all.columns[(df_GMT_all.max(axis=0)<scens['1.5'][1])&(df_GMT_all.max(axis=0)>scens['1.5'][0])]
    ]
    dfbools=pd.concat(
        [df_GMT_15.loc[:,c]<=df_GMT_20.loc[:] for c in df_GMT_15.columns],
        axis=1,
    )
    if len(df_GMT_15[df_GMT_15.columns[dfbools.all()]].columns) == 0:
        minfalsecol = df_GMT_15.columns[dfbools.sum(axis=0).idxmax()]
        df_GMT_15 = df_GMT_15.loc[:,minfalsecol]
    else:    
        maxes = pd.concat(
            [df_GMT_15.loc[:,c]==df_GMT_15.max(axis=1) for c in df_GMT_15[df_GMT_15.columns[dfbools.all()]].columns],
            axis=1,
        )
        maxes.columns = df_GMT_15[df_GMT_15.columns[dfbools.all()]].columns
        df_GMT_15 = df_GMT_15[df_GMT_15.columns[dfbools.all()]].loc[:,maxes.sum(axis=0).idxmax()]
    
    # lower bound
    mins = pd.concat(
            [df_GMT_all.loc[:,c]==df_GMT_all.min(axis=1) for c in df_GMT_all.columns],
            axis=1,
    )
    df_GMT_lb = df_GMT_all.loc[:,df_GMT_all.columns[mins.sum(axis=0).idxmax()]] 

    return df_GMT_lb, df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_30, df_GMT_40



#%%---------------------------------------------------------------#
# Load global mean temperature projections and build              #
# stylized trajectories                                           #
# ----------------------------------------------------------------#

def load_GMT(
    year_start,
    year_end,
    gmt_extend_method='10yrtrend',
    smooth_first_decades=True,
):

    """
    Creation of stylized GMT trajectories based on loaded pathways, using method that in S2S is called 
    'ar6_new_dem4cli', i.e. takes pathways from AR6 scenarios explorer, interpolates to have them at regular 0.1 
    intervals as stylized trajectories, interpolates again to have pathways going from 1.5 to 3.5 degrees in 2100. 

    Input
        year_start, year_end:       desired start and end of stylized trajectories
        gmt_extend_method (str):    options to extend past 2100 - '10yrmean' 'lastyear' and '10yrtrend'
                                    respectively repeat last 10 year mean, repeat last year and extend last 
                                    10 year trend
        filepaths embedded in function 

    Returns 
        df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_OS, df_GMT_noOS, ds_GMT_STS, df_GMT_strj (dfs) :   stylized trajectories. OS and noOS come from Wim's original code, NDC hits ~2.4 in 2100 (defined in settings)
                                                                                                    All are extended until year_end based on gmt_extend_method, except OS and noOS where the value in 2100 is held constant

    Notes
        - original df_GMT_SR15 used only to get historical 1960-1999
        - can still clean this up a bit
    """

    def extend_gmt_to_year_range(df, year_start=year_start, year_end=year_end, gmt_extend_method=gmt_extend_method):
        """
        Extend a GMT time series (Series or DataFrame) to a given year range.
        
        Handles both backward and forward extension using specified method.
        
        Parameters
        ----------
        df : pd.Series or pd.DataFrame
            Input data indexed by year.
        year_start : int, optional
            Earliest year to extend to (fills backward if needed).
        year_end : int, optional
            Latest year to extend to (fills forward if needed).
        gmt_extend_method : {'10yrmean', 'lastyear', '10yrtrend'}, optional
            Method for forward extension beyond the last available year.
            Note: OS and noOS are always extended with 'lastyear' where this year is taken from 2100
        
        Returns
        -------
        pd.DataFrame
            Extended dataset over [year_start, year_end].
        """
        # --- Ensure DataFrame ---
        if isinstance(df, pd.Series):
            df = df.to_frame()
            df.columns = [f"{df.loc[2100].values[0]:.1f}"] if 2100 in df.index else ["value"] # [str(round(df.loc[2100].values[0], 2))] if 2100 in df.index else ["value"]

        df = df.copy()
        df.index = df.index.astype(int)
        
        min_year, max_year = int(np.nanmin(df.index)), int(np.nanmax(df.index))

        # --- Backward extension ---
        if year_start is not None and min_year > year_start:
            first_10y_mean = df.iloc[:10, :].mean()
            for year in range(year_start, min_year):
                df.loc[year] = first_10y_mean
        
        df = df.sort_index()

        # --- Forward extension ---
        if year_end is not None and max_year < year_end:
            if gmt_extend_method == "10yrmean":
                last_10y_mean = df.iloc[-10:, :].mean()
                for year in range(max_year + 1, year_end + 1):
                    df.loc[year] = last_10y_mean

            elif gmt_extend_method == "lastyear":
                last_year_vals = df.iloc[-1, :]
                for year in range(max_year + 1, year_end + 1):
                    df.loc[year] = last_year_vals

            elif gmt_extend_method == "10yrtrend":
                yrs = np.arange(max_year - 9, max_year + 1)
                trends = {col: np.polyfit(yrs, df.loc[yrs, col].astype(float), 1) for col in df.columns}
                for year in range(max_year + 1, year_end + 1):
                    df.loc[year] = {col: np.polyval(trends[col], year) for col in df.columns}

        # --- Sort index and restrict range ---
        df = df.sort_index()
        if year_start is not None or year_end is not None:
            df = df.loc[
                (df.index >= (year_start if year_start is not None else df.index.min())) &
                (df.index <= (year_end if year_end is not None else df.index.max()))
            ]

        return df


    # ---------------------------------------------------------- #
    # Definition trajectories from SR15                          #
    # This is the original scenarios used in Thiery et al.(2021) #                                      
    # ---------------------------------------------------------- #

    # wim's original scenarios; use historical obs years from here, 1960-1999, but replace with ar6 trajectories from 2000
    df_GMT_SR15 = pd.read_excel(dir_temperature_trajectories+'/temperature-trajectories_SR15/GMT_50pc_manualoutput_4pathways.xlsx', header=1);
    df_GMT_SR15 = df_GMT_SR15.iloc[:4,1:].transpose().rename(columns={
        0 : 'IPCCSR15_IMAGE 3.0.1_SSP1-26_GAS',
        1 : 'IPCCSR15_MESSAGE-GLOBIOM 1.0_ADVANCE_INDC_GAS',
        2 : 'IPCCSR15_MESSAGE-GLOBIOM 1.0_SSP2-19_GAS',
        3 : 'IPCCSR15_MESSAGEix-GLOBIOM 1.0_LowEnergyDemand_GAS'
    })

    # currently using only hist from this earlier version of df_GMT_15 (df_GMT_15 gets remade below)
    df_GMT_15 = df_GMT_SR15.loc[year_start:,'IPCCSR15_MESSAGEix-GLOBIOM 1.0_LowEnergyDemand_GAS']
    # check and drop duplicate years
    df_GMT_15 = df_GMT_15[~df_GMT_15.index.duplicated(keep='first')]

    # ---------------------------------------------------------- #
    # Definition of the OverShoot (OS) and no-OverShoot (noOS)   #
    # trajectories from .mat object of Thiery et al.(2021)       #
    # ---------------------------------------------------------- # 

    from scipy.io import loadmat

    # Load GMT_OS
    mat_data = loadmat(dir_temperature_trajectories + '/temperature-trajectories_Wim/GMT_OS.mat', squeeze_me=True)
    GMT_OS = mat_data['GMT_OS'].flatten()
    years = np.arange(1960, 1960 + len(GMT_OS))
    df_GMT_OS = pd.Series(GMT_OS, index=years)
    df_GMT_OS.name = None
    df_GMT_OS.index.name = None

    # Load GMT_noOS
    mat_data = loadmat(dir_temperature_trajectories + '/temperature-trajectories_Wim/GMT_noOS.mat', squeeze_me=True)
    GMT_noOS = mat_data['GMT_noOS'].flatten()
    years = np.arange(1960, 1960 + len(GMT_noOS))
    df_GMT_noOS = pd.Series(GMT_noOS, index=years)
    df_GMT_noOS.name = None
    df_GMT_noOS.index.name = None

    df_GMT_OS = extend_gmt_to_year_range(df_GMT_OS.loc[:2100],year_start, year_end, gmt_extend_method='lastyear')
    df_GMT_noOS = extend_gmt_to_year_range(df_GMT_noOS.loc[:2100],year_start,year_end, gmt_extend_method='lastyear')

    # ---------------------------------------------------------- #
    # Definition of the Stress Test Scenarios (STS)              #
    # by the SPARCCLE project                                    #
    # ---------------------------------------------------------- #

    # Open the NetCDF file
    ds_GMT_STS = xr.open_dataset(dir_temperature_trajectories + '/temperature-trajectories_STS/GSAT_FaIR_SPARCCLE_STSv1.nc', engine='netcdf4')

    # ---------------------------------------------------------- #
    # Definition of stylized trajectories:                       #
    # 'ar6_new_dem4cli' approach in S2S, which is ar6_new with   #
    # bug fixes and introducing options for post-2100 extension  #
    # ---------------------------------------------------------- #
        
    # ------------------------- This is original AR6 approach --------------------------
    # collect new ar6 scens from IASA explorer
    df_GMT_ar6 = pd.read_csv(dir_temperature_trajectories+'/temperature-trajectories_AR6/ar6_c1_c7_nogaps_2000-2100.csv',header=0)
    df_GMT_ar6.loc[:,'Model'] = df_GMT_ar6.loc[:,'Model']+'_'+df_GMT_ar6.loc[:,'Scenario']
    df_GMT_ar6 = df_GMT_ar6.drop(columns=['Scenario','Region','Variable','Unit']).transpose()
    df_GMT_ar6.columns=df_GMT_ar6.loc['Model',:]
    df_GMT_ar6.columns.name = None
    df_GMT_ar6 = df_GMT_ar6.drop(df_GMT_ar6.index[0])
    df_GMT_ar6 = df_GMT_ar6.dropna(axis=1)
    df_GMT_ar6.index = df_GMT_ar6.index.astype(int)

    # stitch with same tseries for early decades

    if smooth_first_decades:
        # option to smooth the first decades, otherwise has more variability than later decades 
        # 2000 is a hot year in the GMT_15 timeseries so it creates a jump to go only until 1999 with the historical, changed for this 

        # opion 1) rolling mean
        df_GMT_15 = df_GMT_15.rolling(21,min_periods=1,center=True).mean()

        # option 2) lowess - looks very similar 
        #frac = np.round( 21 / len(df_GMT_15), 2)
        #y_smo = sm.nonparametric.lowess(df_GMT_15.values.ravel(), np.arange(len(df_GMT_15)), frac=frac, return_sorted=False) # use loess instead to get smoother endpoints
        #df_GMT_15 = pd.DataFrame(y_smo, index=df_GMT_15.index)
        df_hist_all = df_GMT_15.loc[year_start:2016]
    else:
        df_hist_all = df_GMT_15.loc[year_start:1999]
        
    df_hist_all = pd.concat([df_hist_all for i in range(len(df_GMT_ar6.columns))],axis=1)
    df_hist_all.columns = df_GMT_ar6.columns
    df_GMT_ar6 = pd.concat([df_hist_all,df_GMT_ar6],axis=0) # add historical values to additional scenarios
    
    # drop dups
    df_GMT_ar6 = df_GMT_ar6[~df_GMT_ar6.index.duplicated(keep='first')]

    # get new trajects - this overwrites the above df_GMT_15
    df_GMT_lb, df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_30, df_GMT_40 = ar6_scen_grab(
        scen_thresholds,
        df_GMT_ar6,
    )        

    GMT_max = df_GMT_40.iloc[-1]
    GMT_fut_strtyr = int(df_GMT_15.index.where(df_GMT_15==df_GMT_20).max())+1
    ind_fut_strtyr = int(np.argwhere(np.asarray(df_GMT_15.index)==GMT_fut_strtyr))
    GMT_min = df_GMT_lb.loc[GMT_fut_strtyr-1]
    GMT_steps = np.arange(0,GMT_max+0.05,GMT_inc)
    GMT_steps = np.insert(GMT_steps[np.where(GMT_steps>GMT_min)],0,GMT_min)
    n_steps = len(GMT_steps)
    ind_lb = np.argmin(np.abs(GMT_steps-df_GMT_lb.iloc[-1]))
    ind_15 = np.argmin(np.abs(GMT_steps-df_GMT_15.iloc[-1]))
    ind_20 = np.argmin(np.abs(GMT_steps-df_GMT_20.iloc[-1]))
    ind_NDC = np.argmin(np.abs(GMT_steps-df_GMT_NDC.iloc[-1]))
    ind_30 = np.argmin(np.abs(GMT_steps-df_GMT_30.iloc[-1]))
    ind_40 = np.argmin(np.abs(GMT_steps-df_GMT_40.iloc[-1]))
    indices=[ind_lb,ind_15,ind_20,ind_NDC,ind_30,ind_40]
    year_range=np.arange(year_start,2100+1) # ROSA so that extrapolation happens after stylized traj are created
    n_years = len(year_range)
    trj = np.empty((n_years,n_steps))
    trj.fill(np.nan)
    trj[0:ind_fut_strtyr,:] = np.repeat(np.expand_dims(df_GMT_15.loc[:GMT_fut_strtyr-1].values,axis=1),n_steps,axis=1)
    trj[ind_fut_strtyr:,0] = GMT_min
    trj[ind_fut_strtyr:,-1] = np.interp(
        x=year_range[ind_fut_strtyr:],
        xp=[GMT_fut_strtyr,year_range[-1]], 
        fp=[GMT_min,GMT_max],
    )
    trj[:,ind_lb] = df_GMT_lb.values
    trj[:,ind_15] = df_GMT_15.values
    trj[:,ind_20] = df_GMT_20.values
    trj[:,ind_NDC] = df_GMT_NDC.values
    trj[:,ind_30] = df_GMT_30.values
    trj[:,ind_40] = df_GMT_40.values
    trj_msk = np.ma.masked_invalid(trj)
    [xx, yy] = np.meshgrid(range(n_steps),range(n_years))
    x1 = xx[~trj_msk.mask]
    y1 = yy[~trj_msk.mask]
    trj_interpd = interpolate.griddata(
        (x1,y1), # only include coords with valid data
        trj[~trj_msk.mask].ravel(), # inputs are valid only, too
        (xx,yy), # then provide coordinates of ourput array, which include points where interp is required (not ravelled, so has 154x24 shape)
    )
    df_GMT_strj = pd.DataFrame(
        trj_interpd, 
        columns=range(n_steps), 
        index=year_range,
    )        

    # ------------------------- End of original AR6 approach --------------------------
    
    # # Rosa: note that forcing the GMT in 2100 manually causes discontinuity
    # modified it to fix this below, the 1.5 and 3.5 limits are now conditions of the interpolation 

    # Desired GMT range and steps
    GMT_min = 1.5
    GMT_max = 3.5
    GMT_steps = np.round(np.arange(GMT_min, GMT_max + 0.001, 0.1), 2)
    n_years = len(df_GMT_strj)

    # Extract the GMT values corresponding to each original column in df_GMT_strj
    # Suppose the original GMT levels are known — e.g.,:
    orig_GMT_levels = GMT_steps_all = np.round(df_GMT_strj.loc[2100].values, 2)  # or however you originally set them

    # Filter to retain only the columns within the 1.5–3.5°C range
    keep_cols = np.where((orig_GMT_levels >= GMT_min) & (orig_GMT_levels <= GMT_max))[0]
    df_GMT_strj_filtered = df_GMT_strj.iloc[:, keep_cols]
    orig_GMT_levels_filtered = orig_GMT_levels[keep_cols]

    # Interpolate across GMT dimension for each year
    df_GMT_strj_clean = pd.DataFrame(index=df_GMT_strj.index, columns=GMT_steps)
    for yr in df_GMT_strj.index:
        yvals = df_GMT_strj_filtered.loc[yr].values
        df_GMT_strj_clean.loc[yr] = np.interp(GMT_steps, orig_GMT_levels_filtered, yvals)

    # Now df_GMT_strj_clean has smooth trajectories that end at 1.5–3.5°C in 2100

    df_GMT_strj = cp(df_GMT_strj_clean)  


    # ROSA: Do the extrapolation past 2100 here after construction of stylized trajectories 
    # instead of on the original AR6 scenarios - better especially for 10-year trend extension

    df_GMT_strj = extend_gmt_to_year_range(df_GMT_strj)
    df_GMT_NDC = extend_gmt_to_year_range(df_GMT_NDC)
    df_GMT_15 = extend_gmt_to_year_range(df_GMT_15)
    df_GMT_20 = extend_gmt_to_year_range(df_GMT_20)
    

    return df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_OS, df_GMT_noOS, ds_GMT_STS, df_GMT_strj



#%%


def calc_model_gmst(experiments=scenarios,
                    GCMs=None,
                    first_ens_member=True,
                    startyear=1850,
                    endyear=2100
    ):
    """
    Calc model GMST from Pangeo
    Todo: move this big fxn to preprocessing
    Make just a function that does it for specific model simulations / ensemble members if its not available in preprocessed file 
    """

    from tqdm.autonotebook import tqdm 
    import intake
    import fsspec
    import intake_esm.cat
    from collections import defaultdict

    # Monkeypatch the method to use applymap()
    def _columns_with_iterables(self):
        if self._df.empty:
            return set()
        has_iterables = (
            self._df.sample(20, replace=True).applymap(type)
            .isin([list, tuple, set])
            .any()
            .to_dict()
        )
        return {column for column, check in has_iterables.items() if check}

    def drop_all_bounds(ds):
        """Drop coordinates like 'time_bounds' from datasets,
        which can lead to issues when merging."""
        drop_vars = [vname for vname in ds.coords
                    if (('_bounds') in vname ) or ('_bnds') in vname]
        return ds.drop(drop_vars)

    def open_dset(df):
        """Open datasets from cloud storage and return xarray dataset."""
        assert len(df) == 1
        ds = xr.open_zarr(fsspec.get_mapper(df.zstore.values[0]), consolidated=True)
        return drop_all_bounds(ds)

    def open_delayed(df):
        """A dask.delayed wrapper around `open_dsets`.
        Allows us to open many datasets in parallel."""
        return dask.delayed(open_dset)(df)

    def get_lat_name(ds):
        """Figure out what is the latitude coordinate for each dataset."""
        for lat_name in ['lat', 'latitude']:
            if lat_name in ds.coords:
                return lat_name
        raise RuntimeError("Couldn't find a latitude coordinate")

    def global_mean(ds):
        """Return global mean of a whole dataset."""

        lat = ds[get_lat_name(ds)]
        weight = np.cos(np.deg2rad(lat))
        weight /= weight.mean()
        other_dims = set(ds.dims) - {'time'}
        return (ds * weight).mean(other_dims,skipna=True)

    intake_esm.cat.ESMCatalogModel.columns_with_iterables = property(_columns_with_iterables)

    # open catalogue from Pangeo 
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")

    if GCMs:
        query = dict(
        experiment_id=experiments,
        table_id='Amon',            # choose to look at atmospheric variables (A) saved at monthly resolution (mon)               
        variable_id=['tas'],        # choose to look at near-surface air temperature (tas) as our variable
        source_id = GCMs
        )
    else:
        query = dict(
        experiment_id=experiments,
        table_id='Amon',            # choose to look at atmospheric variables (A) saved at monthly resolution (mon)               
        variable_id=['tas'],        # choose to look at near-surface air temperature (tas) as our variable
        )

    col_subset = col.search(require_all_on=["source_id"], **query)

    def get_run_number(member_id):
        """Extract the numeric part of the r# from a member_id string."""
        match = re.match(r"r(\d+)", member_id)
        return int(match.group(1)) if match else float('inf')

    if first_ens_member:
        col_subset = col.search(require_all_on=["source_id"], **query)
        df = col_subset.df.copy()
        # add a temporary numeric column
        df['run_number'] = df['member_id'].map(get_run_number)
        # sort by source_id, experiment_id, then numeric run number
        df = df.sort_values(['source_id', 'experiment_id', 'run_number',"member_id"])
        # drop the helper column
        df = df.drop(columns='run_number')
        # Drop duplicates — keep the first ensemble member per model & experiment
        df = df.drop_duplicates(subset=["source_id", "experiment_id"], keep="first")
        # optional: check result
        print(df.groupby("source_id")[["experiment_id", "member_id"]].nunique())
        # make metadata object 
        df_metadata = df.groupby(by=['source_id', 'experiment_id'])["member_id"].agg(lambda x: ', '.join(x)).reset_index()
        # update catalog with this  
        col_subset.esmcat._df = df
        col_subset
    else:
        ValueError('function not defined for multiple ensemble members')

        # To do: make this possible or select with a dict ! 

    dsets = defaultdict(dict) 

    for group, df in col_subset.df.groupby(by=['source_id', 'experiment_id',"member_id"]):
        dsets[group[0]][group[1]] = open_delayed(df)

        # try here to save member_id so you have it for traceability!! 
        # and try to do this for each member_id and have them saved as a dimension in the ds...

    dsets_ = dask.compute(dict(dsets))[0]

    expt_da = xr.DataArray(expts, dims='experiment_id', name='experiment_id',
                       coords={'experiment_id': expts})

    dsets_aligned = {}

    for k, v in tqdm(dsets_.items()):
        print(k)
        expt_dsets = v.values()
        if any([d is None for d in expt_dsets]):
            print(f"Missing experiment for {k}")
            continue
        
        for ds in expt_dsets:
            ds.coords['year'] = ds.time.dt.year
            print(ds.experiment_id)
            print(ds.time.dt.year.values[0],ds.time.dt.year.values[-1])
            
        # workaround for
        # https://github.com/pydata/xarray/issues/2237#issuecomment-620961663
        dsets_ann_mean = [v[expt].pipe(global_mean)
                                .swap_dims({'time': 'year'})
                                .drop('time')
                                .coarsen(year=12).mean(skipna=True)
                        for expt in expts]
        
        # align everything 
        dsets_aligned[k] = xr.concat(dsets_ann_mean, join='outer',
                                    dim=expt_da)
    
    with progress.ProgressBar():
        dsets_aligned_ = dask.compute(dsets_aligned)[0]
    
    source_ids = list(dsets_aligned_.keys())

    source_da = xr.DataArray(source_ids, dims='source_id', name='source_id',
                            coords={'source_id': source_ids})

    big_ds = xr.concat([ds.reset_coords(drop=True)
                        for ds in dsets_aligned_.values()],
                        dim=source_da)

    df_gmst_all = big_ds.sel(year=slice(startyear, endyear)).to_dataframe().reset_index()

    return df_gmst_all, df_metadata




