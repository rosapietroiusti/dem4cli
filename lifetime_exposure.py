"""
Demographics4Climate : Population and demographics for climate science analysis
----------------------------------------------

2024 Update

Calculate lifetime exposure 

Updated to UNWPP2024 

""" 
#%%



import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd # can maybe delete if i dont open geojson in the end 
from scipy import interpolate
import glob, os, re, sys
import warnings
import openpyxl 

from _settings import * 

script_dir = os.path.abspath( os.path.dirname( __file__ ) )

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






def calc_gmt_anomaly_correction(
    filepath_gmst_obs=os.path.join(data_dir,'gmst-obs/GCH_time_series_of_annual_global_temperatures_1850-2024_wrt_1991-2020.csv'),
    col='ERA5',
    gmt_anomaly_baseline_period = (1850,1900),
    ):

    df_gmt_obs = pd.read_csv(filepath_gmt_obs,skiprows=3,index_col=0, na_values=-999,usecols=[0,1,2,3,4,5])

    anomaly_correction = gmt_obs[[col]].loc[gmt_anomaly_baseline_period[0]:gmt_anomaly_baseline_period[1]].mean()[0]

    return anomaly_correction







def load_climate_data(
    extremes,                   # e.g. "FWI95d"
    model_names,                # GCMs
    df_GMT_strj,                # stylized trajectories
    GMT_extra_trajectories = None,
    climatedata_dir = None,     # structure should be : climatedata_dir/scenario/model/'*{model}*{extreme}.nc
    filepath_model_gmst = os.path.join(data_dir, 'gmst-models/gmst_models_1850_2100_fwi.csv'),
    scenarios = None,
    smoothing=True,
    rolling_window=21,
    min_periods=11,
    gmt_anomaly_baseline_period = (1850,1900),
    gmt_anomaly_correction = 0,
    year_start=1950,
    year_end = 2119,            # 2025 + max life expectancy
    gmt_mapping_method = 'year-to-year', 
    max_diff_valid = .5, 
    ): 


    """ work in progress !!!! """

#%%


    print('Processing climate data')

    # initialise counter, metadata dictionary
    i = 1
    d_climate_data_meta = {}

    if isinstance(extremes, str):
        extremes = [extremes]

    # loop over extremes e.g. FWId95
    for extreme in extremes:

        print(f'Processing for {extreme}')
        
        # loop over models e.g. CanESM5
        for model in model_names: 

            for scenario in scenarios:

                # 1) metadata
                d_climate_data_meta[i] = {
                    'extreme': extreme,
                    'model': model,
                    'scenario': scenario,
                }

                # 2) load climate data: annual count of exceedances of threshold (already preprocessed)
                filepath = glob.glob(os.path.join(climatedata_dir, scenario, model, f'*{model}*{extreme}.nc'))[0]
                print(f'Loading {filepath}')
                da_rcp = xr.open_dataarray(filepath)

                #load associated historical variable
                filepath = glob.glob(os.path.join(climatedata_dir, 'historical', model, f'*{model}*{extreme}.nc'))[0]
                print(f'Loading {filepath}')
                da_hist = xr.open_dataarray(filepath)

                # concat (AFA = area fraction affected, binary data yes/no affected)
                da_AFA = xr.concat([da_hist,da_rcp], dim='time')
                da_AFA['time'] = da_AFA.time.dt.year


                # 3) load GMT 
                df_GMT = pd.read_csv(filepath_model_gmst,index_col=0)
                df_GMT['year'] = df_GMT['year'].astype(int)
                modelname = model if model != 'EC-EARTH' else 'EC-Earth3'
                df_GMT_hist = df_GMT[(df_GMT['experiment_id']=='historical') & (df_GMT['source_id']==modelname)].set_index('year').drop(columns=['experiment_id','source_id']).dropna()
                df_GMT_rcp = df_GMT[(df_GMT['experiment_id']==scenario) & (df_GMT['source_id']==modelname)].set_index('year').drop(columns=['experiment_id','source_id']).dropna()

                # concatenate historical and future GMT data
                df_GMT = pd.concat([df_GMT_hist,df_GMT_rcp])

                # convert GMT from absolute values to anomalies - if you use a non 1850-1900 period should provide a gmt_anomaly_correction
                df_GMT = df_GMT - df_GMT.loc[gmt_anomaly_baseline_period[0] : gmt_anomaly_baseline_period[1]].mean() + gmt_anomaly_correction


                # if needed, repeat mean of last 10 years until entire period of interest is covered
                if da_AFA.time.max() < year_end: 
                    da_AFA_lastyear = da_AFA.isel(time=slice(-10, None)).mean(dim='time').expand_dims(dim='time',axis=0)
                    GMT_lastyear = df_GMT.iloc[-10:,:].mean() # mean of last 10 years to fill time span 

                    for year in range(da_AFA.time.max().values+1,year_end+1): 
                        da_AFA = xr.concat([da_AFA,da_AFA_lastyear.assign_coords(time = [year])], dim='time')
                        if len(df_GMT) < 439: # necessary to avoid this filling from 2100-2113 if GMTs already go to 2299
                            df_GMT = pd.concat([df_GMT,pd.DataFrame(data={'tas':GMT_lastyear['tas']},index=[year])])

                # retain only period of interest
                da_AFA = da_AFA.sel(time=slice(year_start,year_end))
                df_GMT = df_GMT.loc[year_start:year_end,:]
                
                # rolling mean 
                df_GMT = df_GMT.rolling(window=rolling_window,min_periods=min_periods,center=True).mean()

                # save GMT in metadatadict
                d_climate_data_meta[i]['GMT'] = df_GMT 


                # run GMT mapping for stylized trajectories (repeat above but for dataframe of all trajectories)

                # get ISIMIP GMT indices closest to GMT trajectories        
                # store GMT maxdiffs and indices in metadatadict
                d_climate_data_meta[i]['GMT_strj_maxdiff'] = np.empty_like(np.arange(len(df_GMT_strj.columns)))
                d_climate_data_meta[i]['GMT_strj_valid'] = np.empty_like(np.arange(len(df_GMT_strj.columns)))
                d_climate_data_meta[i]['ind_RCP2GMT_strj'] = np.empty_like(df_GMT_strj.values)
                
                # loop over each trajectory 
                for step in range(len(df_GMT_strj.columns)):
                    RCP2GMT_diff = np.min(np.abs(d_climate_data_meta[i]['GMT'].values - df_GMT_strj.iloc[:,step].values.transpose()), axis=0)
                    d_climate_data_meta[i]['ind_RCP2GMT_strj'][:,step] = np.argmin(np.abs(d_climate_data_meta[i]['GMT'].values - df_GMT_strj.iloc[:,step].values.transpose()), axis=0)
                    d_climate_data_meta[i]['GMT_strj_maxdiff'][step] = np.nanmax(RCP2GMT_diff)
                    d_climate_data_meta[i]['GMT_strj_valid'][step] = np.nanmax(RCP2GMT_diff) < max_diff_valid
                    
                d_climate_data_meta[i]['ind_RCP2GMT_strj'] = d_climate_data_meta[i]['ind_RCP2GMT_strj'].astype(int)


                # update counter
                i += 1
    
                # TO DO : ADD ALSO OTHER TRAJECTORIES ! 


                # what are you doing with da_AFA here??? why do you need to load the data? if not using 

#%%
    return d_climate_data_meta 
# %%
