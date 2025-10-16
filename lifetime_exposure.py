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
    GMT_extra_trajectories_names = None,
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


                # 4) if needed, repeat mean of last 10 years until entire period of interest is covered - NOTE: do you need da_AFA here??? you're not using it in this fxn
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
                
                
                # 5) run GMT mapping for stylized trajectories 

                # --------------------------------------------------------- #
                # Step 1: Compute the minimum absolute difference (distance)
                # between the ISIMIP GMT trajectory (d_isimip_meta[i]['GMT'])
                # and each reference GMT trajectory (e.g. 1.5°C, 2.0°C, NDC…).
                # This tells us "how close" the ISIMIP GMT is to the target GMT.
                # --------------------------------------------------------- #

                # --------------------------------------------------------- #
                # Step 2: Get the indices (years) where the ISIMIP GMT is   #
                # closest to the reference GMT trajectories.                #
                # (argmin returns the index of the minimum distance).       #
                # --------------------------------------------------------- #

                # ----------------------------------------------------------- #
                # Step 3: Store the maximum difference (worst case distance)  #
                # between ISIMIP GMT and each reference trajectory.           #
                # This allows checking whether the ISIMIP curve ever deviates #
                # too much from the reference.                                #
                # ----------------------------------------------------------- #

                # ------------------------------------------------------------ #
                # Step 4: Define validity flags (True/False).                  #
                # A trajectory is considered "valid" if the maximum difference #
                # never exceeds a chosen threshold (RCP2GMT_maxdiff_threshold) #
                # ------------------------------------------------------------ #

                # ---------------------------------------------------------------- #
                # Step 5: Save the indices of the years where the GMT              #
                # trajectory is closest to each target. This allows later          #
                # remapping for each trajectory                                    #
                # ---------------------------------------------------------------- #

                # do this for stylized trajectories
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

                
                # do for any extra trajectories - each one should be a df with a single column (i.e. a single pathway)
                if GMT_extra_trajectories:

                    if isinstance(GMT_extra_trajectories, str):
                        GMT_extra_trajectories = [GMT_extra_trajectories]

                    for df_traj, name in zip(GMT_extra_trajectories,GMT_extra_trajectories_names ): 
                        RCP2GMT_diff = np.min(np.abs(d_climate_data_meta[i]['GMT'].values - df_traj.values.transpose()), axis=0)
                        ind_RCP2GMT= np.argmin(np.abs(d_climate_data_meta[i]['GMT'].values - df_traj.values.transpose()), axis=0)
                        d_climate_data_meta[i][f'GMT_{name}_maxdiff'] = np.nanmax(RCP2GMT_diff)
                        d_climate_data_meta[i][f'GMT_{name}_valid'] = np.nanmax(RCP2GMT_diff) < max_diff_valid
                        d_climate_data_meta[i][f'ind_RCP2GMT_{name}'] = ind_RCP2GMT


                # update counter
                i += 1

                # what are you doing with da_AFA here??? why do you need to load the data? if not using 
                # was getting saved as pickle - could do elsewhere, or output in fxn return ?? 

    return d_climate_data_meta 
# %%


def calc_lifetime_exposure(
    d_isimip_meta, 
    df_countries, 
    countries_regions, 
    countries_mask, 
    da_population, 
    df_life_expectancy_5,
    ds_regions,
    da_cohort_size_regions,
    flags,
):


    pass


    # see convo with ChatGPT!! 


    return 