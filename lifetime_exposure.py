"""
Demographics4Climate : Population and demographics for climate science analysis
----------------------------------------------

2024 Update

Calculate lifetime exposure 

Updated to UNWPP2024 

""" 




import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd # can maybe delete if i dont open geojson in the end 
from scipy import interpolate
import glob, os, re, sys
import warnings
import openpyxl 


script_dir = os.path.abspath( os.path.dirname( __file__ ) )


def load_climate_data(
    extremes, # e.g. "FWI95d"
    model_names, 
    climatedata_dir = None, # structure should be 
    scenarios = None,
    df_GMT_strj = None,
    rolling_window=21,
    model_gmst_dir = None,
    gmt_mapping_method = 'year-to-year' # 
    ): 


    """ work in progress !!!! """

    def load_gmst_per_model():
        pass



    print('Processing climate data')

    # initialise counter, metadata dictionary, pic list, pic meta, and 
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

                filepath = glob.glob(os.path.join(climatedata_dir, scenario, model, f'*{model}*{extreme}.nc'))[0]
                print(f'Loading {filepath}')
            
                # load rcp data (AFA: Area Fraction Affected) - and manually add correct years
                da_ssp = xr.open_dataarray(filepath)

                # save metadata
                d_isimip_meta[i] = {
                    'extreme': extreme,
                    'model': model,
                    'scenario': scenario,
                }

                #load associated historical variable
                filepath_hist = glob.glob(os.path.join(climatedata_dir, 'historical', model, f'*{model}*{extreme}.nc'))[0]
                da_hist = xr.open_dataarray(filepath_hist)


                # ROSA edited until here !! 

                # load GMT for rcp and historical period - note that these data are in different files
                if d_isimip_meta[i]['gcm'] == 'hadgem2-es': # .upper() method doesn't work for HadGEM2-ES on linux server (only Windows works here)
                    file_names_gmt = glob.glob(data_dir+'isimip/DerivedInputData/globalmeans/tas/HadGEM2-ES/*.fldmean.yearmean.txt') # ignore running mean files
                else:
                    file_names_gmt = glob.glob(data_dir+'isimip/DerivedInputData/globalmeans/tas/'+d_isimip_meta[i]['gcm'].upper()+'/*.fldmean.yearmean.txt') # ignore running mean files
                file_name_gmt_fut = [s for s in file_names_gmt if d_isimip_meta[i]['rcp'] in s]
                file_name_gmt_his = [s for s in file_names_gmt if '_historical_' in s]
                file_name_gmt_pic = [s for s in file_names_gmt if '_piControl_' in s]

                GMT_fut = pd.read_csv(
                    file_name_gmt_fut[0],
                    delim_whitespace=True,
                    skiprows=1,
                    header=None).rename(columns={0:'year',1:'tas'}).set_index('year')
                GMT_his = pd.read_csv(
                    file_name_gmt_his[0],
                    delim_whitespace=True, 
                    skiprows=1, 
                    header=None).rename(columns={0:'year',1:'tas'}).set_index('year')
                GMT_pic = pd.read_csv(
                    file_name_gmt_pic[0],
                    delim_whitespace=True, 
                    skiprows=1, 
                    header=None).rename(columns={0:'year',1:'tas'}).set_index('year')

                # concatenate historical and future data
                da_AFA = xr.concat([da_AFA_his,da_AFA_rcp], dim='time')
                df_GMT = pd.concat([GMT_his,GMT_fut])

                # convert GMT from absolute values to anomalies - use data from pic until 1861 and from his from then onwards
                df_GMT = df_GMT - pd.concat([GMT_pic.loc[year_start_GMT_ref:np.min(GMT_his.index)-1,:], GMT_his.loc[:year_end_GMT_ref,:]]).mean()

                # if needed, repeat mean of last 10 years until entire period of interest is covered
                if da_AFA.time.max() < year_end: 
                    da_AFA_lastyear = da_AFA.sel(time=slice(da_AFA.time.max()-9,da_AFA.time.max())).mean(dim='time').expand_dims(dim='time',axis=0)
                    GMT_lastyear = df_GMT.iloc[-10:,:].mean() # mean of last 10 years to fill time span 

                    for year in range(da_AFA.time.max().values+1,year_end+1): 
                        da_AFA = xr.concat([da_AFA,da_AFA_lastyear.assign_coords(time = [year])], dim='time')
                        if len(df_GMT) < 439: # necessary to avoid this filling from 2100-2113 if GMTs already go to 2299
                            df_GMT = pd.concat([df_GMT,pd.DataFrame(data={'tas':GMT_lastyear['tas']},index=[year])])

                # retain only period of interest
                da_AFA = da_AFA.sel(time=slice(year_start,year_end))
                df_GMT = df_GMT.loc[year_start:year_end,:]
                
                # rolling mean option
                if flags['rm'] == 'no_rm':
                    
                    pass
                
                else:

                    if flags['rm_config'] =='21':
                    
                        df_GMT = df_GMT.rolling(window=21,min_periods=10,center=True,).mean()

                    if flags['rm_config'] =='11':
                        
                        df_GMT = df_GMT.rolling(window=11,min_periods=5,center=True,).mean()

                # save GMT in metadatadict
                d_isimip_meta[i]['GMT'] = df_GMT 

                # run GMT mapping for stylized trajectories (repeat above but for dataframe of all trajectories)

                # get ISIMIP GMT indices closest to GMT trajectories        
                # store GMT maxdiffs and indices in metadatadict
                d_isimip_meta[i]['GMT_strj_maxdiff'] = np.empty_like(np.arange(len(df_GMT_strj.columns)))
                d_isimip_meta[i]['GMT_strj_valid'] = np.empty_like(np.arange(len(df_GMT_strj.columns)))
                d_isimip_meta[i]['ind_RCP2GMT_strj'] = np.empty_like(df_GMT_strj.values)
                
                for step in range(len(df_GMT_strj.columns)):
                    RCP2GMT_diff = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_strj.loc[:,step].values.transpose()), axis=0)
                    d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step] = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_strj.loc[:,step].values.transpose()), axis=0)
                    d_isimip_meta[i]['GMT_strj_maxdiff'][step] = np.nanmax(RCP2GMT_diff)
                    d_isimip_meta[i]['GMT_strj_valid'][step] = np.nanmax(RCP2GMT_diff) < RCP2GMT_maxdiff_threshold
                    
                d_isimip_meta[i]['ind_RCP2GMT_strj'] = d_isimip_meta[i]['ind_RCP2GMT_strj'].astype(int)

                # update counter
                i += 1
    


    return d_isimip_meta 