"""
Demographics4Climate : Population and demographics for climate science analysis
----------------------------------------------

2025 Update

Calculate land fraction exposed or area-weighted average, and lifetime exposure 

""" 
#%%


import glob, os
import numpy as np
import xarray as xr
import pandas as pd

from ._settings import * 

script_dir = os.path.abspath( os.path.dirname( __file__ ) )

#%%




def calc_gmt_anomaly_correction(
    filepath_gmst_obs=os.path.join('data', 'gmst-obs/GCH_time_series_of_annual_global_temperatures_1850-2024_wrt_1991-2020.csv'),
    col='ERA5',
    gmt_anomaly_baseline_period=(1850, 1900),
    skiprows=3,index_col=0, na_values=-999,
    **kwargs
):
    """
    Calculate mean warming in reference period from observational GMT wrt. 1850–1900.

    If you want to rebaseline everything to a different period (e.g., bias adjustment time-period),
    you can provide this as `gmt_anomaly_baseline_period` and then in load_climate_data use the output
    of calc_gmt_anomaly_correction as the argument gmt_anomaly_correction to re-express everything
    wrt. 1850–1900.

    Inputs
        filepath_gmst_obs (str):                        Filepath to the observational GMST dataset.
        col (str):                                      Column name to use (e.g., 'ERA5', 'HadCRUT5', etc.).
        gmt_anomaly_baseline_period (tuple(int, int)):  Start and end years of the baseline period.
        **kwargs :                                      Additional keyword arguments passed to `pandas.read_csv`.
                                                        Defaults if not overwritten: skiprows=3,index_col=0, na_values=-999

    Returns
        (float) Mean anomaly correction over the specified baseline period.
    """

    # Default read_csv args
    read_csv_kwargs = dict(skiprows=skiprows,index_col=index_col, na_values=na_values)
    # Allow user to override defaults
    read_csv_kwargs.update(kwargs)
    # open gmt file 
    df_gmt_obs = pd.read_csv(filepath_gmst_obs, **read_csv_kwargs)

    # ensure index is year (if not already)
    if not isinstance(df_gmt_obs.index, pd.Index):
        df_gmt_obs.set_index(df_gmt_obs.columns[0], inplace=True)

    anomaly_correction = (
        df_gmt_obs[[col]]
        .loc[gmt_anomaly_baseline_period[0]:gmt_anomaly_baseline_period[1]]
        .mean()
        .iloc[0]
    )

    return anomaly_correction


@timeit
def load_climate_data(
    extremes,                   # e.g. "FWI95d"
    model_names,                # GCMs
    df_GMT_strj,                # stylized trajectories
    GMT_extra_trajectories = None,
    GMT_extra_trajectories_names = None,
    filepath_model_gmst = os.path.join(data_dir, 'gmst-models/gmst_models_1850_2100_fwi.csv'),
    scenarios = None,
    rolling_window=21,
    min_periods=11,
    gmt_anomaly_baseline_period = (1850,1900),
    gmt_anomaly_correction = 0,
    year_start=1950,
    year_end = 2119,            # 2025 + max life expectancy
    gmt_mapping_method = 'year-to-year', 
    max_diff_valid = .2, 
    ): 


    """ 
    Loads 'recipe' for GMT-remapping climate models to target GMT trajectories. 

    Takes information on model/experiment/ensemble member-specific GMT timseries. If provided, adjusts this to a baseline period, 
    and if provided additionally adds a correction (e.g. if you want to adjust to 1985-2014 but still want anomalies expressed 
    relative to 1850-1900).
    Takes information on target GMT pathways to emulate from df_GMT_strj and optionally extra scenarios. Smooths with indicated rolling
    window and matches year-to-year the closest matching year, allowing for a model year to be used more than once. This leads to an 
    underestimation of natural variability wrt. original simulation, but tests show that at decadal timescale averages are well represented, 
    making this suitable for lifetime exposure calculations. 
    Outputs a dictionary with the remapping recipe, including a check of whether any attempted emulation is not valid (i.e. target GMT is too different
    from original GMT e.g. remapping a too hot scenario with an SSP126) 


    
    Inputs




    Returns
        d_climate_data_meta (dict):     Dictionary, each item is one model simulation and single extreme. 
                                        'extreme', 'model', 'scenario' : (str) information on extreme, GCM and experiment of original simulation
                                        'GMT': (df) dataframe with GMT timeseries of original simulation, if provided adjusted to baseline period/correction
                                        'GMT_strj_maxdiff'  : max difference target to original GMT in any remapping attempt
                                        'GMT_strj_valid'    : whether maxdiff is larger than validity threshold and therefore invalid
                                        'ind_RCP2GMT_strj'  : remapping recipe (indexes)
                                        'GMT_{}_maxdiff', 'GMT_{}_valid', 'ind_RCP2GMT_{}' : with {} filled by GMT_extra_trajectories_names 
                                                                                             for all provided GMT stylized trajectories, in the form



    """

    print('Processing climate data')

    # initialise counter, metadata dictionary
    i = 1
    d_climate_data_meta = {}

    if isinstance(extremes, str):
        extremes = [extremes]

    # remove historical if provided in list of scenarios
    scenarios = [s for s in scenarios if s not in ('historical', 'hist')]

    # loop over extremes e.g. FWId95
    # this is not necessary actually!!! same models same remapping recipe!! 
    #TODO: could delete this.... 
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


                # 2) load GMT 
                df_GMT = pd.read_csv(filepath_model_gmst,index_col=0)
                df_GMT['year'] = df_GMT['year'].astype(int)
                modelname = model if model != 'EC-EARTH' else 'EC-Earth3'
                df_GMT_hist = df_GMT[(df_GMT['experiment_id']=='historical') & (df_GMT['source_id']==modelname)].set_index('year').drop(columns=['experiment_id','source_id']).dropna()
                df_GMT_rcp = df_GMT[(df_GMT['experiment_id']==scenario) & (df_GMT['source_id']==modelname)].set_index('year').drop(columns=['experiment_id','source_id']).dropna()

                # concatenate historical and future GMT data
                df_GMT = pd.concat([df_GMT_hist,df_GMT_rcp])

                # convert GMT from absolute values to anomalies - if you use a non 1850-1900 period should provide a gmt_anomaly_correction
                df_GMT = df_GMT - df_GMT.loc[gmt_anomaly_baseline_period[0] : gmt_anomaly_baseline_period[1]].mean() + gmt_anomaly_correction


                # 3) if needed, repeat mean of last 10 years until entire period of interest is covered - NOTE: do you need da_AFA here??? you're not using it in this fxn
                #if da_AFA.time.max() < year_end: 
                if df_GMT.index.max() < year_end: 
                    #da_AFA_lastyear = da_AFA.isel(time=slice(-10, None)).mean(dim='time').expand_dims(dim='time',axis=0)
                    GMT_lastyear = df_GMT.iloc[-10:,:].mean() # mean of last 10 years to fill time span 

                    for year in range(df_GMT.index.max()+1,year_end+1): 
                        #da_AFA = xr.concat([da_AFA,da_AFA_lastyear.assign_coords(time = [year])], dim='time')
                        if len(df_GMT) < 439: # necessary to avoid this filling from 2100-2113 if GMTs already go to 2299
                            df_GMT = pd.concat([df_GMT,pd.DataFrame(data={'tas':GMT_lastyear['tas']},index=[year])])

                # retain only period of interest
                #da_AFA = da_AFA.sel(time=slice(year_start,year_end))
                df_GMT = df_GMT.loc[year_start:year_end,:]

                # rolling mean 
                df_GMT = df_GMT.rolling(window=rolling_window,min_periods=min_periods,center=True).mean()

                # save GMT in metadatadict
                d_climate_data_meta[i]['GMT'] = df_GMT
                
                # 4) run GMT mapping recipe  for stylized trajectories 

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

    return d_climate_data_meta 



    #%%

def get_countries_of_region(
    region, 
    df_countries,
): 

    # Get list of member countries from region
    member_countries = df_countries.loc[df_countries['region']==region]['name'].values

    # not region but income group
    if len(member_countries) == 0: 
        member_countries = df_countries.loc[df_countries['incomegroup']==region]['name'].values

    # get all countries for the world
    if region == 'World':
        member_countries = df_countries['name'].values

    return member_countries 



#%%




def calc_weighted_fldmean(
    da, 
    countries_mask,
    ind_country, 
    weights=None,
    areaweighted=False,
    mask=True
):
    def get_lat_name(da):
        """Figure out what is the latitude coordinate for each dataset."""
        for lat_name in ['lat', 'latitude']:
            if lat_name in da.coords:
                return lat_name
        raise RuntimeError("Couldn't find a latitude coordinate")
    
    def weighted_mean(da, weights, areaweighted):
        """Return weighted mean of dataarray."""
        if weights is not None:
            # weight the AFA of the country under study by the size of its population over time at the grid cell or provide gridcell area file 
            # slice to save memory on alignment 
            if "time" in weights.dims:
                weights = weights.sel(time=da.time)
        elif areaweighted: 
            lat = da[get_lat_name(da)]
            weights = np.cos(np.deg2rad(lat))      

        other_dims = set(da.dims) - {'time'}

        return da.weighted(weights).mean(other_dims,skipna=True).reset_coords(drop=True)

    if mask:
        # only keeps the AFA data for the country under study, for the others a NaN value is attributed
        if np.isscalar(ind_country) or len(ind_country) == 1:

            da_masked = da.where(countries_mask == ind_country)
        
        # if more countries are provided, combine the different masks 
        elif len(ind_country) > 1:
            
            mask = countries_mask.isin(ind_country)
            da_masked = da.where(mask)

        da_weighted_fldmean = weighted_mean(da_masked, weights, areaweighted)

        del da_masked

    else:
        # if already masked outside of the fxn 
        da_weighted_fldmean = weighted_mean(da, weights, areaweighted)

    return da_weighted_fldmean



#%%

@timeit
def load_climate_data_array(climatedata_dir,
                    scenario,
                    model,
                    extreme,
                    year_start,
                    year_end,
                    bbox=None,
                    smoothing_window=None
                    ):
    """" Load climate data, concat historical + rcp, clean data. Assumes there is only one variable of interest"""

    # Auxiliary function to slice  dataset to a particular region and time 
    def cut_to_region_time(da):
        # rename for compatibility with population objects
        if 'latitude' in da.coords:
            da = da.rename({'latitude':'lat', 'longitude':'lon'})
        # slice time
        if da.time.dtype == 'datetime64[ns]':
            da['time'] = da['time'].dt.year
        else:
            raise ValueError(f'time undefined for array {da}')
        # cut space
        if bbox is None:
            return da.sel(time=slice(year_start, year_end))
        latmin, latmax, lonmin, lonmax = bbox 
        if da.lat.values[0] < da.lat.values[-1]: # check if lat is increasing or decreasing
            return da.sel(
                lat=slice(latmin, latmax), lon=slice(lonmin, lonmax), time=slice(year_start, year_end)
                )
        else:
            return da.sel(
                lat=slice(latmax, latmin), lon=slice(lonmin, lonmax), time=slice(year_start, year_end)
                )

    if scenario != 'reanalysis':
        filepath_hist = glob.glob(os.path.join(climatedata_dir, 'historical', model, f'*{model}*{extreme}.nc'))[0]
        filepath_rcp = glob.glob(os.path.join(climatedata_dir, scenario, model, f'*{model}*{extreme}.nc'))[0]
        print(f'Loading {filepath_hist}')
        print(f'Loading {filepath_rcp}')
        da_AFA = xr.open_mfdataset(
                    [filepath_hist, filepath_rcp],
                    combine='nested',
                    concat_dim='time',
                    decode_coords='all',
                    preprocess=cut_to_region_time,
                )

    else:
        filepath = glob.glob(os.path.join(climatedata_dir, scenario, model, f'*{model}*{extreme}.nc'))[0]
        print(f'Loading {filepath}')
        da_AFA = xr.open_mfdataset(
                    [filepath],
                    combine='nested',
                    concat_dim='time',
                    decode_coords='all',
                    preprocess=cut_to_region_time,
                )


    VAR = list(da_AFA.data_vars)[0] # assumes there is only one variable of interest ! 
    da_AFA = da_AFA[VAR]

    # 4) if needed, repeat mean of last 10 years until entire period of interest is covered 
    if da_AFA.time.max() < year_end: 
        da_AFA_lastyear = da_AFA.isel(time=slice(-10, None)).mean(dim='time').expand_dims(dim='time',axis=0)
        for year in range(da_AFA.time.max().values+1,year_end+1): 
            da_AFA = xr.concat([da_AFA,da_AFA_lastyear.assign_coords(time = [year])], dim='time')

    # to take a more RIME-like approach - smooth the climate data also - no natural variability only forced signal - TESTING
    if smoothing_window:
        da_AFA = da_AFA.rolling(time=smoothing_window, center=True, min_periods=1).mean()

    return da_AFA









@timeit
def calc_landfraction_exposed(
    d_climate_data_meta, 
    df_countries, 
    countries_regions, 
    countries_mask, 
    climatedata_dir, # structure should be : climatedata_dir/scenario/model/'*{model}*{extreme}.nc
    GMT_labels = None , 
    GMT_extra_trajectories_names=None,
    year_start=1950,
    year_end=2119,
    bbox=None,
    weights=None,
    areaweighted=True,
    convert_to_binary=False,  
    convert_to_binary_threshold=0,  
    smoothing_window=None  # change this arg name in load_climate data 
):
    """
    Calc area-weighted average of your input hazard dataset per country and per region. 
    Results are given per origina RCP model years and after GMT-remapping 
    year-to-year based on GMT_labels and GMT_extra_trajectories, 
    with the remapping recipe defined in d_climate_data_meta. 

    Note: if your input data is annual binary (0/1) this will give the 
    fraction of land exposed. If your input data is the number of annual 
    exceedances this will give
    the average number of days per year the country is experiencing. 

    Inputs:
        GMT_labels (list):                      e.g. df_GMT_strj.columns

        convert_to_binary (bool)                if your data array is number of exceedances per year, 
                                                and you set this to True you get fraction of land area exposed 
                                                to more than 0 days 

        convert_to_binary_threshold (int)       modify the minimum number of days to convert to binary

        smoothing_window (int)                  window to smooth impact data, default:None

        bbox (opt)                              latmin, latmax, lonmin, lonmax

    Returns:
        ds_lfe_perregion_perrun (ds):       per region, per model year or remapped year, 
                                            per scenario the area-weighted fraction of region exposed 
        
        ds_lfe_percountry_perrun (ds):      per country, per model year or remapped year, 
                                            per scenario the area-weighted fraction of region exposed 
        
        region_names (list):                region names to understand ds_lfe_perregion_perrun


    TODO:
    - ideally would be nice to have option to give GMT_strj or extra_trajectories as 
    user wants to do - i.e. not make it necessary to have both
    tried to implement but was getting fidgety so stopped, and so they can just give 
    you one and its also ok (see in scraps doc)

    - could be better to loop first regions then suffix to not have to reselect src all the time 
    """

    # helper function to get areaweights for area-weighted average (for regular grid) 
    def get_areaweights(da):

        def get_lat_name(da):
            """Get name of the latitude coordinate for each dataset."""
            for lat_name in ['lat', 'latitude']:
                if lat_name in da.coords:
                    return lat_name
            raise RuntimeError("Couldn't find a latitude coordinate")

        lat = da[get_lat_name(da)]
        weights = np.cos(np.deg2rad(lat))  

        # Broadcast to match da
        weights = weights.broadcast_like(da)

        return weights  


    # 1) Build Dataset for regions result

    # get regions and income groups and 'World' (=all countries)
    region_names = np.concatenate([df_countries['region'].dropna().unique(),
                            df_countries['incomegroup'].dropna().unique(),
                            ['World'] # not sure how useful to keep 'World' if you are using a bbox! 
    ])

    nregions = len(region_names)

    # Shared shape for all variables
    year_range = np.arange(year_start, year_end+1)

    shape = (len(d_climate_data_meta), nregions, len(year_range))

    shape_strj = (len(d_climate_data_meta), nregions, len(year_range), len(GMT_labels))

    # Build the data_vars dictionary in a loop
    if GMT_extra_trajectories_names:
        var_suffixes = ['RCP']+GMT_extra_trajectories_names 
    else:
        var_suffixes = ['RCP']

    data_vars = {}
    for suffix in var_suffixes:
        var_name = f'landfrac_peryear_perregion_{suffix}'
        data_vars[var_name] = (
            ['run', 'region', 'time_ind'],
            np.full(shape, np.nan)
        )

    # add the strj variable, it has a different shape 
    data_vars['landfrac_peryear_perregion_strj'] = (
            ['run', 'region', 'time_ind', 'GMT'],
            np.full(shape_strj, np.nan)

    )

    # Build the full dataset
    ds_lfe_perregion_perrun = xr.Dataset(
        data_vars=data_vars,
        coords={
            'run': ('run', np.arange(1, len(d_climate_data_meta) + 1)),
            'region': ('region', np.arange(0, nregions)),
            'time_ind': ('time_ind', np.arange(0, len(year_range), 1)),
            'GMT': ('GMT', GMT_labels)
        }
    )


    # 2) Build Dataset for country result
    shape = (
        len(d_climate_data_meta),
        len(df_countries['name'].values),
        len(year_range)
    )
    shape_strj = (len(d_climate_data_meta), len(df_countries['name'].values), len(year_range), len(GMT_labels))

    # Build the data_vars dictionary in a loop
    data_vars = {}
    for suffix in var_suffixes:
        var_name = f'landfrac_peryear_percountry_{suffix}'
        data_vars[var_name] = (
            ['run', 'country', 'time_ind'],
            np.full(shape, np.nan)
        )

    data_vars['landfrac_peryear_percountry_strj'] = (
            ['run', 'country', 'time_ind', 'GMT'],
            np.full(shape_strj, np.nan)

    )

    # Build the dataset
    ds_lfe_percountry_perrun = xr.Dataset(
        data_vars=data_vars,
        coords={
            'run': ('run', np.arange(1, len(d_climate_data_meta) + 1)),
            'country': ('country', df_countries['name'].values),
            'time_ind': ('time_ind', np.arange(0, len(year_range), 1)),
            'GMT': ('GMT', GMT_labels)
        }
    )


    for i in list(d_climate_data_meta.keys()): 

        print('                         🟠 Remapping Simulation {} of {} 🟠\n'.format(i,len(d_climate_data_meta)))

        scenario = d_climate_data_meta[i]['scenario']
        model = d_climate_data_meta[i]['model']
        extreme = d_climate_data_meta[i]['extreme']

        da_AFA = load_climate_data_array(climatedata_dir,
                    scenario,
                    model,
                    extreme,
                    year_start,
                    year_end,
                    bbox,
                    smoothing_window
                    )
        
        # align for masking if there are minor differences
        da_AFA = da_AFA.interp_like(countries_mask, method='linear')


        if convert_to_binary:
            # convert the data array to binary so the result is the fraction of land exposed 
            da_AFA = (da_AFA > convert_to_binary_threshold).astype(int)


        # loop over warming scenarios : RCP (model years), example trajs, stylized trajectories (strj)
        for suffix in var_suffixes+['strj']:

            print(f'Computing land fraction exposed (LFE) for {suffix} \n')

            # loop over regions - TODO VECTORIZE THIS
            for ind_region,region in enumerate(region_names): # could switch loops of region and suffix so i dont need to reselect the 'src' every time - not sure how much faster it would be? 

                print(f'Computing LFE in {region}                 ', end='\r')

                countries = get_countries_of_region(region, df_countries)

                ind_country = countries_regions.map_keys(countries)

                if suffix == 'RCP': 
                    # calc area-weighted fieldmean for each model year
                    land_frac_perregion = calc_weighted_fldmean(da_AFA,countries_mask,ind_country,weights, areaweighted)

                    ds_lfe_perregion_perrun[f'landfrac_peryear_perregion_{suffix}'].loc[{
                            'run' : i, 
                            'region' : ind_region
                        }] = land_frac_perregion.values
                
                elif suffix in var_suffixes: 
                    # remap based on indexes previously computed 
                    
                    if d_climate_data_meta[i][f'GMT_{suffix}_valid']: # if valid

                        ind_RCP = d_climate_data_meta[i][f'ind_RCP2GMT_{suffix}']

                        # extract the relevant subset for this run and region
                        src = ds_lfe_perregion_perrun['landfrac_peryear_perregion_RCP'].sel(run=i, region=ind_region)

                        # assign remapped 
                        ds_lfe_perregion_perrun[f'landfrac_peryear_perregion_{suffix}'].loc[dict(run=i, region=ind_region)] = src.isel(time_ind=ind_RCP).values

                elif suffix == "strj":
                    # repeat for the regular interval GMT stylized trajectories

                    for step, GMT in enumerate(GMT_labels):

                        if d_climate_data_meta[i]['GMT_strj_valid'][step]:

                            ind_RCP = d_climate_data_meta[i]['ind_RCP2GMT_strj'][:,step]

                            # extract the relevant subset for this run and region
                            src = ds_lfe_perregion_perrun['landfrac_peryear_perregion_RCP'].sel(run=i, region=ind_region)

                            ds_lfe_perregion_perrun[f'landfrac_peryear_perregion_{suffix}'].loc[dict(run=i, region=ind_region, GMT=GMT)] = src.isel(time_ind=ind_RCP).values



            # Calculate for all countries - vectorized without loop 

            print(f'Computing LFE in all countries                              ', end='\r')
                
            if suffix == 'RCP':

                # calculate area-weighted mean per country per model year
                area_weights = get_areaweights(da_AFA)

                # vectorized for all countries at once 
                landfrac_percountry = ( (da_AFA * area_weights).groupby(countries_mask).sum(dim='stacked_lat_lon')
                                        / area_weights.groupby(countries_mask).sum(dim='stacked_lat_lon')
                                    )

                # map country name from country code 
                name_map = xr.DataArray(
                countries_regions.names,
                dims="mask",
                coords={"mask": countries_regions.numbers},
                name="country",
                )

                # change cordinate names and order for easy broadcasting to target dataset 
                landfrac_percountry = landfrac_percountry.assign_coords(
                    country=name_map
                    ).swap_dims({"mask": "country"}
                    ).reindex(country=ds_lfe_percountry_perrun.country
                    ).rename({'time': 'time_ind'}
                    ).assign_coords(time_ind=ds_lfe_percountry_perrun.time_ind)

                ds_lfe_percountry_perrun[f'landfrac_peryear_percountry_{suffix}'].loc[{
                        'run' : i, 
                    }] = landfrac_percountry


            elif suffix in var_suffixes: 
                # remap based on indexes previously computed 
                if d_climate_data_meta[i][f'GMT_{suffix}_valid']: # if valid

                    ind_RCP = d_climate_data_meta[i][f'ind_RCP2GMT_{suffix}']

                    # extract the relevant subset for this run and region
                    src = ds_lfe_percountry_perrun['landfrac_peryear_percountry_RCP'].sel(run=i)

                    # assign remapped 
                    ds_lfe_percountry_perrun[f'landfrac_peryear_percountry_{suffix}'].loc[dict(run=i)] = src.isel(time_ind=ind_RCP).values

            elif suffix == "strj":
                # repeat for the regular interval GMT stylized trajectories

                for step, GMT in enumerate(GMT_labels):

                    if d_climate_data_meta[i]['GMT_strj_valid'][step]:

                        ind_RCP = d_climate_data_meta[i]['ind_RCP2GMT_strj'][:,step]

                        # extract the relevant subset for this run and region
                        src = ds_lfe_percountry_perrun['landfrac_peryear_percountry_RCP'].sel(run=i)

                        ds_lfe_percountry_perrun[f'landfrac_peryear_percountry_{suffix}'].loc[dict(run=i, GMT=GMT)] = src.isel(time_ind=ind_RCP).values


    return ds_lfe_perregion_perrun, ds_lfe_percountry_perrun, region_names


#%%

def calc_life_exposure_rimeX(
    df_exposure,
    df_life_expectancy,
    region_or_country_name
):
    # initialise birth years 
    exposure_birthyears_percountry = np.empty(len(df_life_expectancy))

    for i, birth_year in enumerate(df_life_expectancy.index):
        life_expectancy = df_life_expectancy.loc[birth_year,region_or_country_name] 

        # define death year based on life expectancy
        death_year = birth_year + np.floor(life_expectancy)

        # integrate exposure over full years lived
        exposure_birthyears_percountry[i] = df_exposure.loc[birth_year:death_year,col].sum()

        # add exposure during last (partial) year
        exposure_birthyears_percountry[i] = exposure_birthyears_percountry[i] + \
            df_exposure.loc[death_year+1,col].sum() * \
                (life_expectancy - np.floor(life_expectancy))

    # a series for each column to somehow group into a dataframe
    exposure_birthyears_percountry = pd.Series(
        exposure_birthyears_percountry,
        index=df_life_expectancy.index,
        name=col,
    )

    return exposure_birthyears_percountry




#%%


@timeit
def calc_lifetime_exposure(
    d_climate_data_meta, 
    df_countries, 
    countries_regions, 
    countries_mask, 
    climatedata_dir, # structure should be : climatedata_dir/scenario/model/'*{model}*{extreme}.nc
    da_population, 
    df_life_expectancy_5,
    da_cohort_size,
    GMT_labels = None , #df_GMT_strj.columns
    GMT_extra_trajectories_names=None,
    start_birthyear=1950,
    end_birthyear=2025,
    year_start=1950,
    year_end=2119,
    bbox=None,
    smoothing_window=None,
):

    """
        Calc lifetime exposure to your input hazard dataset per country and per region. 
        Results are given per original RCP model years (optionally with smoothing), as well
        as after GMT-remapping to match stylized trajectories, with the remapping recipe defined 
        in d_climate_data_meta (output of load_climate_data)

        Note: This conserved the units of your input dataset i.e. if your units are days
        the output will be in number of days exposed

        Inputs:
            GMT_labels (list):                      e.g. df_GMT_strj.columns

            smoothing_window (int)                  window to smooth impact data, default:None (recommended: 21)

            bbox (opt)                              latmin, latmax, lonmin, lonmax

        Returns:
            ds_le_perregion_perrun (ds):        average lifetime exposure per region, model year or remapped year, 
                                                scenario and birth year  
                                                
            
            ds_le_percountry_perrun (ds):       average lifetime exposure per country, model year or remapped year, 
                                                scenario and birth year  

            region_names (list):                region names to understand ds_le_perregion_perrun

    """
    def calc_life_exposure(
        df_exposure,
        df_life_expectancy,
        col,
    ):
        # initialise birth years 
        exposure_birthyears_percountry = np.empty(len(df_life_expectancy))

        for i, birth_year in enumerate(df_life_expectancy.index):
            life_expectancy = df_life_expectancy.loc[birth_year,col] 

            # define death year based on life expectancy
            death_year = birth_year + np.floor(life_expectancy)

            # integrate exposure over full years lived
            exposure_birthyears_percountry[i] = df_exposure.loc[birth_year:death_year,col].sum()

            # add exposure during last (partial) year
            exposure_birthyears_percountry[i] = exposure_birthyears_percountry[i] + \
                df_exposure.loc[death_year+1,col].sum() * \
                    (life_expectancy - np.floor(life_expectancy))

        # a series for each column to somehow group into a dataframe
        exposure_birthyears_percountry = pd.Series(
            exposure_birthyears_percountry,
            index=df_life_expectancy.index,
            name=col,
        )

        return exposure_birthyears_percountry


    #---------------------------------------------------------------------#
    # 1) Build Dataset for regions result
    #---------------------------------------------------------------------#

    # get regions and income groups and 'World' (=all countries)

    region_names = np.concatenate([df_countries['region'].dropna().unique(),
                            df_countries['incomegroup'].dropna().unique(),
                            ['World'] # if using a bbox this is all countries in the bbox - rename to all_countries? 
    ])

    # Shared shape for all variables

    nregions = len(region_names)
    birth_years = np.arange(start_birthyear, end_birthyear+1)
    year_range = np.arange(year_start, year_end+1)

    shape = (
        len(d_climate_data_meta), 
        nregions, 
        len(birth_years))

    shape_strj = (
        len(d_climate_data_meta), 
        nregions, 
        len(birth_years), 
        len(GMT_labels))

    # Build the data_vars dictionary in a loop
    
    var_suffixes = ['RCP']+GMT_extra_trajectories_names 

    data_vars = {}
    for suffix in var_suffixes:
        var_name = f'le_perregion_perrun_{suffix}'
        data_vars[var_name] = (
            ['run', 'region', 'birth_year'],
            np.full(shape, np.nan)
        )

    data_vars['le_perregion_perrun_strj'] = (
            ['run', 'region', 'birth_year', 'GMT'],
            np.full(shape_strj, np.nan)

    )

    # Build the dataset
    ds_le_perregion_perrun = xr.Dataset(
        data_vars=data_vars,
        coords={
            'run': ('run', np.arange(1, len(d_climate_data_meta) + 1)),
            'region': ('region', np.arange(0, nregions)),
            'birth_year': ('birth_year', birth_years),
            'GMT': ('GMT', GMT_labels)
        }
    )


    #---------------------------------------------------------------------#
    # 2) Build Dataset for country result
    #---------------------------------------------------------------------#

    # Shared shape for all variables
    shape = (
        len(d_climate_data_meta),
        len(df_countries['name'].values),
        len(birth_years)
    )

    shape_strj = (
        len(d_climate_data_meta), 
        len(df_countries['name'].values), 
        len(birth_years), 
        len(GMT_labels))


    # Build the data_vars dictionary in a loop
    data_vars = {}
    for suffix in var_suffixes:
        var_name = f'le_percountry_perrun_{suffix}'
        data_vars[var_name] = (
            ['run', 'country', 'birth_year'],
            np.full(shape, np.nan)
        )

    data_vars['le_percountry_perrun_strj'] = (
            ['run', 'country', 'birth_year', 'GMT'],
            np.full(shape_strj, np.nan)
    )

    # Build the dataset  
    ds_le_percountry_perrun = xr.Dataset(
        data_vars=data_vars,
        coords={
            'run': ('run', np.arange(1, len(d_climate_data_meta) + 1)),
            'country': ('country', df_countries['name'].values),
            'birth_year': ('birth_year', birth_years),
            'GMT': ('GMT', GMT_labels)
        }
    )

    #---------------------------------------------------------------------#
    # 3) Loop over simulations
    #---------------------------------------------------------------------#

    for i in list(d_climate_data_meta.keys()): 

        print('                         🟠 Remapping Simulation {} of {} 🟠\n'.format(i,len(d_climate_data_meta)))

        scenario = d_climate_data_meta[i]['scenario']
        model = d_climate_data_meta[i]['model']
        extreme = d_climate_data_meta[i]['extreme']

        da_AFA = load_climate_data_array(
            climatedata_dir,
            scenario,
            model,
            extreme,
            year_start,
            year_end,
            bbox,
            smoothing_window
            )
        
        # align for masking if there are minor differences in grid
        da_AFA = da_AFA.interp_like(countries_mask, method='linear')
        
        #---------------------------------------------------------------------#
        # Computation of the weighted by population field mean of AFA for     # 
        # each ISIMIP simulations for each country                            #
        #---------------------------------------------------------------------#


        print('⏳ Computing the Population-Weighted Spatial Average of the Exposure for all countries\n')

        # population-weighted average - instead of using calc_weighted_fldmean() 
        da_exposure_percountry = (da_AFA * da_population).groupby(countries_mask).sum(dim=['stacked_lat_lon']) / da_population.groupby(countries_mask).sum(dim=['stacked_lat_lon']) 

        # country name from country code 
        name_map = xr.DataArray(
        countries_regions.names,
        dims="mask",
        coords={"mask": countries_regions.numbers},
        name="country",
        )

        # convert to dataframe with country names as columns
        df = da_exposure_percountry.assign_coords(country=name_map).swap_dims({"mask": "country"}).to_pandas()

        # reorder to have same order as df_countries (not strictly necessary because broadcasted based on country name later)
        df_exposure_percountry = df.reindex(
                                    df_countries['name'].values,
                                    axis=1
                                )


        print('⏳ Computing the Population-Weighted Spatial Average of the Exposure for all regions\n')

        # TODO: VECTORIZE THIS apply same logic to regions !! 

        # initialise dict
        d_exposure_perregion = {}
        d_life_expectancy_perregion = {}

        # get spatial average per region 
        for region in region_names: 

            print(f'Computing lifetime exposure (LE) in {region}                 ', end='\r')

            countries = get_countries_of_region(region, df_countries)

            ind_country = countries_regions.map_keys(countries)

            # historical + RCP simulations
            d_exposure_perregion[region] = calc_weighted_fldmean( 
                da_AFA,
                countries_mask, 
                ind_country,
                weights=da_population, 
                areaweighted=False
            )

            # weighted avg life expectancy, weighted by n of people aged 0 per country 
            values = df_life_expectancy_5[countries]
            weights = da_cohort_size.sel(country=countries, ages=0).to_pandas().T.loc[values.index]
            d_life_expectancy_perregion[region] = (values * weights).sum(axis=1) / weights.sum(axis=1)


        print(f'Converting to dataframe                              ', end='\r')
        

        # Convert dict to dataframe for vectorizing and integrate exposures   
        # avg population-weighted exposure per country per year
        frame = {k:v.values for k,v in d_exposure_perregion.items()}
        df_exposure_perregion = pd.DataFrame(frame,index=year_range)  

        frame = {k:v.values for k,v in d_life_expectancy_perregion.items()}
        df_life_expectancy_perregion = pd.DataFrame(frame,index=birth_years)  


        # convert to da for saving output 
        da_life_expectancy_perregion = xr.DataArray(
            df_life_expectancy_perregion.values.astype(float),
            dims=['birth_year', 'region'],
            coords={'birth_year': df_life_expectancy_perregion.index, 'region': np.arange(len(region_names))}
        )

        print('🟢 Population-Weighted Spatial Average of the Exposure for all countries and regions Computed')

        # clear memory
        del d_exposure_perregion, d_life_expectancy_perregion 



        # loop over warming scenarios : RCP (model years), example trajs, stylized trajectories (strj)
        for suffix in var_suffixes+['strj']:

            print(f'Computing lifetime exposure (LE) for {suffix} \n')

            if suffix =='RCP':

                # calc lifetime exposure per country without remapping
                d_le_percountry_perrun = df_exposure_percountry.apply(
                    lambda col: calc_life_exposure(
                        df_exposure_percountry,
                        df_life_expectancy_5,
                        col.name,
                        ),
                    axis=0,
                    )

                # convert dataframe to data array of lifetime exposure (le) per country and birth year
                da_le = xr.DataArray(
                    d_le_percountry_perrun,
                    dims=('birth_year', 'country'),
                    coords={
                        'birth_year': birth_years,
                        'country': d_le_percountry_perrun.columns,
                    }
                )

                ds_le_percountry_perrun[f'le_percountry_perrun_{suffix}'].loc[
                    dict(run=i)
                ] = da_le




                # calc lifetime exposure per region without remapping
                d_le_perregion_perrun = df_exposure_perregion.apply(
                    lambda col: calc_life_exposure(
                        df_exposure_perregion,
                        df_life_expectancy_perregion,
                        col.name,
                        ),
                    axis=0,
                    )

                # convert dataframe to data array of lifetime exposure (le) per country and birth year
                ds_le_perregion_perrun[f'le_perregion_perrun_{suffix}'].loc[{
                    'run':i,
                }] = d_le_perregion_perrun.values.transpose()  


        


            elif suffix in var_suffixes:

                if d_climate_data_meta[i][f'GMT_{suffix}_valid']: # if valid

                    ind_RCP = d_climate_data_meta[i][f'ind_RCP2GMT_{suffix}']

                    # remap per pathway and calc lifetime exposure per country 
                    d_le_percountry_perrun = df_exposure_percountry.apply(
                        lambda col: calc_life_exposure(
                            df_exposure_percountry.reindex(
                                df_exposure_percountry.index[ind_RCP]
                                ).set_index(df_exposure_percountry.index),
                            df_life_expectancy_5,
                            col.name,
                            ),
                        axis=0,
                        )

                    # convert dataframe to data array of lifetime exposure (le) per country and birth year
                    da_le = xr.DataArray(
                        d_le_percountry_perrun,
                        dims=('birth_year', 'country'),
                        coords={
                            'birth_year': birth_years,
                            'country': d_le_percountry_perrun.columns,
                        }
                    )

                    ds_le_percountry_perrun[f'le_percountry_perrun_{suffix}'].loc[
                        dict(run=i)
                    ] = da_le

                    


                    # remap per pathway and calc lifetime exposure per region 
                    d_le_perregion_perrun = df_exposure_perregion.apply(
                        lambda col: calc_life_exposure(
                            df_exposure_perregion.reindex(
                                df_exposure_perregion.index[ind_RCP]
                                ).set_index(df_exposure_perregion.index),
                            df_life_expectancy_perregion,
                            col.name,
                            ),
                        axis=0,
                        )

                    # convert dataframe to data array of lifetime exposure (le) per country and birth year
                    ds_le_perregion_perrun[f'le_perregion_perrun_{suffix}'].loc[{
                        'run':i,
                    }] = d_le_perregion_perrun.values.transpose() 
                



            elif suffix == 'strj':

                for step, GMT in enumerate(GMT_labels):

                    if d_climate_data_meta[i]['GMT_strj_valid'][step]:
                        
                        # check validity
                        ind_RCP = d_climate_data_meta[i]['ind_RCP2GMT_strj'][:,step]

                        # remap per pathway and calc lifetime exposure per country 
                        d_le_percountry_perrun = df_exposure_percountry.apply(
                            lambda col: calc_life_exposure(
                                df_exposure_percountry.reindex(
                                    df_exposure_percountry.index[ind_RCP]
                                    ).set_index(df_exposure_percountry.index),
                                df_life_expectancy_5,
                                col.name,
                                ),
                            axis=0,
                            )

                        # convert dataframe to data array of lifetime exposure (le) per country and birth year
                        da_le = xr.DataArray(
                            d_le_percountry_perrun,
                            dims=('birth_year', 'country'),
                            coords={
                                'birth_year': birth_years,
                                'country': d_le_percountry_perrun.columns,
                            }
                        )

                        ds_le_percountry_perrun[f'le_percountry_perrun_{suffix}'].loc[
                            dict(run=i, GMT=GMT)
                        ] = da_le


                        # remap per pathway and calc lifetime exposure per region 
                        d_le_perregion_perrun = df_exposure_perregion.apply(
                            lambda col: calc_life_exposure(
                                df_exposure_perregion.reindex(
                                    df_exposure_perregion.index[ind_RCP]
                                    ).set_index(df_exposure_perregion.index),
                                df_life_expectancy_perregion,
                                col.name,
                                ),
                            axis=0,
                            )

                        # convert dataframe to data array of lifetime exposure (le) per country and birth year
                        ds_le_perregion_perrun[f'le_perregion_perrun_{suffix}'].loc[{
                            'run':i,
                            'GMT':GMT,
                        }] = d_le_perregion_perrun.values.transpose() 


    return ds_le_percountry_perrun, ds_le_perregion_perrun, region_names, da_life_expectancy_perregion





#%%



@timeit
def calc_lifetime_exposure_subnational(
    d_climate_data_meta, 
    df_countries, 
    subnational_regions, 
    subnational_mask, 
    gdf_subnational, 
    climatedata_dir,
    da_population, 
    df_life_expectancy_5,
    da_cohort_size,
    GMT_labels = None , #df_GMT_strj.columns
    GMT_extra_trajectories_names=None,
    start_birthyear=1950,
    end_birthyear=2025,
    year_start=1950,
    year_end=2119,
    bbox=None,
    smoothing_window=None,
):

    def calc_life_exposure(
        df_exposure,
        df_life_expectancy,
        col,
    ):
        # initialise birth years 
        exposure_birthyears_percountry = np.empty(len(df_life_expectancy))

        for i, birth_year in enumerate(df_life_expectancy.index):
            life_expectancy = df_life_expectancy.loc[birth_year] 

            # define death year based on life expectancy
            death_year = birth_year + np.floor(life_expectancy)

            # integrate exposure over full years lived
            exposure_birthyears_percountry[i] = df_exposure.loc[birth_year:death_year,col].sum()

            # add exposure during last (partial) year
            exposure_birthyears_percountry[i] = exposure_birthyears_percountry[i] + \
                df_exposure.loc[death_year+1,col].sum() * \
                    (life_expectancy - np.floor(life_expectancy))

        # a series for each column to somehow group into a dataframe
        exposure_birthyears_percountry = pd.Series(
            exposure_birthyears_percountry,
            index=df_life_expectancy.index,
            name=col,
        )

        return exposure_birthyears_percountry
    

    def get_country(
        region, 
        gdf_subnational,
        df_countries
        ):

        country_code = gdf_subnational.loc[gdf_subnational['id']==region]['country'].values[0]

        # The European Commission generally uses ISO 3166-1 alpha-2 codes with two exceptions
        # EL (not GR) is used to represent Greece, and UK (not GB) is used to represent the United Kingdom
        if country_code == 'EL':
            country_code = 'GR'
        elif country_code == 'UK':
            country_code = 'GB'

        country = df_countries.loc[df_countries['ISO2']==country_code]['name'].values[0]

        return country


    # Build Dataset for regions result

    # get regions names
    region_names = [subnational_regions[i].abbrev for i in range(len(subnational_regions))]

    # Shared shape for all variables
    nregions = len(region_names)

    birth_years = np.arange(start_birthyear, end_birthyear+1)

    year_range = np.arange(year_start, year_end+1)

    shape = (
        len(d_climate_data_meta), 
        nregions, 
        len(birth_years))

    shape_strj = (
        len(d_climate_data_meta), 
        nregions, 
        len(birth_years), 
        len(GMT_labels))

    # Build the data_vars dictionary in a loop
    
    var_suffixes = ['RCP']+GMT_extra_trajectories_names 

    data_vars = {}
    for suffix in var_suffixes:
        var_name = f'le_perregion_perrun_{suffix}'
        data_vars[var_name] = (
            ['run', 'region',  'birth_year'], #'country',
            np.full(shape, np.nan)
        )

    data_vars['le_perregion_perrun_strj'] = (
            ['run', 'region', 'birth_year', 'GMT'], #'country', 
            np.full(shape_strj, np.nan)

    )

    # Build the dataset
    ds_le_perregion_perrun = xr.Dataset(
        data_vars=data_vars,
        coords={
            'run': ('run', np.arange(1, len(d_climate_data_meta) + 1)),
            'region': ('region', np.arange(0, nregions)),
            'birth_year': ('birth_year', birth_years),
            'GMT': ('GMT', GMT_labels)
        }
    )


    # loop over simulations
    
    for i in list(d_climate_data_meta.keys()): 

        print('                         🟠 Remapping Simulation {} of {} 🟠\n'.format(i,len(d_climate_data_meta)))

        scenario = d_climate_data_meta[i]['scenario']
        model = d_climate_data_meta[i]['model']
        extreme = d_climate_data_meta[i]['extreme']

        da_AFA = load_climate_data_array(
            climatedata_dir,
            scenario,
            model,
            extreme,
            year_start,
            year_end,
            bbox,
            smoothing_window
            )
        
        # align for masking if there are minor differences
        da_AFA = da_AFA.interp_like(subnational_mask, method='linear')
        
        #---------------------------------------------------------------------#
        # Computation of the weighted by population field mean of AFA for     # 
        # each ISIMIP simulations for each subnational region                 #
        #---------------------------------------------------------------------#


        print('⏳ Computing the Population-Weighted Spatial Average of the Exposure for all regions\n')

        # calculate population-weighted average
        da_exposure_perregion = (da_AFA * da_population).groupby(subnational_mask).sum(dim=['stacked_lat_lon']) / da_population.groupby(subnational_mask).sum(dim=['stacked_lat_lon']) 

        # region abbreviation from numeric code
        name_map = xr.DataArray(
        subnational_regions.abbrevs,
        dims="mask",
        coords={"mask": subnational_regions.numbers},
        name="name",
        )

        # convert to dataframe
        df = da_exposure_perregion.assign_coords(name=name_map).swap_dims({"mask": "name"}).to_pandas()

        # add missing regions (too small) columns and fill with nan
        missing_regions = set(gdf_subnational['id']) - set(df.columns)
        df[list(missing_regions)] = np.nan

        # reorder columns to match original gdf order (important for broadcasting later)
        df_exposure_perregion = df.reindex(gdf_subnational.id, axis=1)


        print('🟢 Population-Weighted Spatial Average of the Exposure for all regions computed')


        # loop over warming scenarios : RCP (model years), example trajs, stylized trajectories (strj)

        for suffix in var_suffixes+['strj']:

            print(f'Computing lifetime exposure (LE) for {suffix} \n')

            if suffix =='RCP':

                # calc lifetime exposure per region without remapping (original RCP, optionally smoothed)
                d_le_perregion_perrun = df_exposure_perregion.apply(
                    lambda col: calc_life_exposure(
                        df_exposure_perregion,
                        df_life_expectancy_5[get_country(col.name, gdf_subnational, df_countries)], # use the same life expectancy for each region of same country
                        col.name,
                        ),
                    axis=0,
                    )

                # convert dataframe to data array of lifetime exposure (le) per country and birth year
                ds_le_perregion_perrun[f'le_perregion_perrun_{suffix}'].loc[{
                    'run':i,
                }] = d_le_perregion_perrun.values.transpose()  
        
            elif suffix in var_suffixes:

                if d_climate_data_meta[i][f'GMT_{suffix}_valid']: # if valid

                    ind_RCP = d_climate_data_meta[i][f'ind_RCP2GMT_{suffix}']

                    # remap per pathway and calc lifetime exposure per region 
                    d_le_perregion_perrun = df_exposure_perregion.apply(
                        lambda col: calc_life_exposure(
                            df_exposure_perregion.reindex(
                                df_exposure_perregion.index[ind_RCP]
                                ).set_index(df_exposure_perregion.index),
                            df_life_expectancy_5[get_country(col.name, gdf_subnational, df_countries)],
                            col.name,
                            ),
                        axis=0,
                        )

                    # convert dataframe to data array of lifetime exposure (le) per country and birth year
                    ds_le_perregion_perrun[f'le_perregion_perrun_{suffix}'].loc[{
                        'run':i,
                    }] = d_le_perregion_perrun.values.transpose() 
                
            elif suffix == 'strj':

                for step, GMT in enumerate(GMT_labels):

                    if d_climate_data_meta[i]['GMT_strj_valid'][step]:
                        
                        # check validity
                        ind_RCP = d_climate_data_meta[i]['ind_RCP2GMT_strj'][:,step]


                        # remap per pathway and calc lifetime exposure per region 
                        d_le_perregion_perrun = df_exposure_perregion.apply(
                            lambda col: calc_life_exposure(
                                df_exposure_perregion.reindex(
                                    df_exposure_perregion.index[ind_RCP]
                                    ).set_index(df_exposure_perregion.index),
                                df_life_expectancy_5[get_country(col.name, gdf_subnational, df_countries)],
                                col.name,
                                ),
                            axis=0,
                            )

                        # convert dataframe to data array of lifetime exposure (le) per country and birth year
                        ds_le_perregion_perrun[f'le_perregion_perrun_{suffix}'].loc[{
                            'run':i,
                            'GMT':GMT,
                        }] = d_le_perregion_perrun.values.transpose() 
        
        # rename so its not just ints
        ds_le_perregion_perrun['region'] = region_names


    #TODO: could also add a 'country' dimension for more easy accessing 
    

    return ds_le_perregion_perrun, region_names






























#%%











def calc_landfraction_exposed_mmm(
    ds_lfe_perrun,
    GMT_extra_trajectories_names
    ):
    """
    Aggregate across different runs (i.e. across remapped simulations), for each target trajectory 
    to get the spread across (remapped/synthetic) runs for each target simulation 

    Input

    Returns
        mmm:                    multi model mean (mean across synthetic/remapped simulations/runs)
        std:                    std across remapped runs
        lqntl, median, uqntl:   0.25, median (0.5), 0.75 quantile 
        _sm:                    10 year rolling mean
        sample_size:            number of valid remapped simulations for that stylized trajectory

    
    Notes:
    - TODO apply RIME approach to remapping
    - could create a 'quantile' dim instead of separate lqntl,uqntl,median! might be cleaner
    - not sure why you want to smooth the median and std in fxn? can always do this after.. 

    """

    # remove from the output a warning that appears often  
    import warnings
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")

    # -------------------------------------------------- #
    #             Computation of MMM and Stats           #
    # -------------------------------------------------- #

    ds_perrun = ds_lfe_perrun.copy()

    if 'region' in ds_perrun.dims:
        unit='region'
    elif 'country' in ds_perrun.dims:
        unit='country'
    else:
        raise ValueError("spatial unit not defined")
        
    print(f"\nComputing the MMM and stats for each {unit}")

    # loop over target trajectories
    var_suffixes = GMT_extra_trajectories_names+['strj'] 

    for suffix in var_suffixes:

        var_name = f'landfrac_peryear_per{unit}_{suffix}'

        # aggregate 
        mmm = ds_perrun[var_name].mean(dim='run', skipna=True)
        mmm_sm = mmm.rolling(time_ind=10, center=True, min_periods=1).mean()
        std = ds_perrun[var_name].std(dim='run', skipna=True)
        std_sm = std.rolling(time_ind=10, center=True, min_periods=1).mean()
        lqntl = ds_perrun[var_name].reduce(np.nanpercentile, dim='run', q=25, method='linear')
        uqntl = ds_perrun[var_name].reduce(np.nanpercentile, dim='run', q=75, method='linear')
        median = ds_perrun[var_name].median(dim='run')
        median_sm = median.rolling(time_ind=10, center=True, min_periods=1).mean()

        other_dims = set(ds_perrun.dims) - {'run', 'GMT'}
        isel_dict = {dim: 0 for dim in other_dims}
        sample_size = ds_perrun[f'landfrac_peryear_per{unit}_{suffix}'].isel(**isel_dict).notnull().sum(dim='run')

        ds_perrun[f'mmm_{suffix}'] = mmm
        ds_perrun[f'mmm_{suffix}_sm'] = mmm_sm
        ds_perrun[f'std_{suffix}'] = std
        ds_perrun[f'std_{suffix}_sm'] = std_sm
        ds_perrun[f'lqntl_{suffix}'] = lqntl
        ds_perrun[f'uqntl_{suffix}'] = uqntl
        ds_perrun[f'median_{suffix}'] = median
        ds_perrun[f'median_{suffix}_sm'] = median_sm
        ds_perrun[f'sample_size_{suffix}'] = sample_size

        # TODO: replace med,lq,uq with a single var with a quantile dimension?? and dont drop the quantile dimension

    return ds_perrun.reset_coords()





def calc_lifetime_exposure_mmm(
    ds_le_perrun,
    GMT_extra_trajectories_names,
    year_ref=1950
    ):
    """
    NOTE: I changed how the EMF med,uq,lq,mean is calculated! 
    first calc the EMF for each model then aggregate - which i think is more correct
    """

        # remove from the output a warning that appears often  
    import warnings
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")

    # -------------------------------------------------- #
    #             Computation of MMM and Stats           #
    # -------------------------------------------------- #

    ds_perrun = ds_le_perrun.copy()

    if 'region' in ds_perrun.dims:
        unit='region'
    elif 'country' in ds_perrun.dims:
        unit='country'
    else:
        raise ValueError("spatial unit not defined")
    
    print(f"\nComputing the MMM and stats for each {unit}")

    # loop over target trajectories
    var_suffixes = GMT_extra_trajectories_names+['strj'] 

    for suffix in var_suffixes:

        var_name = f'le_per{unit}_perrun_{suffix}'

        mmm = ds_perrun[var_name].mean(dim='run', skipna=True)
        std = ds_perrun[var_name].std(dim='run', skipna=True)
        lqntl = ds_perrun[var_name].reduce(np.nanpercentile, dim='run', q=25, method='linear')
        uqntl = ds_perrun[var_name].reduce(np.nanpercentile, dim='run', q=75, method='linear')
        median = ds_perrun[var_name].median(dim='run')

        # number of valid remapped trajectories 
        other_dims = set(ds_perrun.dims) - {'run', 'GMT'}
        isel_dict = {dim: 0 for dim in other_dims}
        sample_size = ds_perrun[var_name].isel(**isel_dict).notnull().sum(dim='run')

        # EMF compared to reference birthyear
        EMF = ds_perrun[var_name] / ds_perrun[var_name].sel(birth_year=year_ref)

        mmm_EMF = EMF.mean(dim='run', skipna=True)
        std_EMF = EMF.std(dim='run', skipna=True)
        # method coherent with standard pandas/xarray behaviour
        lqntl_EMF = EMF.reduce(np.nanpercentile, dim='run', q=25, method='linear')
        uqntl_EMF = EMF.reduce(np.nanpercentile, dim='run', q=75, method='linear')
        median_EMF = EMF.median(dim='run')

        ds_perrun[f'mmm_{suffix}'] = mmm
        ds_perrun[f'std_{suffix}'] = std
        ds_perrun[f'lqntl_{suffix}'] = lqntl
        ds_perrun[f'uqntl_{suffix}'] = uqntl
        ds_perrun[f'median_{suffix}'] = median
        ds_perrun[f'sample_size_{suffix}'] = sample_size
        ds_perrun[f'mmm_EMF_{suffix}'] = mmm_EMF
        ds_perrun[f'std_EMF_{suffix}'] = std_EMF
        ds_perrun[f'lqntl_EMF_{suffix}'] = lqntl_EMF
        ds_perrun[f'uqntl_EMF_{suffix}'] = uqntl_EMF
        ds_perrun[f'median_EMF_{suffix}'] = median_EMF

    return ds_perrun.reset_coords()





#%%


def calc_lifetime_exposure_map(

):

# da_AFA load

# remap based on d_climate_data_meta

# aggregate for certain birthyears / match based on shapefile - maybe select just one country at a time? Or somehow do for all Europe/all countries???

# aggregate across models

# plot per pixel lifetime exposure, if there is a person there, as abs n years or as % lifetime

    pass


