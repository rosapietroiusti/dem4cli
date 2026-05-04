"""
Demographics4Climate : Population and demographics for climate science analysis
----------------------------------------------

Update 2025 with new data 

To do
> understand how best to deal with having v1 and v2... e.g. 
    - have only one pop_demographics file but make flags, outside of functions 
    or inside functions saying which functions get loaded.... 
    - remove v1? 
""" 
#%%

import glob, os, re, sys
import warnings
from math import ceil
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import regionmask
from shapely.geometry import box

from ._utils import *
from ._settings import *

#%%
# ---------------------------------
# 1. Metadata 
# ---------------------------------


@timeit
def load_country_metadata(
    cfg,
    filepath_world_bank=None,
    filepath_lookuptable=None,
    data_source_cohorts=None,
    worldbank_filter=True,
):
    """
    load country list metadata from worldbank (218 countries) - see what year this classification is from

    Input
        filepath_world_bank (str) 
        filepath_lookuptable (str)
        data_source_cohorts (str): 'UNWPP2024' or 'WCDE'
    
    Returns
        df_metadata (df):       country name, ISO3 code, country code, region, income group 
                                filtered based on life expectancy and cohort size data availability 
                                and if world_bank filter is True also based on WB categorization
    
    """
    if cfg is None:
        raise ValueError(
            "cfg must be provided. Create one with cfg = init_settings()."
        )

    if filepath_world_bank is None:
        filepath_world_bank = cfg.filepath_world_bank_meta

    if filepath_lookuptable is None:
        filepath_lookuptable = cfg.filepath_lookuptable

    if data_source_cohorts is None:
        data_source_cohorts = cfg.cohort_sizes_source

    # 1) World Bank data

    # open world bank categorization: 218 countries total
    df_metadata = pd.read_excel(filepath_world_bank, sheet_name=0)
    # get rid of regions
    df_metadata = df_metadata[~df_metadata['Region'].isna()]
    # rename
    keep_cols =['Economy', 'Code', 'Region', 'Income group']
    d_rename={
        'Economy':'country',
        'Code':'abbreviation', 
        'Region':'region',
        'Income group': 'incomegroup'}
    df_metadata = df_metadata[keep_cols].rename(columns=d_rename) 


    # 2) Lookup table : cohort size and life expectancy

    # open lookup table
    df = pd.read_csv(filepath_lookuptable)

    # overlap of cohort size (WPP/WCDE) and life expectancy (WPP) data
    if data_source_cohorts == 'WCDE':
        # 201 countries
        df_overlap = df[
                        (df[["SSP name", "WPP name"]].notna().all(axis=1)) &
                        (df["Data availability"] == "Full historical + SSP")
                    ].reset_index(drop=True)
    elif data_source_cohorts == 'UNWPP2024':
        # 236 countries
        df_overlap = df[
                        (df[["SSP name", "WPP name"]].notna().all(axis=1))
                    ].reset_index(drop=True)
    else:
        raise ValueError("data_source_cohorts must be WCDE or UNWPP2024")

    df_overlap = df_overlap[["SSP name", "WPP name", "ISO numeric","ISO alpha-2", "ISO alpha-3"]]


    if worldbank_filter:
        # only include countries that are also in WB categorization
        # and have life expectancy and cohort size data
        # 217 with UNWPP cohorts (lose Channel Islands, data available as Jersey/Guernsey)
        # 195 with WCDE
        df_metadata_filtered = df_metadata.merge(
            df_overlap, how='inner', left_on='abbreviation', right_on='ISO alpha-3'
            ).reset_index(drop=True
            ).rename(columns={ 
                'ISO numeric': 'country_code', 
                'ISO alpha-2': 'ISO2',
                'WPP name':'name' })

    else: 
        # include all countries that have all demographic data even if not in WB categorization
        # 236 with UNWPP
        # 201 with WCDE
        df_metadata_filtered = df_metadata.merge(
            df_overlap, how='right', left_on='abbreviation', right_on='ISO alpha-3'
            ).reset_index(drop=True
            ).drop(columns='abbreviation'
            ).rename(columns={
                'ISO alpha-3':'abbreviation', 
                'ISO alpha-2': 'ISO2',
                'ISO numeric': 'country_code', 
                'WPP name' : 'name' })

    # get only useful columns
    df_metadata_filtered = df_metadata_filtered[[
        'abbreviation',  'ISO2', 'region', 'incomegroup', 'country_code', 'SSP name', 'name'   
        ]]
        
    return df_metadata_filtered.set_index('name', drop=False)



# ---------------------------------
# 2. Cohort sizes
# ---------------------------------



@timeit
def load_cohort_sizes(
    cfg,
    dir_cohortsizes=None,
    data_source=None,
    ssp=2,
    by_sex=False,
):

    """
    load population size per age cohort from Wittgenstein Center Data Explorer.
    
    Version 1: WCDE v2 (source: http://dataexplorer.wittgensteincentre.org/wcde-v2/)

    Version 2: version 3.2 beta (for CMIP7)

    data description: Population Size (000's)
    De facto population in a country or region, classified by sex and by five-year age groups. 
    Available in all scenarios and at all geographical scales. 
    For each country data is sorted first by age cohort (0-4, 4-9...). 
    Then they give the population size of that cohort at a snapshot every 5 years (1950, 1955, 1960...).
    Here we assign the data to the central age cohort (i.e. 0-4 assigned to 2).
    
    Input
        dir_cohortsizes (str):      path to cohortsize files
        data_source (str):          'WCDE' or 'UNWPP2024' 
        ssp (int):                  1,2 or 3 for ssp1, ssp2, ssp3 (only if data_source == 'WCDE')
        by_sex (Bool):              TODO (data is available male/female) - in version 2 this is automatic
        

    Returns
        df_cohort_sizes (v1: df, v2: da):   v1: rows are countries, columns are a cohort's (e.g. age=2) 
                                            size each year, then the next cohort 
                                            (columns labelled e.g. 2_1950 age=2, year=1950) 
                                            v2: data array indexed by age 
        ages (arr) : central year of interval (2,7...102)
        years (arr) : years we have data for (1950, 1955...2100)

    Note: 
    - in v2 its a da not a df !!! TODO: change version 1 so it also gives a da?  
    """

    if cfg is None:
        raise ValueError(
            "cfg must be provided. Create one with cfg = init_settings()."
        )

    if dir_cohortsizes is None:
        dir_cohortsizes = cfg.dir_cohortsizes

    if data_source is None:
        data_source = cfg.cohort_sizes_source

    def convert_age_range(age):
        if age == '100+':
            return 100
        else:
            match = re.match(r'(\d+)-{1,2}(\d+)', age)
            if match:
                return int(match.group(1))
            else:
                return int(age)
    
    print(f'loading cohort sizes from {data_source}')

    if cfg.version == 1:

        if not data_source == 'WCDE':
            print(f'error method cohort size undefined in v1 for {data_source}')

        else:

            # open wcde cohort size file 
            filepath = dir_cohortsizes+f'/wicdf_ssp{ssp}.csv'
            df_raw = pd.read_csv(filepath, header=7) # population is in 000's

            if not by_sex:

                # select only relevant rows and cols
                df = df_raw[(df_raw['Sex'] == 'Both') & (df_raw['Age'] != 'All') & (df_raw['Area'] != 'World')][[
                    'Area', 'Year', 'Age', 'Population'
                    ]]
                
                # central year in age bracket e.g. 0-4 becomes 2, 5-9 becomes 7 
                df['Age'] = df['Age'].apply(convert_age_range) + 2 
                    
                # Initialize an empty DataFrame for the final result
                df_cohort_sizes = pd.DataFrame()
                # Get unique ages
                ages = df['Age'].unique()
                years = df['Year'].unique()
                
                # Loop through each age and pivot the data
                for age in ages:
                    subset = df[df['Age'] == age].pivot(index='Area', columns='Year', values='Population')
                    subset.columns = [f'{age}_{year}' for year in subset.columns] # name the columns e.g. 2_1950
                    if df_cohort_sizes.empty:
                        df_cohort_sizes = subset
                    else:
                        df_cohort_sizes = df_cohort_sizes.join(subset, how='outer')

            else:
                
                pass
                # to develop by sex 

    elif cfg.version == 2:

        if data_source == 'WCDE': # cohort sizes from SSP projections v3.2-beta

            filepath = dir_cohortsizes+'/ssp_basic_drivers_release_3.2.beta_full.xlsx'
            df = pd.read_excel(filepath, sheet_name=1)

            # Exclude rows where 'region' contains (R<number>), i.e. world regions
            df = df[~df['Region'].str.contains(r"\(R\d+\)", regex=True, na=False)]

            # Keep only "Population|Male|Age <age>" or "Population|Female|Age <age>"
            population_pattern = (r"^Population\|(Male|Female)\|Age\s(\d{1,2}(-\d{1,2})?|100\+)$")
            df = df[df['Variable'].str.contains(population_pattern, regex=True, na=False)]

            # Extract sex and age from Variable
            df['sex'] = df['Variable'].str.extract(r"^Population\|(Male|Female)", expand=False)
            df['age'] = df['Variable'].str.extract(r"Age\s([\d\+]+(?:-\d+)?)", expand=False)

            # Melt year columns to long format
            year_cols = [c for c in df.columns if re.match(r"^\d{4}$", str(c))]
            df_long = df.melt(
                id_vars=['Region', 'sex', 'age','Scenario'],
                value_vars=year_cols,
                var_name='time',
                value_name='population'
            )
            # clean 
            df_long['time'] = df_long['time'].astype(int)
            df_long['Scenario'] = df_long['Scenario'].replace({'Historical Reference': 'historical'})

            # central year in age bracket e.g. 0-4 becomes 2, 5-9 becomes 7 
            df_long['age'] = df_long['age'].apply(convert_age_range) + 2 

            # make into a data array for easy indexing 
            da = df_long.set_index(['Region', 'time', 'age', 'sex', 'Scenario']).to_xarray()['population']
            da = da.rename({'Region':'country', 'Scenario':'ssp'})

            # select SSP and merge historical + SSP
            da = da.sel(ssp=['historical', f'SSP{ssp}']).max(dim="ssp") * 1000 # data provided as millions, convert to thousands

            # get ages and years
            ages = da.age.values
            years = da.time.values

            if not by_sex:
                da = da.sum(dim='sex') 

            df_cohort_sizes = da.where(da.country != 'World', drop=True) # remove 'World'
    

        elif data_source == 'UNWPP2024': # cohort sizes from UNWPP2024 historical estimates and median projections

            filepath = dir_cohortsizes+'/WPP2024_POP_F01_1_POPULATION_SINGLE_AGE_BOTH_SEXES.xlsx'

            df_list = []

            for sheet in [0,1]: # sheet 0 has historical estimates, sheet 1 has projections
                df_unwpp_raw = pd.read_excel(filepath, 
                sheet_name=sheet,
                skiprows=16) # make this more flex 
        
                # load data and select only data you want
                df_unwpp = df_unwpp_raw[df_unwpp_raw['Type']=='Country/Area'].rename( # get rid of World/region/subregion, keep only countries 
                        columns={'Region, subregion, country or area *':'country', '100+':100, 'Year':'time', 'age':'ages'}) # make this more flex 
                cols = df_unwpp.columns
                idxs = [i for i, col in enumerate(cols) if col in ('country', 'time') or (isinstance(col, int) and 0 <= col < 101)] # cohort size each age each year
                df_unwpp = df_unwpp.iloc[:, idxs]
                df_unwpp['time'] = df_unwpp['time'].astype(int)

                df_list.append(df_unwpp)
            
            # concat historical and projections
            df_unwpp = pd.concat(df_list, axis=0)

            # convert to data array 
            df_indexed = df_unwpp.set_index(['country', 'time'])
            da = df_indexed.to_xarray()
            da = da.to_array(dim='ages').transpose("country", "time", "ages").astype(float)   #.rename({'age':'ages'})
            da = da.assign_coords(ages=[int(a) for a in da.ages.values])  # convert ages to ints

            # get ages and years
            ages = da.ages.values
            years = da.time.values

            df_cohort_sizes = da # TODO: rename this, its not a df anymore


    return df_cohort_sizes, ages, years 



@timeit
def interpolate_cohortsize_countries(
    cfg,
    df_cohort_sizes,
    cohort_ages,
    cohort_years,
    data_source = None,
    extend_method = 'linear',  # linear extends with constant value, slinear does spline linear extraplation
    startyear= 1950,
    endyear = None, # should be last birthyear of interest (2025) + max life expectancy 
): 
    """
    Interpolate cohortsizes from 5 year age brackets to year to year - only necessary for SSP data 
    """

    if data_source is None:
        data_source = cfg.cohort_sizes_source

    #set new coordinates for after interpolation - check you want this & put in flags at start or something !! 
    ages_interpn_cohorts =  np.arange(0,105) 
    years_interpn_cohorts = np.arange(startyear,endyear+1)

    if not data_source == 'UNWPP2024':

        # keep all possible countries (better, you lose less places)
        df_cohort_size_filter = df_cohort_sizes 
        
        def distribute_error_across_years(df_y_values, df_y_mean_bracket, bracket_size): 
            # for a single year / single country in the dataset distribute error in age bracket
            
            # ignore warnings, we get rid of nans later with the nansum
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                # reshape df to array, each row is a bracket, each column is a specific age in that bracket 
                y_values = np.reshape(df_y_values.values, (len(df_y_values)//bracket_size, bracket_size)) #nrows, ncols
                # calculate interpolation error over the bracket as the sum of errors
                delta_bracket = np.sum(y_values - df_y_mean_bracket.values[:, np.newaxis], axis=1) # sums along row
                # calculate relative weights as the value divided by the sum of all values in the bracket
                sum_over_years = np.sum(y_values, axis=1)
                weights = y_values / sum_over_years[:, np.newaxis]
                # compute correction for each y value
                delta_i = weights * delta_bracket[:, np.newaxis]
                # correct the y values 
                y_corrected = np.nansum(np.dstack((y_values,-delta_i)),2).reshape(-1)
                
            return y_corrected

        if cfg.version == 1:
            wcde_years, wcde_ages, wcde_country_data = (
                cohort_years,
                cohort_ages,
                df_cohort_size_filter.values,
            )
            countries = df_cohort_size_filter.index

        elif cfg.version == 2:
            wcde_years, wcde_ages = cohort_years, cohort_ages
            countries = df_cohort_size_filter.country.values

        else:
            raise ValueError("cfg.version must be 1 or 2")

        # initialise dictionary to store cohort sizes dataframes per country with years as rows and ages as columns
        d_cohort_size = {}
        
        # loop over countries
        print('interpolating cohort sizes per country')
        for i,name in enumerate(countries):
            # extract population size per age cohort data from WCDE file and linearly interpolate from 5-year WCDE blocks to pre-defined birth year
            
            if cfg.version == 1 : 
                wcde_per_country = np.reshape(wcde_country_data[i,:],((len(wcde_ages),len(wcde_years)))) 
                wcde_per_country_df = pd.DataFrame(
                    wcde_per_country,
                    index=wcde_ages,
                    columns=wcde_years
                )
            elif cfg.version == 2 : 
                wcde_per_country_df = df_cohort_size_filter.sel(country=name).to_pandas().T
                # every row is an age group (len 21), every column is a year (len 31)

            # Note: now using dataframes to do reindexing and interpolation (see how much slower this makes it cfr. to numpy
            # could do with numpy interpolate.griddata if you accept that ages 0-2 are not interpolated but held constant)

            # interpolate per ages
            wcde_per_country_df = wcde_per_country_df.reindex(ages_interpn_cohorts)
            wcde_per_country_intrp = wcde_per_country_df.astype('float').interpolate(
                    method=extend_method, # original 'linear' filled end values with constants; slinear calls spline linear interp/extrap from scipy interp1d - check if this is ok 
                    limit_direction='both',
                    fill_value='extrapolate',
                    axis=0
                )
            # set negative numbers to zero
            wcde_per_country_intrp[wcde_per_country_intrp<0]=0
            # fix the not mean preserving issue
            wcde_per_country_intrp_correct = wcde_per_country_intrp.copy()
            for y in wcde_years:
                wcde_per_country_intrp_correct.loc[:,y] = distribute_error_across_years(
                    wcde_per_country_intrp.loc[:,y], # interpolated values
                    wcde_per_country_df.dropna().loc[:,y], # true mean
                    bracket_size=5) # bracket size 
            
            # check for neg numbers
            if (wcde_per_country_intrp_correct < 0).any().any():
                print('after interpolation and mean-preserving correction there are some neg numbers in {}, {}, setting them to zero'.format(i,name))
                # set them to zero
                wcde_per_country_intrp_correct[wcde_per_country_intrp_correct<0]=0
        
            # interpolate between years
            wcde_per_country_df = wcde_per_country_intrp_correct.transpose().reindex(years_interpn_cohorts)
            wcde_per_country_intrp_years = wcde_per_country_df.astype('float').interpolate(
                    method=extend_method, # original 'linear' filled end values with constants; slinear calls spline linear interp/extrap from scipy interp1d
                    limit_direction='both',
                    fill_value='extrapolate',
                    axis=0
                )
            d_cohort_size[name] = wcde_per_country_intrp_years / 5 # divide by 5 for 5-year age groups
        
        #  make a data array with the information from all the countries together
        da_cohort_size = xr.DataArray(
            np.asarray([v for k,v in d_cohort_size.items()]), # see whether to include nan countries here
            coords={
                'country': ('country', countries),
                'time': ('time', years_interpn_cohorts),
                'ages': ('ages', ages_interpn_cohorts),
            },
            dims=[
                'country',
                'time',
                'ages',
            ],
            name='cohort_size'
             )

    else:

        if extend_method == 'linear':

            # Reindex and forward-fill
            da_cohort_size = df_cohort_sizes.reindex(time=years_interpn_cohorts, method="ffill").rename('cohort_size') # dont interpolate ages here or will change totals
        
        else:

            print(f'Error {extend_method} undefined for {data_source}')

    return da_cohort_size


# ---------------------------------
# 3. Gridded population and country masks
# ---------------------------------



@timeit
def load_population(
    cfg,
    dir_population=None,
    startyear=1950,
    endyear=2100,
    ssp=2,
    urbanrural=False,
    bbox=None, 
):
    """
    Load gridded population reconstructions (histsoc) + projections (SSPs).
    
    Version 1: From ISIMIP3. Gridded population density at 0.5 degrees, annual expressed as number of people (count). 
    ISIMIP3b has histsoc until 2021 (duplicated from ISIMIP3a), then from Gao et al. 2020 
    (https://doi.org/10.5065/D60Z721H AND https://doi.org/10.7927/q7z9-9r69),
    scaled to match ISIMIP national population projections under different SSPs. 
    Note: this has a known hist-to-ssp transition discontinuity spatially

    Version 2: from COMPASS, gridded pop count data, annual. Resolution defined in settings (0.1 or 0.5). 
    From Dominik Paprotny. 1950-2100 in historical + ssp1-5

    Input: 
        dir_population:         defined in settings, based on version. 
        urbanrural:             False loads only population total, True loads total, urban and rural variables - only available in v1
        startyear, endyear (int)
        ssp (int):              1,2 or 3
        bbox (optional):        tuple or array. (latmin, latmax, lonmin, lonmax) 
    
    Returns:
        da_population: (DataArray)  gridded population density. 

    """
    if dir_population is None:
        dir_population = cfg.dir_population

    # Auxiliary function to slice each dataset to a particular region and time 
    def cut_to_region_time(da):
        # time slice
        if da.time.dtype == 'datetime64[ns]':
            da['time'] = da['time'].dt.year
        else:
            da['time'] = da['time'].astype(int) + startyear_ssp

        if bbox is None:
            return da.sel(time=slice(startyear, endyear))
        # region slicing 
        latmin, latmax, lonmin, lonmax = bbox 
        #if da.lat.values[0] < da.lat.values[-1]: # check if lat is increasing or decreasing
        if da.lat.isel(lat=0) < da.lat.isel(lat=-1):# check if lat is increasing or decreasing
            return da.sel(
                lat=slice(latmin, latmax), lon=slice(lonmin, lonmax), time=slice(startyear, endyear)
                )
        else:
            return da.sel(
                lat=slice(latmax, latmin), lon=slice(lonmin, lonmax), time=slice(startyear, endyear)
                )


    # Initialize list to store datasets
    datasets = []

    if cfg.version == 1: 

        if urbanrural:
            VARs=['urban-population','rural-population','total-population']
        else:
            VARs='total-population'
        
        # for correct opening of times
        startyear_ssp = 2015

        # Load historical data conditionally based on the start and end year
        if startyear <= 1900:
            da_pop_histsoc1 = xr.open_mfdataset(
                os.path.join(dir_population, 'ISIMIP3/ISIMIP3b/histsoc/population_histsoc_30arcmin_annual_1850_1900.nc'),
                combine='nested',
                concat_dim='time',
                decode_coords='all',
                preprocess=cut_to_region_time
            )[VARs]
            datasets.append(da_pop_histsoc1)

        if startyear <= 2014 and endyear >= 1901:
            da_pop_histsoc2 = xr.open_mfdataset(
                os.path.join(dir_population, 'ISIMIP3/ISIMIP3b/histsoc/population_histsoc_30arcmin_annual_1901_2014.nc'),
                combine='nested',
                concat_dim='time',
                decode_coords='all',
                preprocess=cut_to_region_time
            )[VARs]
            datasets.append(da_pop_histsoc2)

        # Load SSP data conditionally
        if endyear >= 2015:
            print(f'opening isimip3 - ssp{ssp}')
            da_pop_sspsoc = xr.open_mfdataset(
                glob.glob(os.path.join(dir_population, f'ISIMIP3/ISIMIP3b/ssp{ssp}*/population_ssp{ssp}_30arcmin_annual_2015_2100.nc'))[0],
                combine='nested',
                concat_dim='time',
                decode_times=False,
                preprocess=cut_to_region_time
            )[VARs]
            da_pop_sspsoc['time'] = np.array([year for year in np.arange(2015, 2101)])
            da_pop_sspsoc = da_pop_sspsoc.sel(time=slice(max(startyear, 2015), endyear))
            datasets.append(da_pop_sspsoc)


    elif cfg.version == 2: 

        VARs='Population_count'
        startyear_ssp = 2025 

        if startyear <= 2025:
            print('opening compass - historical')
            da_pop_histsoc = xr.open_mfdataset(
                sorted(glob.glob(os.path.join(dir_population, 'historical/Population_count_*_historical.nc'))),
                combine='nested',
                concat_dim='time',
                decode_coords='all',
                preprocess=cut_to_region_time,
            )[VARs]
            datasets.append(da_pop_histsoc)

        # Load SSP data 
        if endyear >= 2026:
            print(f'opening compass - ssp{ssp}')
            da_pop_sspsoc = xr.open_mfdataset(
                sorted(glob.glob(os.path.join(dir_population, f'ssp{ssp}/Population_count_*_SSP{ssp}.nc'))),
                combine='nested',
                concat_dim='time',
                decode_coords='all',
                preprocess=cut_to_region_time,
            )[VARs]
            datasets.append(da_pop_sspsoc)

    # Concatenate datasets if there are multiple
    if len(datasets) > 1:
        da_population = xr.concat(datasets, dim='time')
    else:
        da_population = datasets[0]

    # extend past 2100 if necessary by filling with last year
    if endyear>2100:

        da_population = da_population.reindex(time=np.arange(startyear,endyear+1) , method="ffill")
    

    return da_population.rename('total-population')
    






@timeit
def load_countrymask(
    cfg,
    filepath_countrymask=None,
    data_source_countrymask=None,
    df_metadata=None,       
    da_population=None,
    fillcoast=False, # True if you want to preprocess and fill coastal pixels to not lose coastal pops (done already in preprocessed files)
    fix_smallislands=False, # done in preprocessed input files for 0.5, not for 0.1 - TODO: check if necessary at 0.1 or not ! 
    bbox=None,
    filter_countries=True,
    ):
    """
    Load countrymasks - shapefile or fractional mask
    for fractional mask there is option to fill coastal pixels so sum of fraction = 1 so coastal populations are not lost. 

    Inputs:
        filepath_countrymask (str)
        data_source_countrymask (str):      'fractional_mask' or 'shapefile'
        df_metadata (df):                   df with metadata, output from load_metadata function 
        da_population (da):                 only necessary for 'shapefile', the population dataarray to make masks
        fillcoast, fix_smallislands (bool): only for 'fractional_mask' whether to fix coastal pixels so fractions sum to one and fix small island states with errors (only coded for 0.5 deg)
        bbox (opt, array):                  miny, maxy, minx, maxx - bbox to crop the masks to 
        filter_countries (bool):            if True, removes from df_countries the countries not included in country_borders, and removes from country_borders the countries that aren't in df_metadata - NOTE: False is not tested 

    Returns:
        country_borders (da or gdf)
        country_regions, country_mask (regionmask objects - only if country_borders is gdf)
        df_countries (df)


    """

    if filepath_countrymask is None:
        filepath_countrymask = cfg.filepath_countrymask 
    if data_source_countrymask is None:
        data_source_countrymask = cfg.countrymask

    #TODO: divide into two different functions? for frax versus shapefile? 

    def cut_to_region(da, bbox):
        # cut to a predefined region
        latmin, latmax, lonmin, lonmax = bbox
        if da.lat.values[0] < da.lat.values[-1]:
            da = da.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
        else:
            da = da.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax))
        if "country" in da.dims:
            # compute which countries have all-NaN/0 inside the bbox and drop them 
            mask = ~((da.isnull() | (da == 0)).all(dim=("lat","lon")))
            return da.sel(country=mask)
        else:
            return da

    if data_source_countrymask == 'fractional_mask':

        if not fillcoast:
            # Open data - already preprocessed
            da_countrymasks = xr.open_dataarray(filepath_countrymask, chunks='auto')
            if "variable" in da_countrymasks.dims:
                da_countrymasks = da_countrymasks.isel(variable=0)

        # NOTE: could delete this whole section since the data is already preprocessed 
        if fillcoast:
            # Open data 
            ds=xr.open_dataset(filepath_countrymask, chunks='auto')
            da_countrymasks = ds.to_array()

            # clean variable names 
            strings = da_countrymasks['variable'].values
            cleaned_strings = [s[2:] if s.startswith('m_') else s for s in strings]
            da_countrymasks['variable'] = cleaned_strings
            # last variable is 'world', lose it 
            da_countrymasks = da_countrymasks.isel(variable=slice(0,225))

            # fill coastal pixels 
            # sum over all countries 
            countrymask_sum = da_countrymasks.sum(dim='variable')
            # correct for coastal pixels where sum of fraction is less than 1, weighted multiplication for sum to equal one
            da_countrymasks_correct = xr.where(countrymask_sum < 1, da_countrymasks * (1 / countrymask_sum ), da_countrymasks)
            # small area sum = 2, correct for it 
            da_countrymasks_corr = xr.where(da_countrymasks_correct.sum(dim='variable') > 1, da_countrymasks_correct/da_countrymasks_correct.sum(dim='variable'), da_countrymasks_correct)
            da_countrymasks = da_countrymasks_corr

            if fix_smallislands:  
                # TODO change the lat indexing to be with coords!! doesnt work for 0.1 - hard coded for 0.5 deg 
                # Fix issue in Singapore pixel, assign fraction from IOSID to SGP 
                da_countrymasks.loc[dict(lat=da_countrymasks.lat[177], lon=da_countrymasks.lon[567], variable='SGP')] += da_countrymasks.loc[dict(lat=da_countrymasks.lat[177], lon=da_countrymasks.lon[567], variable='IOSID')].values
                da_countrymasks.loc[dict(lat=da_countrymasks.lat[177], lon=da_countrymasks.lon[567], variable='IOSID')] = 0
                # Fix it also in Mauritius 
                da_countrymasks.loc[dict(lat=da_countrymasks.lat[220], lon=da_countrymasks.lon[474], variable='MUS')] += da_countrymasks.loc[dict(lat=da_countrymasks.lat[220], lon=da_countrymasks.lon[474], variable='IOSID')].values
                da_countrymasks.loc[dict(lat=da_countrymasks.lat[220], lon=da_countrymasks.lon[474], variable='IOSID')] = 0
            
            da_countrymasks = da_countrymasks.rename({'variable':'country'})

        if bbox:
            da_countrymasks = cut_to_region(da_countrymasks, bbox)

        if filter_countries: 
            # remove from df_countries those that are not in mask (total 209 countries )
            df_countries = df_metadata.merge(pd.DataFrame(da_countrymasks.country.to_pandas().rename('iso3_mask')), how='inner', left_on='abbreviation', right_on='country').drop(columns='iso3_mask')
            # remove from countrymask those that are not in df_countries 
            select = da_countrymasks.country.isin(df_countries['abbreviation'])
            da_countrymasks = da_countrymasks.sel(country=select)
        else:
            print('Note option to not filter countries based on df_metadata not tested')

        return da_countrymasks, None, None, df_countries 


    elif data_source_countrymask == 'shapefile':

        # open shapefile
        gdf_country_borders_raw = gpd.read_file(filepath_countrymask) # len:255
        df_countries = df_metadata

        #rename from incorrect in shapefile to correct in Worldbank. 
        d_rename={'KOS':'XKX', 'SDS':'SSD', 'PSX':'PSE'}
        gdf_country_borders = gdf_country_borders_raw.replace({"ADM0_A3": d_rename})

        if filter_countries:
            # keep only if in both world bank and gdf : 217 countries 
            gdf_country_borders = gdf_country_borders.merge(
                df_metadata, 
                how='inner',
                left_on='ADM0_A3', 
                right_on='abbreviation'
                )
        else:
            print('Error no option to not filter countries based on df_metadata')


        # if bbox provided, crop the geodataframe and the population data  
        if bbox:
            latmin, latmax, lonmin, lonmax = bbox
            box_crop = box(lonmin, latmin, lonmax, latmax)
            gdf_country_borders = gpd.clip(gdf_country_borders, box_crop)

            da_population = cut_to_region(da_population, bbox) # TODO: am i using this??? 

        # create regions object and mask object
        countries_regions = regionmask.from_geopandas(
            gdf_country_borders, 
            names='name', 
            abbrevs="abbreviation", 
            name="country"
        )
        countries_mask = countries_regions.mask(da_population.lon, da_population.lat)

        # remove countries that have zero population (not resolved in mask)
        df_countries['population'] = np.nan 
        for name in df_countries.index.values: 
            if name in gdf_country_borders['name'].values:
                # only keep countries that are resolved with mask 
                if da_population.where(countries_mask==countries_regions.map_keys(name), drop=True).size != 0:
                    # get mask index and get masked population in 2025 to drop countries that are not resolved with shapefile
                    df_countries.loc[name,'population'] = da_population.sel(time=2025).where(countries_mask==countries_regions.map_keys(name), drop=True).sum().values
        # remove countries that have zero population - total 7.6 billion people covered by 183 countries and shapefile masking (out of 8.2 billion) - use dem4cli gridscale v1 if absolute number is important at gridscale
        df_countries = df_countries[~df_countries.loc[:, 'population'].isnull()]
        
        # clean country borders dataframe for return
        gdf_country_borders = gdf_country_borders.set_index(gdf_country_borders.name
                                    ).loc[:,['geometry','region', 'ADM0_A3']].rename(columns={'ADM0_A3':'abbreviation'}
                                    ).reindex(df_countries.index)




        return gdf_country_borders, countries_regions, countries_mask, df_countries





@timeit
def load_subnational_mask(
    cfg,
    filepath_shp=None,
    da_population=None,
    bbox=None,
    dict_keep=None, 
    dict_drop=None,
    col_name="NAME_LATN",
    col_id= "NUTS_ID",
    col_country='CNTR_CODE',
    ):
    """
    Load subnational shapefile mask 

    Inputs:
        dict_keep (dict):       what elements of shapefile to keep, as a dictionary colname:value 
                                e.g. {'LEVL_CODE': 2, 'CNTR_CODE': 'PT'}
                                This does an exact match
        dict_drop (dict):       what elements to drop as a dictionary colname:value
                                This does a "startswith" match

    Returns:

    """

    if filepath_shp is None:
        filepath_shp = cfg.filepath_shp_subnational

    def make_shp(filepath_shp, dict_keep=None, dict_drop=None):
        shp = gpd.read_file(filepath_shp)

        if dict_keep:
            for key, val in dict_keep.items():
                vals = [val] if isinstance(val, (str, int)) else val
                shp = shp[shp[key].isin(vals)]

        if dict_drop:
            for key, val in dict_drop.items():
                vals = [val] if isinstance(val, (str, int)) else val
                shp = shp[~shp[key].str.startswith(tuple(vals))]

        return shp.reset_index(drop=True).to_crs("EPSG:4326")
    
    def cut_to_region(da, bbox):
        # cut to a predefined region
        latmin, latmax, lonmin, lonmax = bbox
        if da.lat.values[0] < da.lat.values[-1]:
            da = da.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
        else:
            da = da.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax))
        if "country" in da.dims:
            # compute which countries have all-NaN/0 inside the bbox and drop them 
            mask = ~((da.isnull() | (da == 0)).all(dim=("lat","lon")))
            return da.sel(country=mask)
        else:
            return da
    
    gdf = make_shp(filepath_shp, dict_keep=dict_keep, dict_drop=dict_drop)

    if bbox:
        latmin, latmax, lonmin, lonmax = bbox
        # crop to same area 
        box_crop = box(lonmin, latmin, lonmax, latmax) # not sure this is necessary?
        gdf = gpd.clip(gdf, box_crop).reset_index(drop=True)
        da_population = cut_to_region(da_population, bbox)  # TODO: do i really need this ?? 
                                                            # it will make mask on grid of da_population, so it needs to be relevant to broader analysis
                                                            # check if better to force bbox to be provided... or to alternatively not crop da_population 

    # sort alphabetically based on ID (not necessary, but looks cleaner)
    gdf = gdf.sort_values(col_id).reset_index(drop=True)

    # create regions object and mask object
    subnational_regions = regionmask.from_geopandas(
        gdf, 
        names=col_id, 
        abbrevs=col_id, 
        name=col_name
    )

    subnational_mask = subnational_regions.mask(da_population.lon, da_population.lat)

    gdf = gdf.loc[:,[col_id, col_country, col_name, 'geometry']].rename(
            columns={col_id:'id', col_name:'name', col_country:'country'}
            ).set_index('id', drop=False).rename_axis(None)

    # calc population in 2025 from gridded data, to check if some regions are unresolved at the resolution
    gdf['population'] = np.nan
    for idx in gdf['id']:
        gdf.loc[idx,'population'] = da_population.sel(time=2025
        ).where(subnational_mask==subnational_regions.map_keys(idx), drop=True
        ).sum().values

    # possible to add automatic dropping of empty regions

    return gdf, subnational_regions, subnational_mask



# ---------------------------------
# 4. Life expectancy 
# ---------------------------------




@timeit
def load_unwpp_lifeexpectancy(
    cfg,
        filepath_lifeexpectancy = None,
        start_birthyear=1950,
        end_birthyear=2025
):
    """
    Load UNWPP2024 data on e(x) = Life Expectancy at Exact Age x (ex) - Both Sexes.

    The average number of remaining years of life expected by a hypothetical cohort of individuals alive at age x who would be subject during the 
    remaining of their lives to the mortality rates of a given year. It is expressed as years. Has data from birth year 1950 to 2023. 

    Keep only Country name and years left to live at age 5 e(5). 

    Source: 
    https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Mortality
    
    """

    if filepath_lifeexpectancy is None:
        filepath_lifeexpectancy = cfg.filepath_lifeexpectancy

    df_list = []

    for sheet in [0,1]: # sheet 0 has reconstructions up to 2023, sheet 2 has projections
        df_unwpp_raw = pd.read_excel(filepath_lifeexpectancy, 
                sheet_name=sheet,
                skiprows=16) # make this more flex 
        
        df_unwpp = df_unwpp_raw[df_unwpp_raw['Type']=='Country/Area'].rename(
                columns={'Region, subregion, country or area *':'Country'}) # make this more flex 
        
        cols = df_unwpp.columns
        

        # get only life expectancy at age 5
        idxs = [i for i, col in enumerate(cols) if col in ('Country',  'Year', 5)] # or: 'ISO3 Alpha-code'
        # decide whether to keep country name or ISO3 
        # probably better ISO3 ! 
        
        df_unwpp = df_unwpp.iloc[:, idxs].pivot(
            index='Year',
            columns='Country',
            values=5)
        
        # years left to live of someone who is 5 years old in that year
        
        df_unwpp.index = df_unwpp.index.astype(int)

        df_list.append(df_unwpp)

    df_unwpp = pd.concat(df_list, axis=0).loc[start_birthyear:int(end_birthyear + 5)]

    return df_unwpp



@timeit
def get_life_expectancies(df_unwpp,
                         start_birthyear=1950,
                         end_birthyear=2025):
    
    """
    Takes UNWPP life expectancy data expressed as years left to live at age of 5, 
    subtracts 5 from Year to get it at birth year but ignoring infant mortality, 
    adds 5 to account for the 5 years of life already lived, adds 6 to account for increase 
    in life expectancy through the life of an individual (i.e. move from "period" life expectancy to 
    "cohort" life expectancy, see Goldstein & Wachter (2006) "Relationships between period and cohort 
    life expectancy: Gaps and lags")

    Thus get life expectancy in each year for each country at birth 
    expressed in "cohort" way, neglecting infant mortality.

    Adter end of data, extends by filling with constant value 

    Inputs


    Returns 

    """
    
    df_life_expectancy_5 = df_unwpp.copy()
    df_life_expectancy_5.index = df_life_expectancy_5.index-5 # year of birth: 2023 (age 5) becomes 2018 (age 0)
    df_life_expectancy_5 = df_life_expectancy_5 + 5 + 6 

    if df_life_expectancy_5.index[-1] < end_birthyear :
        # extend for last years
        df_life_expectancy_5_extend = df_life_expectancy_5.reindex(
                    np.arange(start_birthyear,end_birthyear+1)).astype( 
                    'float').interpolate() # extrapolation: fills last years constant 
    
        return df_life_expectancy_5_extend
    else:
        return df_life_expectancy_5.loc[start_birthyear:end_birthyear ]




# -----------------------------------------------------
# 5. Wrapper function demographic data country level
# -----------------------------------------------------


@timeit
def preprocess_all_country_data(
    cfg,
    start_birthyear=1950,
    end_birthyear=2025,
    ssp=2,
    extend_method="linear",
    by_sex=False,
    urbanrural=False,
    bbox=None,
    fillcoast=False,
    fix_smallislands=False,
    filter_countries=True,
    worldbank_filter=True,
):
    filepath_lifeexpectancy = cfg.filepath_lifeexpectancy
    dir_cohortsizes = cfg.dir_cohortsizes
    data_source_cohorts = cfg.cohort_sizes_source
    dir_population = cfg.dir_population
    filepath_countrymask = cfg.filepath_countrymask
    data_source_countrymask = cfg.countrymask
    filepath_world_bank = cfg.filepath_world_bank_meta
    filepath_lookuptable = cfg.filepath_lookuptable

    # metadata from worldbank, unwpp and availability of cohort data - filters already countries 
    df_metadata =  load_country_metadata(cfg,
                                        filepath_world_bank = filepath_world_bank,
                                        filepath_lookuptable=filepath_lookuptable,
                                        data_source_cohorts = data_source_cohorts,
                                        worldbank_filter=worldbank_filter) 


    # load life expectancy data and clean 
    df_unwpp = load_unwpp_lifeexpectancy(cfg, filepath_lifeexpectancy = filepath_lifeexpectancy) 
    # go from 'period' to 'cohort' life expectancy
    df_life_expectancy_5 = get_life_expectancies(df_unwpp,
                                            start_birthyear=start_birthyear,
                                            end_birthyear=end_birthyear)


    # calculate end year as last birth year + maximum life expectancy
    # cohort sizes are extrapolated, gridded pop data is held constant (check!)
    endyear = ceil(max(df_life_expectancy_5.values.flatten()) + end_birthyear)


    # loads raw cohort size from WCDE ssps or UNWPP2024 (reconstruction + projections) and cleans to keep only relevant information
    df_cohort_sizes, ages, years = load_cohort_sizes(cfg, dir_cohortsizes, data_source=data_source_cohorts, ssp=ssp, by_sex=by_sex)
    # for WCDE, interpolates cohort sizes from 5 year to single year and corrects to preserve mean and extends past 2100
    # for UNWPP extends past 2100 only
    da_cohort_size = interpolate_cohortsize_countries(
                        cfg,
                        df_cohort_sizes,
                        ages,
                        years,
                        data_source=data_source_cohorts,
                        extend_method=extend_method, 
                        startyear=start_birthyear,
                        endyear=endyear,
                    )


    # load gridded population data, optional cropping in space and time
    da_population = load_population(
                        cfg,
                        dir_population= dir_population, 
                        startyear=start_birthyear,
                        endyear=endyear,  
                        ssp=ssp,
                        urbanrural=urbanrural,
                        bbox = bbox ,
                        )

    # open countrymasks, optional preprocessing (already done in default input files)
    country_borders, countries_regions, countries_mask, df_countries = load_countrymask(
                                        cfg,
                                        filepath_countrymask,
                                        data_source_countrymask = data_source_countrymask,
                                        df_metadata=df_metadata,
                                        da_population = da_population,
                                        fillcoast=fillcoast, # fill coastal pixels to not lose coastal pops
                                        fix_smallislands=fix_smallislands, # done in preprocessed input files for 0.5, not for 0.1
                                        bbox=bbox,
                                        filter_countries=filter_countries,
                                        )
    
    if filter_countries:
        # life expectancy
        df_life_expectancy_5 = df_life_expectancy_5[df_countries["name"]]
        
        # cohort sizes
        name_cohorts = "name" if data_source_cohorts == "UNWPP2024" else "SSP name"
        da_cohort_size = da_cohort_size.sel(country=df_countries[name_cohorts].to_list()) # rename the SSP name to the WPP name?
        
        # if cohort sizes are from WCDE rename from SSP name to WPP name
        if data_source_cohorts == 'WCDE':
            mapping = dict(zip(df_countries['SSP name'], df_countries['name']))
            da_cohort_size = da_cohort_size.assign_coords(
                country = [mapping[c] for c in da_cohort_size.country.values]
            )

        # WCDE: in demographic datasets and have world bank region/income info = 195 countries
                # and shapefile resolved = 180 countries
                # and frax mask resolved = 192 countries
        # UNWPP: in demographic datasets and have world bank region/income info = 217 countries
                # and shapefile resolved = 183 countries
                # and frax mask resolved = 209 countries
        

    # pack country information
    d_countries = {
        'info_pop': df_countries,
        'borders': country_borders,     
        'population_map': da_population,
        'birth_years': None,
        'life_expectancy_5': df_life_expectancy_5, 
        'cohort_size': da_cohort_size,
        'mask': (countries_regions,countries_mask),                  
    }


    return d_countries







# ---------------------------------------------
# 6. Wrapper function gridscale demographics
# ---------------------------------------------



def get_gridscale_demographics(
    da_population,
    da_countrymasks,
    df_countries_matched, 
    da_cohort_size,
    startyear=2000,
    endyear=2005,
    #chunksize=100
):
    """
    To do: make a wrapper function that runs all previous and does this
    make a function that does this just for one country/region if one only wants a certain country? - doing it ! to clean up nicer later 
    """

    da_pop = da_population.sel(time=slice(startyear, endyear))   # TODO: check optimal chunking sizes and whether to chunk here or above,myabe here?  #.chunk({'time': chunksize, 'lat': chunksize, 'lon': chunksize})
    
    # Initialize the combined demographics DataArray
    da_pop_demographics = None

    
    # Option for running over one country only 
    if da_cohort_size.country.values.size > 1:
        ls_countries = da_cohort_size.country.values
    else:
        ls_countries = [da_cohort_size.country.values]


    # Loop over countries in WCDE cohort sizes
    for country in ls_countries:
        print(country)
    
        # Get iso3 code of the country in the mask 
        iso = df_countries_matched[df_countries_matched['country_wcde']==country]['iso3_frac'].values[0]
    
        # if this isocode is in the mask file 
        if iso in da_countrymasks['variable']: # TODO: do this in a slightly more intelligent way??? similar to what i was doing b4 with the dataframs, instead of if
        
            # Get cohort sizes of the country
            if da_cohort_size.country.values.size > 1:
                da_smple_cht = da_cohort_size.sel(country=country).sel(time=slice(startyear, endyear)) 
            else:
                da_smple_cht = da_cohort_size.sel(time=slice(startyear, endyear)) 

            # Cohort relative sizes in the sample country
            da_smple_cht_prp = da_smple_cht / da_smple_cht.sum(dim='ages')
        
            # Get population of that country and multiply by fraction of each cohort
            pop_country = ((da_pop * da_countrymasks.sel(variable=iso)) * da_smple_cht_prp).drop_vars(['variable', 'country'])
        
            if da_pop_demographics is None:
                da_pop_demographics = pop_country
            else:
                da_pop_demographics += pop_country
        
            # Explicitly clear intermediate variables to free up memory
            del iso, da_smple_cht, da_smple_cht_prp, pop_country
        
        else:

            print(f'**iso {iso} not in mask')

    
    da_pop_demographics = da_pop_demographics.compute()
    
    return da_pop_demographics

