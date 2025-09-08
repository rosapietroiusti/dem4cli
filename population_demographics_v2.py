"""
Demographics4Climate : Population and demographics for climate science analysis
----------------------------------------------

Update 2025 with new data 

> udnerstand how best to deal with having v1 and v2... 

options
- have only one pop_demographics file but make flags, outside of functions or inside functions saying which functions get loaded.... 

- make wrapper function for S2S 

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

from _utils import * 
from _settings import *

#%%
# ---------------------------------
# 1. Metadata 
# ---------------------------------

# is this getting used anywhere? 
# only v1

def load_country_metadata(
    filepath_isimip_countries = os.path.join(script_dir, 'data/country-masks/isipedia-countries/countryData.json'),
    filepath_world_bank = os.path.join(script_dir, 'data/income-groups/world_bank/CLASS.xlsx'),
    keep_names='isimip',
    keep_stats=False,

):
    """
    load country list from isipedia-coutries (country masks metadata files from Perette 2023, https://github.com/ISI-MIP/isipedia-countries). For 195 official/observer UN countries. 
    and metadata from worldbank.

    Input
        keep_names (str) what country names to keep, can be 'isimip', 'world_bank', 'both'  
        filepath_isimip_countries
        filepath_world_bank
        keep stats (Bool) from isimip_countries 
    
    Returns
        df_metadata: table with country name, ISO3 code, country code, region and income group, where available
    
    """

    # open isimip metadata  
    df_isimip_metadata = pd.read_json(filepath_isimip_countries).replace(-9999, np.nan)
    # open world bank metadata
    df_wb_countries = pd.read_excel(filepath_world_bank, sheet_name=0)
    # merge keep list of countries from isimip and info from world bank
    df_merge = df_isimip_metadata.merge(df_wb_countries, how='inner',left_on='country_iso3', right_on='Code')

    # keep only some of the info and clean up column names 
    if keep_names =='isimip':
        keep_cols = ['country', 'Code', 'country_code','Region', 'Income group']
        d_rename = {'Code':'country_iso3', 'Region':'region', 'Income group': 'income_group'}
        
    elif keep_names == 'world_bank':
        keep_cols =['Economy', 'Code', 'country_code','Region', 'Income group']
        d_rename={'Economy':'country','Code':'country_iso3', 'Region':'region', 'Income group': 'income_group'}
        
    elif keep_names == 'both':
        keep_cols=['country','Economy', 'Code', 'country_code','Region', 'Income group']
        d_rename={'Economy':'country_wb','Code':'country_iso3', 'Region':'region', 'Income group': 'income_group'}     

    if keep_stats == True:
        keep_cols=keep_cols+list(df_isimip_metadata.columns[3:])

    df_metadata = df_merge[keep_cols].rename(columns=d_rename) #.head(196) # 'Economy', 
        
    return df_metadata



# COULD DELETE THIS ! Not getting used 
#v1

def load_country_stats(
    filepath_isimip_stats = os.path.join(script_dir, 'data/country-masks/isipedia-countries/countryprofiledata.json')
                      ):
    """
    Load statistics for 195 official/observer UN countries from isipedia-countries. 
    """

    df_isimip_stats = pd.read_json(filepath_isimip_stats).T.reset_index(drop=True).replace(-9999, np.nan).rename(columns={'iso3':'country_iso3'})

    return df_isimip_stats



# ---------------------------------
# 2. Cohort sizes
# ---------------------------------



def load_cohort_sizes( 
    dir_cohortsizes = dir_cohortsizes,
    ssp = 2,
    by_sex = False
):
    """
    load population size per age cohort from Wittgenstein Center Data Explorer.
    
    Version 1: (source: http://dataexplorer.wittgensteincentre.org/wcde-v2/)

    Version 2: version 3.2 beta (for CMIP7)

    data description: Population Size (000's)
    De facto population in a country or region, classified by sex and by five-year age groups. Available in all scenarios and at all geographical scales. For each country data is sorted first by age cohort (0-4, 4-9...). So all the first data refers to the 0-4 age cohort. 
    Then they give the population size of that cohort at a snapshot every 5 years (1950, 1955, 1960...).
    Here we assign the data to the central age cohort (i.e. 0-4 assigned to 2).
    
    Input
        filepaths_wcde (str): path to csv files for different ssps
        ssp (int): 1,2 or 3 for ssp1, ssp2, ssp3
        by_sex (Bool): TODO (data is available male/female) - in version 2 this is automatic

    Returns
        df_cohort_sizes (df): rows are countries, columns are a cohort's (e.g. age=2) 
                                size each year, then the next cohort (columns labelled e.g. 2_1950 age=2, year=1950) 
                                (version 1, in version 2 its a self-explanatory data array)
        ages (arr) : central year of interval (2,7...102)
        years (arr) : years we have data for (1950, 1955...2100)

    Note: this also exists from UNWPP2024! Could be more consistent with life expectancy data 
    """

    def convert_age_range(age):
        if age == '100+':
            return 100
        else:
            match = re.match(r'(\d+)-{1,2}(\d+)', age)
            if match:
                return int(match.group(1))
            else:
                return int(age)
            

    if flags['version'] == 1:

        # open wcde cohort size file 
        filepath = dir_cohortsizes+f'/wicdf_ssp{ssp}.csv'
        df_raw = pd.read_csv(filepath, header=7) # population is in 000's

        # cohort size specific
        if not by_sex:

            # select only relevant rows and cols
            df = df_raw[(df_raw['Sex'] == 'Both') & (df_raw['Age'] != 'All')][['Area', 'Year', 'Age', 'Population']]
            
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
            # TO DEVELOP ! BY SEX ! 

    elif flags['version'] == 2:

        filepath = dir_cohortsizes+f'/ssp_basic_drivers_release_3.2.beta_full.xlsx'
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
        da = da.sel(ssp=['historical', f'SSP{ssp}']).max(dim="ssp")

        # get ages and years
        ages = da.age.values
        years = da.time.values

        if not by_sex:
            da = da.sum(dim='sex') # in the alternative it doesnt really get used anyway... 

        df_cohort_sizes = da.where(da.country != 'World', drop=True)
 
        # note its a da not a df !!! TODO: change the version 1 so it also gives a da? or rename this to cohortsizes_raw or something! 

    return df_cohort_sizes, ages, years



def interpolate_cohortsize_countries(
    df_cohort_sizes,
    cohort_ages,
    cohort_years,
): 
    """
    Interpolate cohortsizes from 5 year age brackets to year to year
    """

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

    if flags ['version'] == 1 : 
        wcde_years, wcde_ages, wcde_country_data = cohort_years, cohort_ages, df_cohort_size_filter.values 
        countries = df_cohort_size_filter.index

    elif flags ['version'] == 2 : 
        wcde_years, wcde_ages = cohort_years, cohort_ages
        countries = df_cohort_size_filter.country.values # its not a df its a da in v2! 
    
    # initialise dictionary to store cohort sizes dataframes per country with years as rows and ages as columns
    d_cohort_size = {}
    
    # loop over countries
    print('interpolating cohort sizes per country')
    for i,name in enumerate(countries):
        # extract population size per age cohort data from WCDE file and linearly interpolate from 5-year WCDE blocks to pre-defined birth year
        
        if flags ['version'] == 1 : 
            wcde_per_country = np.reshape(wcde_country_data[i,:],((len(wcde_ages),len(wcde_years)))) 
            wcde_per_country_df = pd.DataFrame(
                wcde_per_country,
                index=wcde_ages,
                columns=wcde_years
            )
        elif flags ['version'] == 2 : 
            wcde_per_country_df = df_cohort_size_filter.sel(country=name).to_pandas().T
            # every row is an age group (len 21), every column is a year (len 31)

        # Note: now using dataframes to do reindexing and interpolation (see how much slower this makes it cfr. to numpy
        # could do with numpy interpolate.griddata if you accept that ages 0-2 are not interpolated but held constant 
        # decide what assumption we want to use!) 

        #set new coordinates for after interpolation - check you want this & put in flags at start or something !! 
        ages_interpn_cohorts =  np.arange(0,105) 
        years_interpn_cohorts = np.arange(1950,2100+1)
    
        # interpolate per ages
        wcde_per_country_df = wcde_per_country_df.reindex(ages_interpn_cohorts)
        wcde_per_country_df
        wcde_per_country_intrp = wcde_per_country_df.astype('float').interpolate(
                method='slinear', # original 'linear' filled end values with constants; slinear calls spline linear interp/extrap from scipy interp1d - check if this is ok 
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
            # TODO: modify distribute_error_across_years to not reintroduce negative numbers? 
    
        # interpolate between years
        wcde_per_country_df = wcde_per_country_intrp_correct.transpose().reindex(years_interpn_cohorts)
        wcde_per_country_intrp_years = wcde_per_country_df.astype('float').interpolate(
                method='slinear', # original 'linear' filled end values with constants; slinear calls spline linear interp/extrap from scipy interp1d
                limit_direction='both',
                fill_value='extrapolate',
                axis=0
            )
        d_cohort_size[name] = wcde_per_country_intrp_years / 5
    
        #  make a data array with the information from all the countries together
    da_cohort_size = xr.DataArray(
        np.asarray([v for k,v in d_cohort_size.items()]), # see whether to include nan countries here -  np.asarray([v for k,v in d_cohort_size.items() if k in df_cohort_size_filter['country'].values])
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


    return da_cohort_size


# ---------------------------------
# 3. Gridded population and country masks
# ---------------------------------



def load_population(
    dir_population= dir_population, 
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
        if da.lat.values[0] < da.lat.values[-1]: # check if lat is increasing or decreasing
            return da.sel(
                lat=slice(latmin, latmax), lon=slice(lonmin, lonmax), time=slice(startyear, endyear)
                )
        else:
            return da.sel(
                lat=slice(latmax, latmin), lon=slice(lonmin, lonmax), time=slice(startyear, endyear)
                )


    # Initialize list to store datasets
    datasets = []

    if flags['version'] == 1: 

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
            #da_pop_histsoc1['time'] = da_pop_histsoc1['time'].dt.year
            #da_pop_histsoc1 = da_pop_histsoc1.sel(time=slice(startyear, min(endyear, 1900)))
            datasets.append(da_pop_histsoc1)

        if startyear <= 2014 and endyear >= 1901:
            da_pop_histsoc2 = xr.open_mfdataset(
                os.path.join(dir_population, 'ISIMIP3/ISIMIP3b/histsoc/population_histsoc_30arcmin_annual_1901_2014.nc'),
                combine='nested',
                concat_dim='time',
                decode_coords='all',
                preprocess=cut_to_region_time
            )[VARs]
            #da_pop_histsoc2['time'] = da_pop_histsoc2['time'].dt.year
            #da_pop_histsoc2 = da_pop_histsoc2.sel(time=slice(max(startyear, 1901), min(endyear, 2014)))
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


    elif flags['version'] == 2: 

        VARs='Population_count'
        startyear_ssp = 2025 

        if startyear <= 2025:
            print(f'opening compass - historical')
            da_pop_histsoc = xr.open_mfdataset(
                sorted(glob.glob(os.path.join(dir_population, f'historical/Population_count_*_historical.nc'))),
                combine='nested',
                concat_dim='time',
                decode_coords='all',
                preprocess=cut_to_region_time,
            )[VARs]
            #da_pop_histsoc['time'] = da_pop_histsoc.time.dt.year
            #da_pop_histsoc = da_pop_histsoc.sel(time=slice(startyear, endyear))
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
            #da_pop_sspsoc['time'] = da_pop_sspsoc.time.dt.year
            #da_pop_sspsoc = da_pop_sspsoc.sel(time=slice(max(startyear, 2015), endyear))
            datasets.append(da_pop_sspsoc)

    # Concatenate datasets if there are multiple
    if len(datasets) > 1:
        da_population = xr.concat(datasets, dim='time')
    else:
        da_population = datasets[0]
    

    return da_population.rename('total-population')
    







def load_countrymasks_fillcoasts(
    filepath_countrymask=filepath_countrymask,
    preprocess=False, # True if you want to preprocess
    fillcoast=False, # fill coastal pixels to not lose coastal pops (done in preprocessed files)
    fix_smallislands=False, # done in preprocessed input files for 0.5, not for 0.1 - TODO: check if necessary at 0.1 or not ! 
    bbox=None,
    ):
    """
    Load countrymasks - option to fill coastal pixels so sum of fraction = 1 so coastal populations are not lost. 

    """

    def cut_to_region(da, bbox):
        # cut to a predefined region
        latmin, latmax, lonmin, lonmax = bbox
        if da.lat.values[0] < da.lat.values[-1]:
            da = da.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
        else:
            da = da.sel(lat=slice(latmax, latmin), lon=slice(lonmin, lonmax))
        # compute which countries have all-NaN/0 inside the bbox and drop them 
        mask = ~((da.isnull() | (da == 0)).all(dim=("lat","lon")))
        return da.sel(country=mask)

    if not preprocess:
        # Open data - already preprocessed
        da_countrymasks = xr.open_dataarray(filepath_countrymask, chunks='auto')
        if "variable" in da_countrymasks.dims:
            da_countrymasks = da_countrymasks.isel(variable=0)


    if preprocess:
        # Open data 
        ds=xr.open_dataset(filepath_countrymask, chunks='auto')
        da_countrymasks = ds.to_array()
        # clean
        strings = da_countrymasks['variable'].values
        cleaned_strings = [s[2:] if s.startswith('m_') else s for s in strings]
        da_countrymasks['variable'] = cleaned_strings
        # last variable is 'world', lose it 
        da_countrymasks = da_countrymasks.isel(variable=slice(0,225))

        if fillcoast:
            # sum over all countries 
            countrymask_sum = da_countrymasks.sum(dim='variable')
            # Part 2. Correct for coastal pixels 
            # where sum of fraction is less than 1, weighted multiplication for sum to equal one
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

    if not bbox:
        return da_countrymasks
    else: 
        return cut_to_region(da_countrymasks, bbox)







# ---------------------------------
# 4. Life expectancy 
# ---------------------------------




def load_unwpp_lifeexpectancy(
        filepath_lifeexpectancy = filepath_lifeexpectancy
):
    """
    Load UNWPP2024 data on e(x) = Life Expectancy at Exact Age x (ex) - Both Sexes.

    The average number of remaining years of life expected by a hypothetical cohort of individuals alive at age x who would be subject during the 
    remaining of their lives to the mortality rates of a given year. It is expressed as years. Has data from birth year 1950 to 2023. 

    Keep only Country name and years left to live at age 5 e(5). 

    Source: 
    https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Mortality
    
    """

    
    df_unwpp_raw = pd.read_excel(filepath_lifeexpectancy, 
             sheet_name=0,
             skiprows=16) # make this more flex 
    
    df_unwpp = df_unwpp_raw[df_unwpp_raw['Type']=='Country/Area'].rename(
              columns={'Region, subregion, country or area *':'Country'}) # make this more flex 
    
    cols = df_unwpp.columns
    

    # get only life expectancy at age 5
    idxs = [i for i, col in enumerate(cols) if col in ('Country',  'Year', 5)] #  'ISO3 Alpha-code'
    # decide whether to keep country name or ISO3 
    # probably better ISO3 ! 
    
    df_unwpp = df_unwpp.iloc[:, idxs].pivot(
        index='Year',
        columns='Country',
        values=5)
    
    # years left to live of someone who is 5 years old in that year
    
    df_unwpp.index = df_unwpp.index.astype(int)

    return df_unwpp



def get_life_expectancies(df_unwpp,
                         extend=True,
                         start_birthyear=1950,
                         end_birthyear=2025):
    
    """
    - Takes UNWPP life expectancy data expressed as years left to live at age of 5, 
    subtracts 5 from Year to get it at birth year but ignoring infant mortality, 
    adds 5 to account for the 5 years of life already lived, adds 6 to account for increase 
    in life expectancy through the life of an individual (i.e. move from "period" life expectancy to 
    "cohort" life expectancy, see Goldstein & Wachter (2006) "Relationships between period and cohort 
    life expectancy: Gaps and lags")
    - Thus get life expectancy in each year for each country at birth 
    expressed in "cohort" way, neglecting infant mortality.
    - Data ends for 2018 cohort (5 y.o. in 2023), extends by filling with constant value 

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
        return df_life_expectancy_5












# -----------------------------------------------------
# 5. Wrapper function demographic data country level
# -----------------------------------------------------


def preprocess_all_country_data(

    dir_cohortsizes = dir_cohortsizes,  # cohort size data
    ssp=2, 
    by_sex=False,                       # NOTE by_sex not implemented
                                            
    filepath_lifeexpectancy = filepath_lifeexpectancy, # life expectancy data
    start_birthyear=1950,
    end_birthyear=2025, 

    dir_population= dir_population,     # gridded pop data 
    startyear=1950,
    endyear=2100,
    urbanrural=False,                   # NOTE urbanrural not implemented for v2
    bbox = None,

    filepath_countrymask = filepath_countrymask,    # country masks 
    preprocess=False,                               # NOTE preprocessing is already done in standard input files 
    fillcoast=False, 
    fix_smallislands=False,

    ):

    df_cohort_sizes, ages, years = load_cohort_sizes(dir_cohortsizes, ssp=ssp, by_sex=by_sex)
    # loads raw cohort size data from historical + ssp and cleans to keep only relevant information

    da_cohort_size = interpolate_cohortsize_countries(
                        df_cohort_sizes,
                        ages,
                        years,
                    )
    # interpolates cohort sizes from 5 year to single year and corrects to preserve mean


    df_unwpp = load_unwpp_lifeexpectancy(filepath_lifeexpectancy = filepath_lifeexpectancy) 
    # load life expectancy data and clean 

    df_life_expectancy_5 = get_life_expectancies(df_unwpp,
                                            start_birthyear=start_birthyear,
                                            end_birthyear=end_birthyear) 
    # go from 'period' to 'cohort' life expectancy

    da_population = load_population(
    dir_population= dir_population, 
    startyear=startyear,
    endyear=endyear,
    ssp=ssp,
    urbanrural=urbanrural,
    bbox = bbox 
    )
    # load gridded population data, optional cropping

    da_countrymasks = load_countrymasks_fillcoasts(
                            filepath_countrymask,
                            preprocess=preprocess, 
                            fillcoast=fillcoast, # fill coastal pixels to not lose coastal pops 
                            fix_smallislands=fix_smallislands, # done in preprocessed input files for 0.5, not for 0.1 
                            bbox=bbox,
                            )
    # open countrymasks, optional preprocessing (already done in default input files)


    # pack country information
    d_countries = {
        'info_pop': None, 
        'borders': da_countrymasks, # note this is now a dataarray not geodf borders
        'population_map': da_population,
        'birth_years': None,
        'life_expectancy_5': df_life_expectancy_5, 
        'cohort_size': da_cohort_size,
        'mask': None,
    }


    # TODO: harmonize what countries are included in d_countries? based on overlap between all sources 
    # otherwise you have all the countries in all the objects
    # TODO: see if Amaury needs other objects I didn't include in d_countries 

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
    chunksize=100
):
    """
    To do: make a wrapper function that runs all previous and does this
    make a function that does this just for one country/region if one only wants a certain country? - doing it ! to clean up nicer later 
    """

    da_pop = da_population.sel(time=slice(startyear, endyear)) #.chunk({'time': chunksize, 'lat': chunksize, 'lon': chunksize})  # check optimal chunking sizes and whether to chunk here or above,myabe here? 
    
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
        if iso in da_countrymasks['variable']: # do this in a slightly more intelligent way??? similar to what i was doing b4 with the dataframs, instead of if
        
            # Get cohort sizes of the country
            if da_cohort_size.country.values.size > 1:
                da_smple_cht = da_cohort_size.sel(country=country).sel(time=slice(startyear, endyear)) #.chunk({'time': 10, 'ages': 10})
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
            print('**iso not in mask')
            pass
    
    da_pop_demographics = da_pop_demographics.compute()
    
    return da_pop_demographics






def population_demographics_gridscale_global(
    startyear=2000,
    endyear=2005,
    ssp=2,
    urbanrural=False,
    chunksize=100
):
    """
    Wrapper function to run previous functions choosing isimip round and ssp, for filepaths see component functions. 
    """

    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
    
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    
    with HiddenPrints():
        df_countries_matched = match_country_names_all_mask_frac();

        df_cohort_sizes, ages, years = load_cohort_sizes(ssp=ssp)

        da_population = load_population(ssp=ssp,
                                    startyear=startyear,
                                    endyear=endyear,
                                   urbanrural=urbanrural)

    print('loading country masks')
    da_countrymasks = load_countrymasks_fillcoasts() #.chunk({'lat': chunksize, 'lon': chunksize})

    print('interpolating cohort sizes per country')
    with HiddenPrints():
        da_cohort_size = interpolate_cohortsize_countries(df_cohort_sizes,
                                                 ages,
                                                 years)
    print('calculating gridscale demographics')
    with HiddenPrints():
        da_pop_demographics = get_gridscale_demographics(da_population,
                                                 da_countrymasks,
                                                 df_countries_matched,
                                                 da_cohort_size,
                                                 startyear=startyear,
                                                 endyear=endyear);



    return da_pop_demographics




