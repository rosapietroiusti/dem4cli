"""
Some key functions to calculate lifetime exposure
----------------------------------------------------

main functions that call other functions:
- calc_lifetime_exposure() : calculates country-scale average per-person exposure per each age category


based on Grant et al 2023, making some edits to simplify:
- removed GMT-mapping
- outputting dfs for single runs/simulations instead of dictionaries with each simulation as an entry in the dictionary
- removed everything to do with PIC

rosa.pietroiusti@vub.be


Flags:ISSUE, QUESTION, TODO
- can we remove countries_regions ? its only used as an index 
- calc_cohort_lifetime_exposure, understand better where the actual cohort lifetime exposure is calc'd

Functions in original script and included Y/N:

1) exposure.py
- lreg, vectorize_linreg, resample - put in utils
- get_countries_of_region - N
- calc-exposure_mmm_xr - N
- calc_exposure_mmm_pic_xr - N
- calc_exposure_trends - N
- calc_weighted_fldmean - Y 
- calc_life_exposure - Y
- calc_lifetime_exposure - Y
- calc_cohort_lifetime_exposure - Y (but to check better what it does)
- calc_lifetime_exposure_pic - N 

2) emergence.py 
- calc_birthyear_align - Y (understand better)



For more functions and information see Luke Grant's original scripts. 

"""

import pandas as pd 
import numpy as np 
import xarray as xr

# --------------------------------------------------------------------------
# 1. Country-scale exposure calculations (from exposure.py)
# --------------------------------------------------------------------------

def calc_weighted_fldmean(
    da, # data you are averaging
    weights, # weights you are using to average
    countries_mask, # regionmask object of countries
    ind_country, # indices
):
    """
    calculate weighted average of a climate dataset, per country 
    e..g weights could be a population map or an area map, to calculate an area-weighted average or a population weighted average 
    """
    if isinstance(ind_country, list):
        if len(ind_country) > 1:
            # more than one country
            mask = xr.DataArray(
                np.in1d(countries_mask,ind_country).reshape(countries_mask.shape),
                dims=countries_mask.dims,
                coords=countries_mask.coords,
            )
            da_masked = da.where(mask)
        else:
            pass 
    elif len([ind_country]) == 1:
        # one country
        da_masked = da.where(countries_mask == ind_country)
    
    da_weighted_fldmean = da_masked.weighted(weights).mean(dim=("lat", "lon"))
    
    return da_weighted_fldmean


def calc_life_exposure(
    df_exposure, # rows are years, columns are countries, fields are field-average 
    df_life_expectancy, # rows are years, columns are countries 
    col, # what column of the dfs you are doing this on (country name)
):
    """
    integrate exposure over an individual's lifetime 
    """

    # initialise birth years 
    exposure_birthyears_percountry = np.empty(len(df_life_expectancy))

    for i, birth_year in enumerate(df_life_expectancy.index):

        life_expectancy = df_life_expectancy.loc[birth_year,col] 

        # define death year based on life expectancy
        death_year = birth_year + np.floor(life_expectancy)

        # integrate exposure over full years lived
        exposure_birthyears_percountry[i] = df_exposure.loc[birth_year:death_year,col].sum()

        # add exposure during last (partial) year (QUESTION: what is going on here????)
        exposure_birthyears_percountry[i] = exposure_birthyears_percountry[i] + \
            df_exposure.loc[death_year+1,col].sum() * \
                (life_expectancy - np.floor(life_expectancy))

    # a series for each column to somehow group into a dataframe
    exposure_birthyears_percountry = pd.Series(
        exposure_birthyears_percountry, # accumulated exposure over lifetime 
        index=df_life_expectancy.index, # year of birth 
        name=col, # country name 
    )

    return exposure_birthyears_percountry

#%% ----------------------------------------------------------------
# main function n. 1 
# proposal to rename: calc_average_lifetime_exposure()
def calc_lifetime_exposure(
    da_AFA, 
    df_countries, 
    countries_regions, # can we delete this ??
    countries_mask, 
    da_population, 
    df_life_expectancy_5, 
):
    """
    convert yearly climate exposure data (e.g. Area Fraction Affected (AFA)) to average per-country number of extremes affecting one individual across life span. First calculates population-weighted average per country of AFA, then accumulates this over lifetime of individual in that country of each age group i.e. assumes equal population distribution throughout the country (not gridscale).
    Note, this version does not do GMT-remapping. 
    
    Inputs
        da_AFA (da): climate data, annual
        df_countries (df): dataframe with country names 
        countries_regions, # why do we need this ? I think delete and use df_countries.columns instead???
        countries_mask (regionmask object): to mask the climate data
        da_population (da): da of population to do a population-weighted average
        df_life_expectancy_5 (df): life expectancy at birth, rows are year of birth, columns are countries 
    
    Returns
        df_le (df): accumulated lifetime exposure (sum of da_AFA during lifetime), rows are birth years, columns are countries, 
                    average for country (units: number of events)
    
    """
    # initialise dicts
    d_exposure_peryear_percountry = {}

    # get spatial average per country
    for j, country in enumerate(df_countries['name']):

        print('processing country '+str(j+1)+' of '+str(len(df_countries)), end='\r')

        # calculate mean per country weighted by population
        ind_country = countries_regions.map_keys(country)

        # historical + RCP simulations - average over the country, per year, weighted by where the population lives??
        d_exposure_peryear_percountry[country] = calc_weighted_fldmean( 
            da_AFA, # da, area affected 
            da_population, # weights (isimip gridded population array) 
            countries_mask, # mask (regionmask object)
            ind_country, # ind_country (index) - dont think we need this ? 
        )

    # convert dict to dataframe for vectorizing and integrate exposures       
    frame = {k:v.values for k,v in d_exposure_peryear_percountry.items()}
    df_exposure = pd.DataFrame(frame,index=d_exposure_peryear_percountry[country].time.values)  # ISSUE prev: year_range, but then this is defined somewhere else          

    df_le = df_exposure.apply(
        lambda col: calc_life_exposure(
            df_exposure, # reindex here if you want to do GMT remapping 
            df_life_expectancy_5,
            col.name,
        ),
        axis=0,
    )

    return df_le
        

def calc_cohort_exposure_peryear( # original name: calc_cohort_lifetime_exposure
    da_AFA,
    df_countries,
    countries_regions, # delete?
    countries_mask,
    da_population,
    da_cohort_size
                                 ):  
    """
    Multiplies population-weighted avereage annual exposure by cohort size each year in each country. i.e. computes 
    every year how many people of each age group have experienced an event, on average (in unit 000's of people of people)
    
    original name: calc_cohort_lifetime_exposure - but no lifetime accumulation is happening here ! its just multiplication 
    
    Inputs
        da_AFA (da): climate data, annual occurence data 
        df_countries (df): dataframe with country names 
        countries_regions, # why do we need this ? I think delete and use df_countries.columns instead???
        countries_mask (regionmask object): to mask the climate data
        da_population (da): da of population to do a population-weighted average
        df_life_expectancy_5 (df): life expectancy at birth, rows are year of birth, columns are countries 
    
    Returns
        da_exposure_cohort (da):  per age cohort, per country number of people experiencing events each year (unit: 000's people)
    
    """
    # initialise dicts
    d_exposure_peryear_percountry = {}

    # get spatial average per country per year, weighted by population that year
    for j, country in enumerate(df_countries['name']):
        print('processing country '+str(j+1)+' of '+str(len(df_countries)), end='\r')

        # calculate mean per country weighted by population
        ind_country = countries_regions.map_keys(country)
        d_exposure_peryear_percountry[country] = calc_weighted_fldmean( 
            da_AFA,
            da_population, 
            countries_mask, 
            ind_country, 
        )
    # convert dictionary to data array
    da_exposure_peryear_percountry = xr.DataArray(
        list(d_exposure_peryear_percountry.values()),
        coords={
            'country': ('country', list(d_exposure_peryear_percountry.keys())),
            'time': ('time', da_AFA.time.values),
        },
        dims=[
            'country',
            'time',
        ],
    )
    
    # note. apply GMT mapping here if doing ! 
    
    # multiply average exposure each year by cohort size of each age each year 
    da_exposure_cohort = da_exposure_peryear_percountry * da_cohort_size 
    
    # note. luke was making da_exposure_peryear_perage_percountry_strj (the same as da_exposure_cohort but with gmt mapping), 
    # and da_exposure_peryear_perage_percountry_strj which was just a dimensional expansion of d_exposure_peryear_percountry
    # to have the _perage column (but was multiplied by xr.full_like(da_cohort_size,1)
    da_exposure_peryear_perage_percountry = da_exposure_peryear_percountry * xr.full_like(da_cohort_size,1) # a dimensional expansion!! 
    # could also just export da_exposure_peryear_percountry and do this dimensional expansion later wherever it is that you need it ! 
    
    return da_exposure_cohort, da_exposure_peryear_perage_percountry
    
# --------------------------------------------------------------------------
# script emergence.py 
# Not sure if these are useful/necessary! Check with Luke !!
# --------------------------------------------------------------------------

def calc_birthyear_align(
    da,
    df_life_expectancy,
    #by_emergence,
    future_births=False
):
    """
    Aligns population data by birth year and age in order to track people across time. 
    
    Later function (where is this?):
    Calculates number of people in each country, born each year (starting from 1960, usually) that are alive any given year up to 2113.
    I.e. each year, people are entering via migration and birth and exiting via migration and death, and you need to separate the newborns
    from those that were already alive. 
    
    Inputs
        da (da): da_cohort_size, the number of people of each age alive each year from WCDE (av. 1950-2100)
        df_life_expectancy (df): life expectancy of each person each year up to those born in 2020 
        future_births (flag): if True calculates for all those born between 1960 and 2100, holding life expectancy constant 

    Returns
        da_all (da):  per birth year, per year, number of people alive aligned (not yet separated) 
    
    """
    
    if future_births == True: # maybe add this as an argument in the function instead of inside the fxn 
        by_emergence = np.arange(1960,2101)
    else:
        by_emergence = birth_years # 1960 to 2020 
    
    country_list = []
    
    # loop through countries
    for country in da.country.values:
        
        birthyear_list = []
        
        # per birth year, make (year,age) selections
        for by in by_emergence:
            
            # use life expectancy information where available (until 2020)
            if by <= year_ref:            
                
                death_year = by + np.ceil(df_life_expectancy.loc[by,country]) # since we don't have AFA, best to round life expec up and then multiply last year of exposure by fraction of final year lived
                time = xr.DataArray(np.arange(by,death_year+1),dims='cohort')
                ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                data = da.sel(country=country,time=time,ages=ages) # paired selections
                data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,death_year+1,dtype='int')})
                data = data.reindex({'time':year_range}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                data = data.assign_coords({'birth_year':by}).drop_vars('ages')
                data.loc[{'time':death_year}] = data.loc[{'time':death_year}] * (df_life_expectancy.loc[by,country] - np.floor(df_life_expectancy.loc[by,country]))
                birthyear_list.append(data)
            
            # after 2020, assume constant life expectancy    
            elif by > year_ref and by < year_end:
                
                death_year = by + np.ceil(df_life_expectancy.loc[year_ref,country]) #for years after 2020, just take 2020 life expectancy
                
                # if lifespan not encompassed by 2113, set death to 2113
                if death_year > year_end:
                    
                    death_year = year_end
                
                time = xr.DataArray(np.arange(by,death_year+1),dims='cohort')
                ages = xr.DataArray(np.arange(0,len(time)),dims='cohort')
                data = da.sel(country=country,time=time,ages=ages) # paired selections
                data = data.rename({'cohort':'time'}).assign_coords({'time':np.arange(by,death_year+1,dtype='int')})
                data = data.reindex({'time':year_range}).squeeze() # reindex so that birth year cohort span exists between 1960-2213 (e.g. 1970 birth year has 10 years of nans before data starts, and nans after death year)
                data = data.assign_coords({'birth_year':by}).drop_vars('ages')
                data.loc[{'time':death_year}] = data.loc[{'time':death_year}] * (df_life_expectancy.loc[year_ref,country] - np.floor(df_life_expectancy.loc[year_ref,country]))
                birthyear_list.append(data)
            
            # for 2100, use single year of exposure    
            elif by == 2100:
                
                time = xr.DataArray([2100],dims='cohort')
                ages = xr.DataArray([0],dims='cohort')
                data = da.sel(country=country,time=time,ages=ages)
                data = data.rename({'cohort':'time'}).assign_coords({'time':[year_end]})
                data = data.reindex({'time':year_range}).squeeze()
                data = data.assign_coords({'birth_year':by}).drop_vars('ages')
                birthyear_list.append(data)            
        
        da_data = xr.concat(birthyear_list,dim='birth_year')
        country_list.append(da_data)
        
    da_all = xr.concat(country_list,dim='country')
    
    return da_all # rename this ? 


    
    
    
# create dataset out of birthyear aligned cohort sizes
def ds_cohort_align(
    da,
    da_aligned,
):
    """
    based on previous output calculates different things including total population, per cohort size... and weights 
    
    Inputs
        da (da): da_cohort_size, the number of people of each age alive each year from WCDE (av. 1950-2100)
        da_aligned (da): output of calc_birthyear_align # shall we just include that function inside this one? 

    Returns
        ds_cohort_sizes (ds) :  dataset with coords: country, birth_year, time, ages, different variables:
                                    1. t_population: (=da_t) total population in country each year
                                    2. population (=da_aligned/da_cohort_aligned) : tracking the birth cohort size each year, 
                                    how many people of each birth cohort there are each year 
                                    3. by_population_y0: size of 1960 cohort in 1960, size of 1961 cohort in 1961... (=da_by_p0)
                                    4. by_population: sum of the people in a given cohort each year (i.e. how many people in 1960 b.c. in 1960 + in 1961 etc..) 
                                    double counts over lifetime (=da_by) one number per birth cohort! 
                                    7. t_weights: total population in country that year divided by total population in world that year 
                                    6. by_y0_weights: same thing but just taking the cohort size in year 0 
                                    5. by_weights: ratio between by_population of a certain birth cohort (i.e. sum of that cohort's size over every year), 
                                    and the sum of that birth cohort's multi temporal sum over all countries
    
    """
    da_t = da.sum(dim='ages') # sum population in country over time (i.e. sum over all ages, get population each year in country)
    da_by = da_aligned.sum(dim='time') # sum population over birth years (with duplicate counting as we sum across birth cohorts lifespan) 
    # i.e. sum of the number of people in a given cohort each year 
    # population over birth years represented by first year of lifespan
    
    da_times=xr.DataArray(da_aligned.birth_year.data,dims='birth_year') # birth years 
    da_birth_years=xr.DataArray(da_aligned.birth_year.data,dims='birth_year') # birth years 
    da_by_y0 = da_aligned.sel(time=da_times,birth_year=da_birth_years) # select the cohort size in the year of birth (i.e. size of 1960 cohort in 1960 etc). 
    
    ds_cohort_sizes = xr.Dataset(
        data_vars={
            't_population': (da_t.dims,da_t.data), # population per timestep across countries for t_weights
            'population': (da_aligned.dims,da_aligned.data), # population per birth year distributed over time for emergence mask
            'by_population_y0': (da_by_y0.dims,da_by_y0.data), # size of each cohort in its y0 
            'by_population': (da_by.dims,da_by.data), # see above, da_by, for each birth year, get total number of people for by_weights (population in ds; duplicate counting risk with time sum)
        },
        coords={
            'country': ('country',da_aligned.country.data),
            'birth_year': ('birth_year',da_aligned.birth_year.data),
            'time': ('time',da_aligned.time.data),
            'ages': ('ages',da.ages.data)
        },
    )
    # calculate weights
    ds_cohort_sizes['t_weights'] = (ds_cohort_sizes['t_population'] / ds_cohort_sizes['t_population'].sum(dim='country')) # add cohort weights to dataset   
    ds_cohort_sizes['by_y0_weights'] = (ds_cohort_sizes['by_population_y0'] / ds_cohort_sizes['by_population_y0'].sum(dim='country')) # add cohort weights to dataset
    ds_cohort_sizes['by_weights'] = (ds_cohort_sizes['by_population'] / ds_cohort_sizes['by_population'].sum(dim='country')) # add cohort weights to dataset
    
    return ds_cohort_sizes


def ds_exposure_align( # rename this !! 
    da,
):
        
    da_cumsum = da.cumsum(dim='time') #.where(da>0) # why are you doing where da >0 ??? 
    ds_exposure_cohort = xr.Dataset(
        data_vars={
            'exposure': (da.dims, da.data),
            'exposure_cumulative': (da_cumsum.dims, da_cumsum.data)
        },
        coords={
            'country': ('country', da.country.data),
            'birth_year': ('birth_year', da.birth_year.data),
            'time': ('time', da.time.data),
        },
    )
     
    return ds_exposure_cohort # rename this ?? call it cumsum or something ??

    
    
    

# ----------------------------------------------------------------------------------------------------- 
# 2. Gridscale population exposure calculations
# ----------------------------------------------------------------------------------------------------- 
