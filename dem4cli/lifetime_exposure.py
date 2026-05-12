"""
Demographics4Climate : Population and demographics for climate science analysis
----------------------------------------------

2026 Update by Quentin Lejeune

Calculate lifetime exposure from RIME-X results.

""" 
#%%


import os
import numpy as np
import xarray as xr

from ._settings import * 

script_dir = os.path.abspath( os.path.dirname( __file__ ) )

#%%

@timeit
def calc_lifetime_exposure_rimex(df_life_expectancy, start_birthyear, end_birthyear, df_ts_quantiles, country):
    
    # 1. Transform the input DataFrame into a dataset with 'quantile', 'year' and 'scenario' as dimensions
    ds_ts_quantiles = (df_ts_quantiles
        .set_index(['quantile', 'year'])
        .stack(level=1)
        .rename_axis(index=['quantile', 'year', 'scenario'])
        .to_xarray()
    )


    # 2. Vectorised lifetime integration
    # Birth years as coordinate
    birth_years = np.arange(start_birthyear, end_birthyear + 1)

    # Life expectancy for all cohorts
    life_exp = xr.DataArray(
        df_life_expectancy.loc[birth_years, country].values, 
        coords={"birth_year": birth_years}, 
        dims="birth_year"
    )

    death_year = birth_years + np.floor(life_exp)
    fraction_lastyr = life_exp - np.floor(life_exp)

    # Expand dataset to include birth_year dimension
    ds_expanded = ds_ts_quantiles.expand_dims(birth_year=birth_years)

    # Broadcast year and birth_year
    year = ds_expanded.year
    birth = ds_expanded.birth_year

    # Compute exposure for fully lived years 
    full_mask = (year >= birth) & (year <= (death_year - 1))
    exposure_fullyrs = ds_expanded.where(full_mask).sum(dim="year")

    # Compute exposure for partially lived years 
    partial_mask = (year == death_year)
    exposure_lastyr = (ds_expanded.where(partial_mask).sum(dim="year") * fraction_lastyr)

    # Total lifetime exposure
    ds_lifetime_exp = exposure_fullyrs + exposure_lastyr


    return ds_lifetime_exp



@timeit
def calc_lifetime_exposure_rimex_with_years_extension(df_life_expectancy, start_birthyear, end_birthyear, df_ts_quantiles, country):
    
    # 1. Transform the input DataFrame into a dataset with 'quantile', 'year' and 'scenario' as dimensions
    ds_ts_quantiles = (df_ts_quantiles
        .set_index(['quantile', 'year'])
        .stack(level=1)
        .rename_axis(index=['quantile', 'year', 'scenario'])
        .to_xarray()
    )


    # 2. Determine maximum death year and maximum year currently in the dataset
    death_years = []
    for birth_yr in range(start_birthyear, end_birthyear + 1):
        life_expectancy = df_life_expectancy.loc[birth_yr, country]
        death_yr = birth_yr + life_expectancy
        death_years.append(death_yr)
    max_death_yr = int(np.floor(max(death_years)))
    current_max_year = ds_ts_quantiles['year'].max().values


    # 3. If needed, extend the data up until the maximum death year
    if max_death_yr > current_max_year:
        # Calculate the mean of the last 10 years for each quantile and column
        last_10_years_mean = ds_ts_quantiles.isel(year=slice(-10, None)).mean(dim='year')

        # Create an extended 'year' coordinate
        extended_years = np.arange(int(ds_ts_quantiles['year'].min()), max_death_yr + 1)

        # Reindex the dataset to the extended years, fill missing values with the meanof the last 10 years
        ds_ts_quantiles = (ds_ts_quantiles.reindex(year=extended_years).fillna(last_10_years_mean))


    # 4. Vectorised lifetime integration
    # Birth years as coordinate
    birth_years = np.arange(start_birthyear, end_birthyear + 1)

    # Life expectancy for all cohorts
    life_exp = xr.DataArray(
        df_life_expectancy.loc[birth_years, country].values, 
        coords={"birth_year": birth_years}, 
        dims="birth_year"
    )

    death_year = birth_years + np.floor(life_exp)
    fraction_lastyr = life_exp - np.floor(life_exp)

    # Expand dataset to include birth_year dimension
    ds_expanded = ds_ts_quantiles.expand_dims(birth_year=birth_years)

    # Broadcast year and birth_year
    year = ds_expanded.year
    birth = ds_expanded.birth_year

    # Compute exposure for fully lived years 
    full_mask = (year >= birth) & (year <= (death_year - 1))
    exposure_fullyrs = ds_expanded.where(full_mask).sum(dim="year")

    # Compute exposure for partially lived years 
    partial_mask = (year == death_year)
    exposure_lastyr = (ds_expanded.where(partial_mask).sum(dim="year") * fraction_lastyr)

    # Total lifetime exposure
    ds_lifetime_exp = exposure_fullyrs + exposure_lastyr


    return ds_lifetime_exp

