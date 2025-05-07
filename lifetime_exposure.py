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


def load_unwpp():
    """
    Load UNWPP2024 data on e(x) = Life Expectancy at Exact Age x (ex) - Both Sexes.

    The average number of remaining years of life expected by a hypothetical cohort of individuals alive at age x who would be subject during the remaining of their lives to the mortality rates of a given year. It is expressed as years.

    Keep only Country name and years left to live at age 5 e(5). 

    Source: 
    https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Mortality
    
    """
    
    filepath_unwpp = os.path.join(script_dir, 'data/life-expectancy/UN_WPP2024/WPP2024_MORT_F05_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx')
    
    df_unwpp_raw = pd.read_excel(filepath_unwpp, 
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
                         interp=True):
    
    """
    - Takes UNWPP life expectancy data expressed as years left to live at age of 5, 
    subtracts 5 from Year to get it at birth year but ignoring infant mortality, 
    adds 5 to account for the 5 years of life already lived, adds 6 to account for increase 
    in life expectancy through the life of an individual (i.e. move from "period" life expectancy to 
    "cohort" life expectancy, see Goldstein & Wachter (2006) "Relationships between period and cohort 
    life expectancy: Gaps and lags")
    - Thus get life expectancy in each year for each country at birth 
    expressed in "cohort" way, neglecting infant mortality.
    - Data ends for 2018 cohort (5 y.o. in 2023), extend to 2020 cohort by filling with constant value 

    """
    
    df_life_expectancy_5 = df_unwpp.copy()
    df_life_expectancy_5.index = df_life_expectancy_5.index-5 # year of birth 
    df_life_expectancy_5 = df_life_expectancy_5 + 5 + 6 

    if interp:
        # extend up to 2020 
        df_life_expectancy_5_interp = df_life_expectancy_5.reindex(
        np.arange(1945,2020+1)).astype( # make this more flexible 
        'float').interpolate() # fills last two years constant at 2018 level 
    
        return df_life_expectancy_5_interp
    else:
        return df_life_expectancy_5