# Demographics4Climate

This is a stand-alone module to preprocess demographic data (i.e. population size, cohort size, life expectancy) at yearly resolution at gridscale level and at national level, and compute lifetime exposure from an annual climate dataset that the user can flexibly provide.  

Based on Thiery et al (2021), Grant et al (2025), Vanderkelen et al (in prep), Pietroiusti et al (in prep), Laridon et al (in prep). Updated in 2025 with new available data.

Contact: rosa.pietroiusti@vub.be

> [!WARNING]
> Work in progress: functions to calculate lifetime exposure.

## Data used in version 2

1. **Population cohort sizes** from 1950 to 2100 per country (reconstructions and projections). Dem4cli-v2 currently uses data from Wittgenstein Center, SSPs drivers version 3.2-beta (May 2025 release, not publicly distributed yet, request for access). Cohort size data is available at a country level for the period 1950-2100 (reconstructions up to 2025 and projections thereafter) expressed for 5-year age cohorts at 5-year time snapshots. Dem4cli supports running the module for this data for SSP1, SSP2, SSP3.
2. **Gridded population data** reconstructions and projections for SSP1, SSP2 and SSP3. Dem4cli-v2 currently uses data from the COMPASS project (credit: Dominik Paprotny, [documentation here](https://compass-climate.eu/Public%20Deliverables/D3.1_Exposure%20datasets%20at%20multiple%20scales.pdf)), for the period 1950-2100, reconstructions until 2025 and projections thereafter, harmonized with SSP version 3.2-beta national totals for projections. Not publicly distributed yet, request for access. 
3. **Life expectancy data** from UNWPP2024 expressed as years left to live at the age of 5 (ex) (UNWPP2024, https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Mortality)
3. **Isipedia fractional country masks** (Perrette 2023, https://github.com/ISI-MIP/isipedia-countries). 
4. **Metadata on income levels and regions** from World Bank (WB 2023, https://datatopics.worldbank.org/world-development-indicators/the-world-by-income-and-region.html)

### Data availability 

Data necessary to run dem4cli-v1 is available in a zenodo repository: https://zenodo.org/records/15425666 (access by request). 

To run dem4cli, you can include the 'data' folder in the same folder as the 'population_demographics.py' script

```
<SCRIPT_DIR>/data/
```

> [!WARNING]
> Work in progress: Preparing data to run dem4cli-v2 is in progress. 


## What this module does 

You can set your settings in _settings.py


```
flags = {}

flags['version'] = 2 
                                    # v1 
                                    # v2 : new gridded population and cohortsize data 


flags['pop_resolution'] = 0.5       # 0.1 or 0.5 degrees (regular grid) for v2, only 0.5 degrees for v1 

```

### Part 1: Demographic Data Preprocessing at country-level

WCDE cohort size estimates are linearly interpolated from age-brackets to exact ages, correcting such that the mean is preserved, and then linearly interpolated from snapshots every 5 years to yearly values, so that you have a cohort size value for each exact age each year. 

UNWPP2024 data is turned from life expectancy expressed as years left to live at the age of 5 (e(x)) into  life expectancy at birth, neglecting child mortality, by subtracting 5 from the birth year. Period life expectancy is turned into cohort life expectancy, by adding 6 to the life expectancy value based on the lags theory in Goldstein & Wachter (2006) "Relationships between period and cohort life expectancy: Gaps and lags". Life expectancy data is then interpolated linearly to get it for each exact year instead of every 5 years (note: this is currently not corrected to remain mean-preserving).

Gridded population data, country masks and country metadata are opened and all objects are filtered to be obtained for matching countries. 

You can run this as, e.g.:

```
from population_demographics_v2 import * 

d_countries = preprocess_all_country_data(

    dir_cohortsizes = dir_cohortsizes,                  # cohort size data
    ssp=2, 
                                            
    filepath_lifeexpectancy = filepath_lifeexpectancy,  # life expectancy data
    start_birthyear=1950,
    end_birthyear=2025, 

    dir_population= dir_population,                     # gridded pop data 
    startyear=1950,
    endyear=2100,
    bbox = None,                                        # option to provide a bounding box

    filepath_countrymask = filepath_countrymask,        # country masks 
    
    filepath_lookuptable = filepath_lookuptable,        # country list filtering
    filter_countries=True,
    worldbank_filter=True, 

    )
```

This returns a dictionary that can be unpacked e.g. as 

```

df_countries = d_countries['info_pop']
da_countrymasks = d_countries['borders'] 
da_regions = df_countries['region'].unique()
da_population = d_countries['population_map']
df_life_expectancy_5 = d_countries['life_expectancy_5']
da_cohort_size = d_countries['cohort_size']

```


 ### Part 2: Lifetime exposure
 
> [!WARNING]
> Work in progress




### Part 3: Gridscale Demographics (currently only tested for dem4cli-v1)

Using the fractional country masks the proportion of cohort size in each country each year is applied to the gridded population of that country, assuming the cohort proportions are constant across the country. The population totals from the gridded population data are thus conserved (with ~0.03-0.05% of population lost due to mismatch between the countries covered by WCDE and those available in fractional country masks). 

Option to output separate variables for urban, rural and total population.

You can run this as, e.g.:

```
from population_demographics import * 

da_pop_demographics_ssp3 = population_demographics_gridscale_global(startyear=2000,
                                                                    endyear=2003,
                                                                    ssp=3,
                                                                    urbanrural=False) 
```

> [!WARNING]
> Testing for v2 is work in progress