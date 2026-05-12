# Demographics4Climate

dem4cli is a stand-alone module to preprocess demographic data (life expectancy, population size, cohort size data) and compute age-specific exposure, or lifetime exposure over stylized trajectories on user-defined climate hazard datasets.

Contact: rosa.pietroiusti@vub.be


## Install dem4cli 

A development install of package and dependencies can be done after cloning the repo, in pip-editable -e mode (that way code edits will propagate without the need for re-installing)

```
git clone https://github.com/rosapietroiusti/dem4cli.git
cd dem4cli
pip install -e .
```

This will clone the main branch. To clone a specific branch (e.g. coupling_rime-x_dev): 

```
git clone -b coupling_rime-x_dev --single-branch https://github.com/rosapietroiusti/dem4cli.git
cd dem4cli
pip install -e .
```

Alternatively (recommended) you can create a conda environment after you have cloned the repository and install dependencies there. 

```
conda create -n dem4cli-env python=3.10
conda activate dem4cli-env
cd dem4cli
pip install -e .
```
## Run dem4cli

You can import all functions or only some functions of the package, e.g.:

```
from dem4cli import *
```

To import only functions to preprocess life expectancies (e.g. if running coupled with another emulator):

```
from dem4cli.population_demographics import load_unwpp_lifeexpectancy, get_life_expectancies
```


## Input data description

1. **Life expectancy data** from UNWPP2024: https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Mortality. 
2. **Gridded population data** reconstructions and projections. *dem4cli* uses data from the COMPASS project (received from Dominik Paprotny, [documentation here](https://compass-climate.eu/Public%20Deliverables/D3.1_Exposure%20datasets%20at%20multiple%20scales.pdf)), for the period 1950-2100, reconstructions until 2025 and projections thereafter, harmonized with SSP version 3.2-beta national totals for projections. Available at 0.1 or 0.5 degrees for SSP1, SSP2 and SSP3 as ancillary package data. 
3. **Cohort sizes** reconstructions and projections at country level from 1950 to 2100. 
    1) Option 1: UNWPP2024 cohort size reconstructions until 2023 and projections thereafter at single year and single age intervals. The mediuim variant best estimate is used in *dem4cli*, this is roughly similar to SSP2 fertility projections. 
    2) Option 2: Wittgenstein Center, SSPs drivers version 3.2-beta (May 2025 release, not publicly distributed yet, request for data access). Data is available as reconstructions up to 2025 and projections thereafter, expressed for 5-year age cohorts at 5-year time snapshots. *dem4cli* supports using this data for SSP1, SSP2 or SSP3.
4. **Country masks** 
    1) Country shapefiles: from naturalearth.
    2) Subnational shapefiles: ancillary package data contains shapefiles at NUTS2 and NUTS3 level, for Europe, from Eurostat. 
    3) Fractional gridded country masks: from ISIpedia, (Perrette 2023, https://github.com/ISI-MIP/isipedia-countries). 
5. **Metadata on income levels and world regions** from the World Bank 2023: https://datatopics.worldbank.org/world-development-indicators/the-world-by-income-and-region.html. 

### Data availability 

All data necessary to run dem4cli is available in a zenodo repository: https://zenodo.org/records/15425666 (access by request). 

To run dem4cli, you can include the 'data' folder in the same folder containing the 'pyproject.toml' file:

```
dem4cli/data/
```

Data should have the following folder structure: 

```
├── data/                     
│   ├── cohort-sizes/
│   ├── country-masks/
│   ├── gmst-models/
│   ├── gmst-obs/
│   ├── gridded-pop/
│   ├── income-groups/
│   ├── life-expectancy/
│   └── temperature-trajectories/
```

## Settings

You can set your settings in _settings.py


```
flags = {}

flags['version'] = 2 
                                    # v1 
                                    # v2 : new gridded population and cohortsize data 


flags['pop_resolution'] = 0.5       # 0.1 or 0.5 degrees (regular grid) for v2, only 0.5 degrees for v1 

```


### Part 1: Demographic data preprocessing 

UNWPP2024 life expectancy data is turned from life expectancy expressed as years left to live at the age of 5 (e(x)) into  life expectancy at birth, neglecting child mortality, by subtracting 5 from the calendar year and adding 5 years to the life expectancy. Period life expectancy is turned into cohort life expectancy, by adding 6 to the life expectancy value based on the lags theory in Goldstein & Wachter (2006) "Relationships between period and cohort life expectancy: Gaps and lags".

If using cohort sizes from WCDE, estimates are linearly interpolated from 5-year age-brackets to exact ages, correcting such that the mean is preserved, and then linearly interpolated from snapshots every 5 years to yearly values, so that you have a cohort size value for each exact age each year. If using cohort sizes from UNWPP2024 this is not necessary as data is provided for exact ages/years. 

Gridded population data, country masks and country metadata are opened and all objects are filtered and harmonized to be obtained for matching countries and on the same grid. Optionally, all demographic objects can be filtered/cropped based on a bounding box. 

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

This returns a dictionary that can be unpacked as 

```

df_countries = d_countries['info_pop']
da_countrymasks = d_countries['borders'] 
da_regions = df_countries['region'].unique()
da_population = d_countries['population_map']
df_life_expectancy_5 = d_countries['life_expectancy_5']
da_cohort_size = d_countries['cohort_size']

```


 ### Part 2: Land fraction exposed & Lifetime exposure - emulated pathways
 
Users can flexibly load preprocessed annual gridded climate hazard data. This data can be binary (yes/no hazard occurrence during the year) or can represent the number of exceedances of a threshold per year. 

The location of this data should be in [XX LOCATION], with the folder structure matching [XX FOLDER STRUCTURE]

*dem4cli* can be used to compute 




### Part 3: Gridscale Demographics

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


## References

*Dem4cli* is described in Pietroiusti et al. (2026, in review, preprint: https://eartharxiv.org/repository/view/12831/).

*Dem4cli* is based on Thiery et al (2021 https://www.science.org/doi/10.1126/science.abi7339), Grant et al (2025 https://www.nature.com/articles/s41586-025-08907-1), Vanderkelen et al (2026, in review), Pietroiusti et al. (2026, in review), Laridon et al (2026, in prep). 
