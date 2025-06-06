# Demographics4Climate

This is a stand-alone module to calculate population demographics, i.e. population size, cohort size, life expectancy at yearly resolution at gridscale level.

Based on Thiery et al (2021), Grant et al (in review), Vanderkelen et al (in prep), Pietroiusti et al (in prep). Updated in 2024 with new available data, and now possible to run for 1950-2100 under SSP1, SSP2 or SSP3. 

Contact: rosa.pietroiusti@vub.be

> [!WARNING]
> Work in progress: functions to calculate lifetime exposure, at country and gridscale level, taken from Grant et al (2024, in rev)

## Data used


1. Wittgenstein Center Data Explorer population cohort size from 1950 to 2100 per country (at snapshots every 5 years) (http://dataexplorer.wittgensteincentre.org/wcde-v2/), available for SSP1, SSP2, SSP3. Cohort size data from WCDE is available at a country level for the period 1950-2100 (reconstructions up to 2015 and projections thereafter) expressed for 5-year age cohorts at 5-year time snapshots.
2. ISIMIP gridded population data reconstructions and projections for SSP1, SSP2 and SSP3 from ISIMIP3a/b (histsoc up to 2021 based on HYDE v3.3 
Klein Goldewijk et al.2022), from 2022 SSP projections (based on Gao et al. 2020 https://doi.org/10.5065/D60Z721H and https://doi.org/10.7927/q7z9-9r69), scaled to match ISIMIP national population projections under different SSPs. Projections are based on the national SSP scenarios from Lutz et al. (55) and gridded population projections from the National Center for Atmospheric Research (NCAR).
3. Isipedia fractional country masks are used to match the datasets (Perrette 2023, https://github.com/ISI-MIP/isipedia-countries). 
4. Metadata on income levels and regions from World Bank (WB 2023, https://datatopics.worldbank.org/world-development-indicators/the-world-by-income-and-region.html)

### Data availability 

Data necessary to run dem4cli is available in a zenodo repository: https://zenodo.org/records/15425666 (access by request). 

To run dem4cli, you can include the 'data' folder in the same folder as the 'population_demographics.py' script

```
<SCRIPT_DIR>/data/
```


## What this module does 

### Part 1: Gridscale Demographics 

WCDE cohort size estimates are linearly interpolated from age-brackets to exact ages, correcting such that the mean is preserved, and then linearly interpolated from snapshots every 5 years to yearly values, so that you have a cohort size value for each exact age each year. Then, using the fractional country masks the proportion of cohort size in each country each year is applied to the gridded population of that country, assuming the cohort proportions are constant across the country. The population totals from the gridded population data are thus conserved (with ~0.03-0.05% of population lost due to mismatch between the countries covered by WCDE and those available in fractional country masks). 

Option to output separate variables for urban, rural and total population.

You can run this as, e.g.:

```
from population_demographics import * 

da_pop_demographics_ssp3 = population_demographics_gridscale_global(startyear=2000,
                                                                    endyear=2003,
                                                                    ssp=3,
                                                                    urbanrural=False) 
```


 ### Part 2: Lifetime exposure
 
> [!WARNING]
> 2024 update is work in progress, updating to UNWPP2024:
> - turns UNWPP2019 from life expectancy expressed as years left to live at the age of 5 (ex) into  life expectancy at birth, neglecting child mortality, by subtracting 5 from the central year of the estimate
>  - turns the period life expectancy into cohort life expectancy, by adding 6 to the life expectancy value based on the lags theory in Goldstein & Wachter (2006) "Relationships between period and cohort life expectancy: Gaps and lags"
>  - interpolates linearly the life expectancy data to get it for each exact year instead of every 5 years (note: this is not corrected to remain mean-preserving).
>  - Old data: UN World Population Prospects 2019 life expectancy data at 5 years old from 1950 to 2020 per country (as average in 5-year brackets) (https://population.un.org/wpp/Download/Standard/Mortality/)

