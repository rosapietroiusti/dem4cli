


# define data directories and paths

# define v1 vs. v2 of dem4cli with the new vs. old data 


"""
To integrate
- GMST trajectories from Luke
- new population data
- new cohort size data

- how to include old and new versions? all as flags? kind of messy.... or just overwrite and have only v2 ? or make a separate .py file with functions
"""

import os, sys, re 
from _utils import * 

flags = {}

flags['version'] = 2 
                        # v1.0 
                        # v2.0 : new pop data, new cohortsize data, 


flags['pop_resolution'] = 0.1 # 0.1 or 0.5 for v2, only 0.5 for v1 

flags['GMT_mapping'] = 'year_to_year'
                        # 'year_to_year' = Wim/Luke method
                        # 'STITCHES' = stitches approach to remapping - TO DEVELOP
                    


script_dir = os.path.abspath( os.path.dirname( __file__ ) )
data_dir = os.path.join(script_dir, 'data')


# Data paths for different versions

if flags['version'] == 1: 

    dir_population = None # paths defined in the functions - maybe change this??
    dir_cohortsizes = None
    filepath_countrymask = None
    filepath_lifeexpectancy = None
    filepath_lookuptable = None # fxn in _match_countries.py

elif flags['version'] == 2:

    dir_population = '/data/brussel/vo/000/bvo00012/data/dataset/COMPASS/v2/'+float_to_str(flags['pop_resolution']) # make a symlink in dem4cli? 
    dir_cohortsizes = os.path.join(data_dir, 'cohort-sizes/WCDE_v3.2.beta')
    filepath_countrymask = '/data/brussel/vo/000/bvo00012/data/dataset/COMPASS/v2/'+'Country_map.nc' # copy or symlink in dem4cli? 
    filepath_lifeexpectancy = os.path.join(data_dir, 'UNWPP2024/WPP2024_MORT_F05_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx')
    filepath_lookuptable = '/data/brussel/vo/000/bvo00012/data/dataset/COMPASS/v2/'+'cross_reference_SSP3_2_to_ISO.xlsx'



# settings for GMT mapping 
dir_temperature_trajectories = os.path.join(data_dir, 'temperature-trajectories') # for stylized trajectory creation


GMT_inc = 0.1
scen_thresholds = {
    '3.0': [2.9,3.0],
    'NDC': [2.35,2.4], # this is not 2.7 it's 2.4 ! Update value? 
    '2.0': [1.95,2.0],
    '1.5': [1.45, 1.5],
}