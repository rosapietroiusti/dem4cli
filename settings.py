


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


flags = {}

flags['version'] = 1 
                        # v1.0 
                        # v2.0 : new pop data, new cohortsize data, 




flags['GMT_mapping'] = 'year_to_year'
                        # 'year_to_year' = Wim/Luke method
                        # 'STITCHES' = stitches approach to remapping 
                    


script_dir = os.path.abspath( os.path.dirname( __file__ ) )
data_dir = os.path.join(script_dir, 'data')


# if flags version == 1


# elif flags version == 2 

# set paths 


dir_temperature_trajectories = os.path.join(data_dir, 'temperature-trajectories')


dir_climate_data = None # not sure this is necessary here or outside dem4cli


filepaths_population = None

filepath_cohortsizes = None

filepath_countrymask = None

filepath_lifeexpectancy = None



# settings for GMT mapping 

GMT_inc = 0.1
scen_thresholds = {
    '3.0': [2.9,3.0],
    'NDC': [2.35,2.4], # this is not 2.7 it's 2.4 ! Update value? 
    '2.0': [1.95,2.0],
    '1.5': [1.45, 1.5],
}