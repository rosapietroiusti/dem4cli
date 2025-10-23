


# define data directories and paths

# define v1 vs. v2 of dem4cli with the new vs. old data - change this! 


"""

To integrate
- GMST trajectories from Luke - DONE
- new population data from Dominik
- new cohort size data from Dominik 

- how to include old and new versions? all as flags? kind of messy.... or just overwrite and have only v2 ? or make a separate .py file with functions 
- For now work on separate .py file with flags in each fxn - later 
- Later, figure out how to do this in a more object oriented way !! as a self object that I assign all these things to! And the fxns automatically do what they have to - Ask Ali! 


"""

import os, sys, re 
from ._utils import * 




# put these settings in functions!! as arguments ! 

flags = {}

flags['version'] = 2 
                        # v1.0 
                        # v2.0 : new pop data, new cohortsize data, 


flags['pop_resolution'] = 0.1 # 0.1 or 0.5 for v2, only 0.5 for v1 

flags['GMT_mapping'] = 'year_to_year'
                        # 'year_to_year' = Wim/Luke method
                        # 'STITCHES' = stitches approach to remapping - TO DEVELOP
                    

flags['cohort_sizes_source'] = 'UNWPP2024'
                        # 'UNWPP2024'
                        # 'WCDE' (these are SSPs)


flags['countrymask'] = 'shapefile' 
                        # 'shapefile' 
                        # 'fractional_mask' (not fully implemented Lexp) - TO DEVELOP

script_dir = os.path.abspath( os.path.dirname( __file__ ) )
data_dir = os.path.join(script_dir, 'data')


# Data paths for different versions

# pulling filepaths out of functions (mostly) to make it easier to switch between versions

if flags['version'] == 1: 

    dir_population = os.path.join(data_dir, 'gridded-pop/') 
    dir_cohortsizes = os.path.join(data_dir, 'cohort-sizes/WCDE')
    filepath_countrymask = os.path.join(data_dir, 'country-masks/isipedia-countries/preprocessed/countrymasks_fractional_'+float_to_str(flags['pop_resolution'])+'deg_filledcoasts.nc')
    filepath_lifeexpectancy = os.path.join(data_dir, 'life-expectancy/UN_WPP2024/WPP2024_MORT_F05_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx')
    filepath_lookuptable = os.path.join(data_dir, 'country-masks/lookup_table_dem4cli_v1.xlsx' )
                                         

elif flags['version'] == 2:

    dir_population = '/data/brussel/vo/000/bvo00012/data/dataset/COMPASS/v2/population_count/'+float_to_str(flags['pop_resolution'])+'deg' # make a symlink in dem4cli? 
    if flags['cohort_sizes_source'] == 'UNWPP2024':
        dir_cohortsizes = os.path.join(data_dir, 'cohort-sizes/UN_WPP2024')
    elif flags['cohort_sizes_source'] == 'WCDE':
        dir_cohortsizes = os.path.join(data_dir, 'cohort-sizes/WCDE_v3.2.beta')

    if flags['countrymask'] =='shapefile':
        filepath_countrymask = os.path.join(data_dir, 'country-masks/natural_earth/Cultural_10m/Countries/ne_10m_admin_0_countries.shp') # TODO: copy this here and implement this flag! 
    else:
        filepath_countrymask = os.path.join(data_dir, 'country-masks/isipedia-countries/preprocessed/countrymasks_fractional_'+float_to_str(flags['pop_resolution'] )+'deg_filledcoasts.nc')
    filepath_lifeexpectancy = os.path.join(data_dir, 'life-expectancy/UN_WPP2024/WPP2024_MORT_F05_1_LIFE_EXPECTANCY_BY_AGE_BOTH_SEXES.xlsx')
    filepath_lookuptable_original = '/data/brussel/vo/000/bvo00012/data/dataset/COMPASS/v2/'+'cross_reference_SSP3_2_to_ISO.xlsx'
    filepath_lookuptable = data_dir+'/country-masks/lookup_table_dem4cli_v2.csv'


filepath_isimip_countries_meta = os.path.join(data_dir, 'country-masks/isipedia-countries/countryData.json')
filepath_world_bank_meta = os.path.join(data_dir, 'income-groups/world_bank/CLASS.xlsx')


# settings for GMT mapping / stylized trajectory creation
dir_temperature_trajectories = os.path.join(data_dir, 'temperature-trajectories') 
GMT_inc = 0.1
scen_thresholds = { # peak warming between these values
    '3.0': [2.9,3.0],
    'NDC': [2.35,2.4], # this is not 2.7 it's 2.4 ? Update value? 
    '2.0': [1.95,2.0],
    '1.5': [1.45, 1.5],
}
scenarios = ['historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585']



# delete from dem4cli??? 


bbox_europe = [ 31.99,  71.09, -14.96,  34.94]