


# define data directories and paths

# define v1 vs. v2 of dem4cli with the new vs. old data 


"""
To integrate
- GMST trajectories from Luke
- new population data
- new cohort size data

- how to include old and new versions? all as flags? kind of messy.... or just overwrite and have only v2 ? or make a separate .py file with functions
"""

flags = {}

flags['version'] = 1 
                        # v1.0 
                        # v2.0 : new pop data, new cohortsize data, 




flags['GMT_mapping'] = 'year_to_year'
                        # 'year_to_year' = Wim/Luke method
                        # 'STITCHES' = stitches approach to remapping 
                    


data_dir = pass

dir_climate_data = pass # not sure this is necessary here or outside dem4cli

# if flags version == 1


# elif flags version == 2 

# set paths 




filepaths_population = pass

filepath_cohortsizes = pass

filepath_countrymask = pass

filepath_lifeexpectancy = pass

filepaths_gmtpaths = pass 

