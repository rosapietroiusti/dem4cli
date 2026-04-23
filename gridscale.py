

# note tested only for v1 ! 





def population_demographics_gridscale_global(
    startyear=2000,
    endyear=2005,
    ssp=2,
    urbanrural=False,
    #chunksize=100
):
    """
    Wrapper function to run previous functions choosing isimip round and ssp, for filepaths see component functions. 
    """

    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
    
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    
    with HiddenPrints():
        df_countries_matched = match_country_names_all_mask_frac()

        df_cohort_sizes, ages, years = load_cohort_sizes(ssp=ssp)

        da_population = load_population(ssp=ssp,
                                    startyear=startyear,
                                    endyear=endyear,
                                   urbanrural=urbanrural)

    print('loading country masks')
    da_countrymasks = load_countrymasks_fillcoasts() #.chunk({'lat': chunksize, 'lon': chunksize})

    print('interpolating cohort sizes per country')
    with HiddenPrints():
        da_cohort_size = interpolate_cohortsize_countries(df_cohort_sizes,
                                                 ages,
                                                 years)
    print('calculating gridscale demographics')
    with HiddenPrints():
        da_pop_demographics = get_gridscale_demographics(da_population,
                                                 da_countrymasks,
                                                 df_countries_matched,
                                                 da_cohort_size,
                                                 startyear=startyear,
                                                 endyear=endyear)



    return da_pop_demographics