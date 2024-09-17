"""
Demographics4Climate : Population and demographics for climate science analysis
----------------------------------------------

2024 Update

Calculate gridscale demographics based on 
- gridded population from ISIMIP2/ISIMIP3
- cohort sizes from WCDE
- metadata on countries from ISIMIP2/3, WCDE, isipedia, world bank... 

To do
Code slightly to clean up
test for isimip2 and 3 
make final wrapper function
remove geojson if not necessary and edit match countrynames to use mask instead 

""" 

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd # can maybe delete if i dont open geojson in the end 
from scipy import interpolate
import glob, os, re, sys
import warnings
import openpyxl 





def load_country_metadata(
    filepath_isimip_countries = './data/country-masks/isipedia-countries/countryData.json',
    filepath_world_bank = './data/income-groups/world_bank/CLASS.xlsx',
    keep_names='isimip',
    keep_stats=False,

):
    """
    load country list from isipedia-coutries (country masks metadata files from Perette 2023, https://github.com/ISI-MIP/isipedia-countries). For 195 official/observer UN countries. 
    and metadata from worldbank.

    Input
        keep_names (str) what country names to keep, can be 'isimip', 'world_bank', 'both'  
        filepath_isimip_countries
        filepath_world_bank
        keep stats (Bool) from isimip_countries 
    
    Returns
        df_metadata: table with country name, ISO3 code, country code, region and income group, where available
    
    """

    # open isimip metadata  
    df_isimip_metadata = pd.read_json(filepath_isimip_countries).replace(-9999, np.nan)
    # open world bank metadata
    df_wb_countries = pd.read_excel(filepath_world_bank, sheet_name=0)
    # merge keep list of countries from isimip and info from world bank
    df_merge = df_isimip_metadata.merge(df_wb_countries, how='inner',left_on='country_iso3', right_on='Code')

    # keep only some of the info and clean up column names 
    if keep_names =='isimip':
        keep_cols = ['country', 'Code', 'country_code','Region', 'Income group']
        d_rename = {'Code':'country_iso3', 'Region':'region', 'Income group': 'income_group'}
        
    elif keep_names == 'world_bank':
        keep_cols =['Economy', 'Code', 'country_code','Region', 'Income group']
        d_rename={'Economy':'country','Code':'country_iso3', 'Region':'region', 'Income group': 'income_group'}
        
    elif keep_names == 'both':
        keep_cols=['country','Economy', 'Code', 'country_code','Region', 'Income group']
        d_rename={'Economy':'country_wb','Code':'country_iso3', 'Region':'region', 'Income group': 'income_group'}     

    if keep_stats == True:
        keep_cols=keep_cols+list(df_isimip_metadata.columns[3:])

    df_metadata = df_merge[keep_cols].rename(columns=d_rename) #.head(196) # 'Economy', 
        
    return df_metadata






# COULD DELETE THIS ! 
def load_country_stats(
    filepath_isimip_stats = './data/country-masks/isipedia-countries/countryprofiledata.json'
                      ):
    """
    Load statistics for 195 official/observer UN countries from isipedia-countries. 
    """

    df_isimip_stats = pd.read_json(filepath_isimip_stats).T.reset_index(drop=True).replace(-9999, np.nan).rename(columns={'iso3':'country_iso3'})

    return df_isimip_stats







def load_cohort_sizes( 
    filepaths_wcde = ['./data/cohort-sizes/WCDE/wicdf_ssp1.csv', 
                      './data/cohort-sizes/WCDE/wicdf_ssp2.csv', 
                      './data/cohort-sizes/WCDE/wicdf_ssp3.csv'],
                      ssp = 2,
                      by_sex = False
):
    """
    load population size per age cohort from Wittgenstein Center Data Explorer (source: http://dataexplorer.wittgensteincentre.org/wcde-v2/)

    data description: Population Size (000's)
    De facto population in a country or region, classified by sex and by five-year age groups. Available in all scenarios and at all geographical scales. For each country data is sorted first by age cohort (0-4, 4-9...). So all the first data refers to the 0-4 age cohort. 
    Then they give the population size of that cohort at a snapshot every 5 years (1950, 1955, 1960...).
    Here we assign the data to the central age cohort (i.e. 0-4 assigned to 2).
    
    Input
        filepaths_wcde (str): path to csv files for different ssps
        sel_ssp (int): 1,2 or 3 for ssp1, ssp2, ssp3
        by_sex (Bool): TODO (data is available male/female)

    Returns
        df_cohort_sizes (df): rows are countries, columns are a cohort's (e.g. age=2) 
                                size each year, then the next cohort (columns labelled e.g. 2_1950 age=2, year=1950)
        ages (arr) : central year of interval (2,7...102)
        years (arr) : years we have data for (1950, 1955...2100)
    """

    def convert_age_range(age):
        if age == '100+':
            return 100
        else:
            match = re.match(r'(\d+)--\d+', age)
            if match:
                return int(match.group(1))
            else:
                return int(age)
            

    # open wcde cohort size file 
    filepath = filepaths_wcde[ssp-1]
    df_raw = pd.read_csv(filepath, header=7) # population is in 000's

    # total national population through time (rows = countries with names from WCDE, check they match, columns = years)
    df_pop_national = df_raw[(df_raw['Sex'] == 'Both') & (df_raw['Age'] == 'All')][['Area', 'Year', 'Population']].pivot(index="Area", columns="Year", values="Population")

    # cohort size specific
    if by_sex == False:

        # select only relevant rows and cols
        df = df_raw[(df_raw['Sex'] == 'Both') & (df_raw['Age'] != 'All')][['Area', 'Year', 'Age', 'Population']]
        
        # central year in age bracket e.g. 0-4 becomes 2, 5-9 becomes 7 
        df['Age'] = df['Age'].apply(convert_age_range) + 2 
              
        # Initialize an empty DataFrame for the final result
        df_cohort_sizes = pd.DataFrame()
        # Get unique ages
        ages = df['Age'].unique()
        years = df['Year'].unique()
        
        # Loop through each age and pivot the data
        for age in ages:
            subset = df[df['Age'] == age].pivot(index='Area', columns='Year', values='Population')
            subset.columns = [f'{age}_{year}' for year in subset.columns] # name the columns e.g. 2_1950
            if df_cohort_sizes.empty:
                df_cohort_sizes = subset
            else:
                df_cohort_sizes = df_cohort_sizes.join(subset, how='outer')

    else:
        pass
        # TO DEVELOP ! BY SEX ! 
    
    return df_cohort_sizes, ages, years




def load_population(
    dir_population='./data/gridded-pop/', 
    startyear=1850,
    endyear=2100,
    ssp=3,
    urbanrural=False,
    chunksize=100
):
    """
    Load gridded population reconstructions (histsoc) + projections (SSPs) from ISIMIP. 
    Gridded population density at 0.5 degrees, annual expressed as number of people. 
    ISIMIP2b has histsoc until 2005. ISIMIP3b has histsoc until 2021 (duplicated from ISIMIP3a), 
    then from Gao et al. 2020 (https://doi.org/10.5065/D60Z721H AND https://doi.org/10.7927/q7z9-9r69),
    scaled to match ISIMIP national population projections under different SSPs. 

    Notes: Other SSPs are available from Gao et al. but haven't been scaled to match ISIMIP - > also 5 arcmin is available too! 
    national population totals. 
    Did they fix the hist-to-ssp transition period? Dont think so. Fix this if important for analyses. 

    Input: 
        filepaths to gridded population (embedded in function for now). 
        Implemented combinations isimip3-ssp1, isimip2-ssp2, isimip3-ssp3. 
        urbanrural: False loads only population total, True loads total, urban and rural variables 
    
    Returns:
        da_population: (DataArray)  gridded population density. 
    """

    if urbanrural:
        VARs=['urban-population','rural-population','total-population']
    else:
        VARs='total-population'
    
    # Initialize list to store datasets
    datasets = []

    # Load historical data conditionally based on the start and end year
    if startyear <= 1900:
        da_pop_histsoc1 = xr.open_dataset(
            os.path.join(dir_population, 'ISIMIP3/ISIMIP3b/histsoc/population_histsoc_30arcmin_annual_1850_1900.nc')
        )[VARs]
        da_pop_histsoc1['time'] = da_pop_histsoc1['time'].dt.year
        da_pop_histsoc1 = da_pop_histsoc1.sel(time=slice(startyear, min(endyear, 1900)))
        datasets.append(da_pop_histsoc1)

    if startyear <= 2014 and endyear >= 1901:
        da_pop_histsoc2 = xr.open_dataset(
            os.path.join(dir_population, 'ISIMIP3/ISIMIP3b/histsoc/population_histsoc_30arcmin_annual_1901_2014.nc')
        )[VARs]
        da_pop_histsoc2['time'] = da_pop_histsoc2['time'].dt.year
        da_pop_histsoc2 = da_pop_histsoc2.sel(time=slice(max(startyear, 1901), min(endyear, 2014)))
        datasets.append(da_pop_histsoc2)

    # Load SSP data conditionally
    if endyear >= 2015:
        print(f'opening isimip3 - ssp{ssp}')
        da_pop_sspsoc = xr.open_dataset(
            glob.glob(os.path.join(dir_population, f'ISIMIP3/ISIMIP3b/ssp{ssp}*/population_ssp{ssp}_30arcmin_annual_2015_2100.nc'))[0],
            decode_times=False
        )[VARs]
        da_pop_sspsoc['time'] = np.array([year for year in np.arange(2015, 2101)])
        da_pop_sspsoc = da_pop_sspsoc.sel(time=slice(max(startyear, 2015), endyear))
        datasets.append(da_pop_sspsoc)

    # Concatenate datasets if there are multiple
    if len(datasets) > 1:
        da_population = xr.concat(datasets, dim='time')
    else:
        da_population = datasets[0]
    
    return da_population





def load_countrymasks_fillcoasts(
    filepath='./data/country-masks/isipedia-countries/countrymasks_fractional.nc',
fillcoast=True):

    # Part 1. Open data 
    
    ds=xr.open_dataset(filepath)
    da_countrymasks = ds.to_array()

    strings = da_countrymasks['variable'].values
    cleaned_strings = [s[2:] if s.startswith('m_') else s for s in strings]
    da_countrymasks['variable'] = cleaned_strings
    # last variable is 'world', lose it 
    da_countrymasks = da_countrymasks.isel(variable=slice(0,225))
    # sum over all countries 
    countrymask_sum = da_countrymasks.isel(variable=slice(0,225)).sum(dim='variable')

    if fillcoast:
        # Part 2. Correct for coastal pixels 
        
        # where sum of fraction is less than 1, weighted multiplication for sum to equal one
        da_countrymasks_correct = xr.where(countrymask_sum < 1, da_countrymasks*(1/da_countrymasks.sum(dim='variable')), da_countrymasks)
        # small area sum = 2, correct for it 
        da_countrymasks_corr = xr.where(da_countrymasks_correct.sum(dim='variable') > 1, da_countrymasks_correct/da_countrymasks_correct.sum(dim='variable'), da_countrymasks_correct)
    
    
        return da_countrymasks_corr
    else:
        return da_countrymasks







def match_country_names_all_mask_frac(
    filepath_isimip_countries = './data/country-masks/isipedia-countries/countryData.json',
    filepath_world_bank = './data/income-groups/world_bank/CLASS.xlsx',
    filepaths_wcde = ['./data/cohort-sizes/WCDE/wicdf_ssp1.csv',
                      './data/cohort-sizes/WCDE/wicdf_ssp2.csv', 
                      './data/cohort-sizes/WCDE/wicdf_ssp3.csv'],
    filepath_mask='./data/country-masks/isipedia-countries/countrymasks.geojson',
    filepath_mask_frac='./data/country-masks/isipedia-countries/countrymasks_fractional.nc',
):
    """
    A somewhat ugly function that matches country names and country codes between all data sources used. Namely, 
    isimip_countries : 195 UN official/observer countries (from isipedia-countries)
    world_bank : includes countries, region and income group information for 218 countries/admin units
    wcde : 202 countries/administrative regions
    mask (geojson): 208 countries/admin groups
    mask (fractional mask): 225 countries/admin groups

    Todo: get all info from WB not only for 195 isimip countries ! 
    """

    # load metadata from isimip and world bank
    df_metadata = load_country_metadata(filepath_isimip_countries = filepath_isimip_countries, filepath_world_bank=filepath_world_bank, keep_names='both')
    # load cohortsize metadata and rename column for consistency
    df_wcde, none, none = load_cohort_sizes( filepaths_wcde = filepaths_wcde)
    df_wcde = df_wcde.reset_index()[['Area']].rename(columns={'Area':'country_wcde'})
    # open geojson mask 
    df_mask =gpd.read_file(filepath_mask).iloc[:,[12,14]].rename(columns={'ISIPEDIA':'iso3_mask', 'NAME':'country_mask'})
    # open da countrymask
    da_frac=load_countrymasks_fillcoasts(filepath=filepath_mask_frac,fillcoast=False)
    df_frac=da_frac['variable'].to_pandas().rename('iso3_frac') # don't actually need it here

    # Step 1: Merge wcde on 'country' (isimip)
    merged_df = df_metadata.merge(df_wcde, how='outer', left_on='country', right_on='country_wcde', indicator='merge_country')

    # Step 2: Merge wcde on 'country_wb' 
    unmatched_df = merged_df[merged_df['merge_country'] == 'left_only'].drop(columns=['country_wcde', 'merge_country'])
    second_merge = unmatched_df.merge(df_wcde, how='left', left_on='country_wb', right_on='country_wcde', indicator='merge_country_wb')

    # Combine matched results
    final_merged_df = pd.concat([merged_df[merged_df['merge_country'] != 'left_only'], second_merge])
    final_merged_df

    # Step 3: Check for common words for remaining unmatched rows
    remaining_unmatched = final_merged_df[final_merged_df['merge_country_wb'] == 'left_only'].copy()
    df_wcde_tomatch = final_merged_df[ final_merged_df['merge_country']=='right_only']

    def find_common_word_match(row, choices, column):
        row_value = row[column]

        # Define stopwords to ignore and minimum word length
        stopwords = {'State','of','of)','Korea','and', 'States', 'United', 'Islands'}
        min_length = 3
        # Define specific mappings for manual matches
        specific_matches = {
            'United States': 'United States of America',
            'Eswatini (Kingdom of)': 'Swaziland',
        }
        # Handle specific matches first
        if row_value in specific_matches:
            return specific_matches[row_value], None
        # Clean the row value by removing stopwords and words shorter than min_length
        row_words = set(word for word in row_value.split() if len(word) >= min_length and word not in stopwords)
        for choice in choices:
            choice_words = set(word for word in choice.split() if len(word) >= min_length and word not in stopwords)
            common_words = row_words & choice_words
            if len(common_words) >= 2:  # Check for at least two common words
                return choice, None
        for choice in choices:
            choice_words = set(word for word in choice.split() if len(word) >= min_length and word not in stopwords)
            common_words = row_words & choice_words
            if len(common_words) == 1:  # Check for exactly one common word
                #print(choice, common_words)
                return choice, common_words
        return None, None
    
    # Apply the function and capture matches with one common word
    remaining_unmatched[['country_wcde', 'common_words']] = remaining_unmatched.apply(
        lambda row: pd.Series(find_common_word_match(row, df_wcde_tomatch['country_wcde'].tolist(), 'country')), axis=1)
    
    # Filter rows where only one common word was found - can delete this was for checking
    matches_with_one_word = remaining_unmatched[remaining_unmatched['common_words'].apply(lambda x: x is not None and len(x) == 1)]

    # Remove the common_words column
    remaining_unmatched = remaining_unmatched.drop(columns=['common_words'])

    # Step 4: Final merge using common word matches
    common_word_matched_df = remaining_unmatched.merge(df_wcde, how='left', on='country_wcde', indicator='merge_common_word')

    # Combine all matched results
    final_combined_df = pd.concat([final_merged_df[final_merged_df['merge_country_wb'] != 'left_only'], common_word_matched_df])

    # drop duplicate rows of country_wcde that have already been assigned 
    def drop_duplicate_assigned_rows(final_combined_df,column):
        # Step 1: Identify and filter non-unique 'country_wcde' values
        non_unique_country_wcde = final_combined_df['country_wcde'].value_counts()[lambda x: x > 1].index
        non_unique_rows = final_combined_df[final_combined_df['country_wcde'].isin(non_unique_country_wcde)]
        # Step 2: Remove rows with NaN in 'country' from the non-unique rows
        final_combined_df = final_combined_df.drop(non_unique_rows[non_unique_rows[column].isna()].index)
        return final_combined_df

    final_combined_df = drop_duplicate_assigned_rows(final_combined_df,'country_iso3')

    # Step 5: Check for a common substring of 4 characters or more for remaining unmatched rows
    df_wcde_tomatch = final_combined_df[ final_combined_df['merge_country']=='right_only']
    remaining_unmatched = final_combined_df[final_combined_df['merge_common_word'] == 'left_only'].copy()
    
    def find_common_substring_match(row, choices, column, min_length=4):
        # Define stopwords to ignore in matching 
        stopwords_substring = ['States','United','Republic','mini','tini','land','e of', ' of', ' of ','l Is','Islands']
        cleaned_row_value = ' '.join([word for word in row[column].split() if word not in stopwords_substring])
        for choice in choices:
            cleaned_choice = ' '.join([word for word in choice.split() if word not in stopwords_substring])
            for i in range(len(cleaned_row_value) - min_length + 1):
                substr = cleaned_row_value[i:i+min_length]
                if substr in cleaned_choice and substr not in stopwords_substring:
                    #print(substr, choice) 
                    return choice
        return None
    
    remaining_unmatched['country_wcde'] = remaining_unmatched.apply(
        lambda row: find_common_substring_match(row, df_wcde_tomatch['country_wcde'].tolist(), 'country'), axis=1)

    # Step 6: Final merge using common substring matches
    substring_matched_df = remaining_unmatched.merge(df_wcde, how='left', on='country_wcde', indicator='merge_substring')
    
    # Combine all matched results
    final_combined_df = pd.concat([final_combined_df[final_combined_df['merge_common_word'] != 'left_only'], substring_matched_df])
    final_combined_df = drop_duplicate_assigned_rows(final_combined_df,'country_iso3')

    # Part 2. Include mask countries that are not in 195 country list

    # Step 1: do a first outer merge of the combined df and the countries in the mask
    # do a first outer merge with mask coutnries based on isocode
    df_merge = final_combined_df.merge(df_mask, how='outer', left_on='country_iso3', right_on='iso3_mask',indicator='merge_country_mask')

    # get unmatched countries in mask
    unmatched_mask = df_merge[df_merge['merge_country_mask']=='right_only']
    # get unmatched countries in wcde
    df_wcde_unmatched = final_combined_df[ final_combined_df['merge_country']=='right_only']

    # Step 2: match mask with wcde based on common name of country 
    df_unmatched_mask = unmatched_mask.drop(columns=['country_wcde','merge_country_mask'])
    second_merge = df_unmatched_mask.merge(df_wcde_unmatched[['country_wcde']], how='left', left_on='country_mask', right_on='country_wcde', indicator='merge_country_msk_n')
    combined_df = pd.concat([df_merge[df_merge['merge_country_mask'] !='right_only'],second_merge])

    # Step 3: find unmatched countries and match based on common substring
    remaining_unmatched = combined_df[combined_df['merge_country_msk_n'] =='left_only'].copy()
    final_combined_df = drop_duplicate_assigned_rows(combined_df,'country_mask')
    df_wcde_tomatch = final_combined_df[ final_combined_df['merge_country']=='right_only']

    remaining_unmatched['country_wcde'] = remaining_unmatched.apply(
    lambda row: find_common_substring_match(row, df_wcde_tomatch['country_wcde'].tolist(), 'country_mask'), axis=1)

    substring_matched_df = remaining_unmatched.merge(df_wcde, how='left', on='country_wcde', indicator='merge_substring_msk')

    # combine
    final_combined_df = pd.concat([final_combined_df[final_combined_df['merge_country_msk_n'] !='left_only'], substring_matched_df])
    final_combined_df = drop_duplicate_assigned_rows(final_combined_df, 'country_mask')

    # merge also from fractional countrymask codes
    df_merge = final_combined_df.merge(df_frac, how='outer',left_on='iso3_mask',right_on='variable',indicator='merge_frac')
    df_both = df_merge[df_merge['merge_frac']=='both']
    df_unmatched = df_merge[df_merge['merge_frac']=='left_only']
    df_tomatch = df_merge[df_merge['merge_frac']=='right_only']
    second_merge = df_unmatched.drop(columns='iso3_frac').merge(df_tomatch['iso3_frac'], how='outer',left_on='country_iso3',right_on='iso3_frac',indicator='merge_frac2')

    final_combined_df=pd.concat([df_both,second_merge])

    
    # Identify and print unmatched countries
    unmatched_countries = final_combined_df[final_combined_df['merge_substring'] == 'left_only']
    print("Unmatched ISIMIP countries (without WCDE data) after all merges:")
    print(unmatched_countries[['country', 'country_wb']])

    # WCDE countries unmatched
    df_wcde_unmatched = final_combined_df[ final_combined_df['merge_country']=='right_only']
    print("Unmatched WCDE countries after all merges:")
    print(df_wcde_unmatched[['country_wcde']])  
    
    # Identify and print unmatched mask countries
    unmatched_countries = final_combined_df[(final_combined_df['merge_substring_msk'] == 'left_only') | (final_combined_df['merge_frac2'] == 'right_only') ]
    print("Unmatched ISIMIP mask countries (geojson + frac mask) after all merges:")
    print(unmatched_countries[['country_mask', 'iso3_frac']])

    
    # Drop merge indicator columns
    df_countries_matched = final_combined_df.drop(columns=['merge_country', 'merge_country_wb', 
                                                           'merge_common_word', 'merge_substring', 
                                                           'merge_country_msk_n', 'merge_substring_msk',
                                                           'merge_frac','merge_frac2', # cols to drop
                                                          ])[['country', 
                                                              'country_wb', 
                                                              'country_wcde', 
                                                              'country_mask',
                                                              'country_iso3', 
                                                              'iso3_mask',
                                                              'iso3_frac', 
                                                              'country_code', 
                                                              'region',
                                                              'income_group']] # cols to keep 

    
    return df_countries_matched.sort_values(['country','country_wcde','country_mask']).reset_index(drop=True)








def interpolate_cohortsize_countries(
    df_cohort_sizes,
    cohort_ages,
    cohort_years,
): 
    """
    """

    # keep all possible countries (better, you lose less places)
    df_cohort_size_filter = df_cohort_sizes 
    
    def distribute_error_across_years(df_y_values, df_y_mean_bracket, bracket_size): 
        # for a single year / single country in the dataset distribute error in age bracket
        
        # ignore warnings, we get rid of nans later with the nansum
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # reshape df to array, each row is a bracket, each column is a specific age in that bracket 
            y_values = np.reshape(df_y_values.values, (len(df_y_values)//bracket_size, bracket_size)) #nrows, ncols
            # calculate interpolation error over the bracket as the sum of errors
            delta_bracket = np.sum(y_values - df_y_mean_bracket.values[:, np.newaxis], axis=1) # sums along row
            # calculate relative weights as the value divided by the sum of all values in the bracket
            sum_over_years = np.sum(y_values, axis=1)
            weights = y_values / sum_over_years[:, np.newaxis]
            # compute correction for each y value
            delta_i = weights * delta_bracket[:, np.newaxis]
            # correct the y values 
            y_corrected = np.nansum(np.dstack((y_values,-delta_i)),2).reshape(-1)
            
        return y_corrected
    
    # unpack loaded wcde values
    wcde_years, wcde_ages, wcde_country_data = cohort_years, cohort_ages, df_cohort_size_filter.values #.iloc[:,2:]
    
    # initialise dictionary to store cohort sizes dataframes per country with years as rows and ages as columns
    d_cohort_size = {}
    
    # loop over countries
    print('interpolating cohort sizes per country')
    for i,name in enumerate(df_cohort_size_filter.index):
        # extract population size per age cohort data from WCDE file and linearly interpolate from 5-year WCDE blocks to pre-defined birth year
        wcde_per_country = np.reshape(wcde_country_data[i,:],((len(wcde_ages),len(wcde_years)))) 
        # every row is an age cohort (len 21), every column is a year (len 31)
        # use dataframes to do reindexing and interpolation (see how much slower this makes it cfr. to numpy - could do with numpy interpolate.griddata if you accept that ages 0-2 are not interpolated but held constant - decide what assumption we want to use!) 
        wcde_per_country_df = pd.DataFrame(
            wcde_per_country,
            index=wcde_ages,
            columns=wcde_years
        )
    
        #set new coordinates after interpolation - check you want this & put in flags at start or something !! 
        ages_interpn_cohorts =  np.arange(0,105) # ISSUE: understand if OK np.arange(104,-1,-1) #np.arange(100,-1,-1) # new_ages in luke's script (prev: np.arange(0,105))
        years_interpn_cohorts = np.arange(1950,2100+1)
    
        # interpolate per ages
        wcde_per_country_df = wcde_per_country_df.reindex(ages_interpn_cohorts)
        wcde_per_country_df
        wcde_per_country_intrp = wcde_per_country_df.astype('float').interpolate(
                method='slinear', # original 'linear' filled end values with constants; slinear calls spline linear interp/extrap from scipy interp1d
                limit_direction='both',
                fill_value='extrapolate',
                axis=0
            )
        # set negative numbers to zero
        wcde_per_country_intrp[wcde_per_country_intrp<0]=0
        # fix the not mean preserving issue
        wcde_per_country_intrp_correct = wcde_per_country_intrp.copy()
        for y in wcde_years:
            wcde_per_country_intrp_correct.loc[:,y] = distribute_error_across_years(
                wcde_per_country_intrp.loc[:,y], # interpolated values
                wcde_per_country_df.dropna().loc[:,y], # true mean
                bracket_size=5) # bracket size 
         
        # check for neg numbers
        if (wcde_per_country_intrp_correct < 0).any().any():
            print('after interpolation and mean-preserving correction there are some neg numbers in {}, {}, setting them to zero'.format(i,name))
            # set them to zero
            wcde_per_country_intrp_correct[wcde_per_country_intrp_correct<0]=0
            # TODO: modify distribute_error_across_years to not reintroduce negative numbers 
    
        # interpolate between years
        wcde_per_country_df = wcde_per_country_intrp_correct.transpose().reindex(years_interpn_cohorts)
        wcde_per_country_intrp_years = wcde_per_country_df.astype('float').interpolate(
                method='slinear', # original 'linear' filled end values with constants; slinear calls spline linear interp/extrap from scipy interp1d
                limit_direction='both',
                fill_value='extrapolate',
                axis=0
            )
        d_cohort_size[name] = wcde_per_country_intrp_years / 5
    
        #  make a data array with the information from all the countries together
    da_cohort_size = xr.DataArray(
        np.asarray([v for k,v in d_cohort_size.items()]), # see whether to include nan countries here -  np.asarray([v for k,v in d_cohort_size.items() if k in df_cohort_size_filter['country'].values])
        coords={
            'country': ('country', df_cohort_size_filter.index),
            'time': ('time', years_interpn_cohorts),
            'ages': ('ages', ages_interpn_cohorts),
        },
        dims=[
            'country',
            'time',
            'ages',
        ],
        name='cohort_size'
)


    return da_cohort_size











def get_gridscale_demographics(
    da_population,
    da_countrymasks,
    df_countries_matched, 
    da_cohort_size,
    startyear=2000,
    endyear=2005,
    chunksize=100
):
    """
    To do: make a wrapper function that runs all previous and does this
    make a function that does this just for one country/region if one only wants a certain country?
    """

    da_pop = da_population.sel(time=slice(startyear, endyear)).chunk({'time': chunksize, 'lat': chunksize, 'lon': chunksize})  # check optimal chunking sizes and whether to chunk here or above,myabe here? 
    
    # Initialize the combined demographics DataArray
    da_pop_demographics = None
    
    # Fix issue in Singapore pixel, assign fraction from IOSID to SGP 
    da_countrymasks.loc[dict(lat=da_countrymasks.lat[177], lon=da_countrymasks.lon[567], variable='SGP')] += da_countrymasks.loc[dict(lat=da_countrymasks.lat[177], lon=da_countrymasks.lon[567], variable='IOSID')].values
    da_countrymasks.loc[dict(lat=da_countrymasks.lat[177], lon=da_countrymasks.lon[567], variable='IOSID')] = 0
    
    # Fix it also in Mauritius 
    da_countrymasks.loc[dict(lat=da_countrymasks.lat[220], lon=da_countrymasks.lon[474], variable='MUS')] += da_countrymasks.loc[dict(lat=da_countrymasks.lat[220], lon=da_countrymasks.lon[474], variable='IOSID')].values
    da_countrymasks.loc[dict(lat=da_countrymasks.lat[220], lon=da_countrymasks.lon[474], variable='IOSID')] = 0
    
    
    # Loop over countries in WCDE cohort sizes
    for country in da_cohort_size.country.values:
        print(country)
    
        # Get iso3 code of the country in the mask 
        iso = df_countries_matched[df_countries_matched['country_wcde']==country]['iso3_frac'].values[0]
    
        # if this isocode is in the mask file 
        if iso in da_countrymasks['variable']: # do this in a slightly more intelligent way??? similar to what i was doing b4 with the dataframs, instead of if
        
            # Get cohort sizes of the country
            da_smple_cht = da_cohort_size.sel(country=country).sel(time=slice(startyear, endyear)).chunk({'time': 10, 'ages': 10})
        
            # Cohort relative sizes in the sample country
            da_smple_cht_prp = da_smple_cht / da_smple_cht.sum(dim='ages')
        
            # Get population of that country and multiply by fraction of each cohort
            pop_country = ((da_pop * da_countrymasks.sel(variable=iso)) * da_smple_cht_prp).drop_vars(['variable', 'country'])
        
            if da_pop_demographics is None:
                da_pop_demographics = pop_country
            else:
                da_pop_demographics += pop_country
        
            # Explicitly clear intermediate variables to free up memory
            del iso, da_smple_cht, da_smple_cht_prp, pop_country
        else:
            print('**iso not in mask')
            pass
    
    da_pop_demographics = da_pop_demographics.compute()
    
    return da_pop_demographics






def population_demographics_gridscale_global(
    startyear=2000,
    endyear=2005,
    ssp=2,
    urbanrural=False,
    chunksize=100
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
        df_countries_matched = match_country_names_all_mask_frac();

    df_cohort_sizes, ages, years = load_cohort_sizes(ssp=ssp)

    da_population = load_population(ssp=ssp,
                                    startyear=startyear,
                                    endyear=endyear,
                                   urbanrural=urbanrural)

    print('loading country masks')
    da_countrymasks = load_countrymasks_fillcoasts().chunk({'lat': chunksize, 'lon': chunksize})

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
                                                 endyear=endyear);



    return da_pop_demographics
