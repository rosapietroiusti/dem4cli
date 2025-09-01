"""
dem4cli v1.0 match country names

"""



import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd 
from scipy import interpolate
import glob, os, re, sys
import warnings
import openpyxl 

from settings import * 
from utils import * 
from population_demographics import * 


def match_country_names_all_mask_frac(
    filepath_isimip_countries = os.path.join(script_dir, 'data/country-masks/isipedia-countries/countryData.json'),
    filepath_world_bank = os.path.join(script_dir, 'data/income-groups/world_bank/CLASS.xlsx'),
    filepaths_wcde = [os.path.join(script_dir, 'data/cohort-sizes/WCDE/wicdf_ssp1.csv'),
                      os.path.join(script_dir, 'data/cohort-sizes/WCDE/wicdf_ssp2.csv'), 
                      os.path.join(script_dir, 'data/cohort-sizes/WCDE/wicdf_ssp3.csv')],
    filepath_mask=os.path.join(script_dir, 'data/country-masks/isipedia-countries/countrymasks.geojson'),
    filepath_mask_frac=os.path.join(script_dir, 'data/country-masks/isipedia-countries/countrymasks_fractional.nc'),
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
    print("\nUnmatched WCDE countries after all merges:")
    print(df_wcde_unmatched[['country_wcde']])  
    
    # Identify and print unmatched mask countries
    unmatched_countries = final_combined_df[(final_combined_df['merge_substring_msk'] == 'left_only') | (final_combined_df['merge_frac2'] == 'right_only') ]
    print("\nUnmatched ISIMIP mask countries (geojson + frac mask) after all merges:")
    print(unmatched_countries[['country_mask', 'iso3_frac']])

    
    # Drop merge indicator columns
    df_countries_matched = final_combined_df.drop(columns=['merge_country', 'merge_country_wb', 
                                                           'merge_common_word', 'merge_substring', 
                                                           'merge_country_msk_n', 'merge_substring_msk',
                                                           'merge_frac','merge_frac2',              # cols to drop
                                                          ])[['country', # isimip country data
                                                              'country_wb', # world bank
                                                              'country_wcde',  # wcde
                                                              'country_mask',  # geojson mask
                                                              'country_iso3', # world bank
                                                              'iso3_mask', # geojson mask 
                                                              'iso3_frac', # fractional mask
                                                              'country_code', # isimip country data
                                                              'region', # world bank
                                                              'income_group']] # world bank               #  cols to keep 

    df_countries_matched = df_countries_matched.rename(columns={"country": "country"})

    return df_countries_matched.sort_values(['country','country_wcde','country_mask']).reset_index(drop=True)

