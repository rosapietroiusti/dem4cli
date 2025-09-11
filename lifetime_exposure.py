"""
Demographics4Climate : Population and demographics for climate science analysis
----------------------------------------------

2024 Update

Calculate lifetime exposure 

Updated to UNWPP2024 

""" 




import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd # can maybe delete if i dont open geojson in the end 
from scipy import interpolate
import glob, os, re, sys
import warnings
import openpyxl 


script_dir = os.path.abspath( os.path.dirname( __file__ ) )



def load_isimip(
    extremes, 
    model_names,
    df_GMT_15,
    df_GMT_20,
    df_GMT_NDC,
    df_GMT_OS,
    df_GMT_noOS,
    ds_GMT_STS,
    df_GMT_strj,
    flags,
): 
    
    if flags['run']: 

        print('Processing ISIMIP data')

        # initialise counter, metadata dictionary, pic list, pic meta, and 
        i = 1
        d_isimip_meta = {}
        pic_list = []
        d_pic_meta = {}

        # rolling mean option
        if flags['rm'] == 'no_rm':

            print("\nNo smoothing apply to the GMT pathways under the RCP trajectories of the ESM/ISIMIP model\n")
            
            pass
        
        else:
            
            print("\nSmoothing apply to the GMT pathways under the RCP trajectories of the ESM/ISIMIP model\n")

        if flags['extr']=="all":

            if not os.path.exists(data_dir+'{}/{}'.format(flags['version'],flags['extr'])):
                    os.mkdir(data_dir+'{}/{}'.format(flags['version'],flags['extr']))

        # loop over extremes
        for extreme in extremes:

            print('Processing for {}'.format(extreme))

            if not os.path.exists(data_dir+'{}/{}'.format(flags['version'],extreme)):
                os.mkdir(data_dir+'{}/{}'.format(flags['version'],extreme))

            # define all models
            models = model_names[extreme]

            # loop over models
            for model in models: 

                # store all files starting with model name
                #file_names = sorted(glob.glob(data_dir+'isimip/'+flags['extr']+'/'+model.lower()+'/'+model.lower()+'*rcp*landarea*2099*')) #Luke's version
                file_names = sorted(glob.glob(data_dir+'isimip/'+extreme+'/'+model.lower()+'/'+model.lower()+'*rcp*landarea*2099*'))
                for file_name in file_names: 

                    print('Loading '+file_name.split('\\')[-1]+' ('+str(i)+')')

                    # load rcp data (AFA: Area Fraction Affected) - and manually add correct years
                    da_AFA_rcp = open_dataarray_isimip(file_name)

                    # save metadata
                    d_isimip_meta[i] = {
                        'model': file_name.split('_')[0].split('\\')[-1],
                        'gcm': file_name.split('_')[1],
                        'rcp': file_name.split('_')[2],
                        'extreme': file_name.split('_')[3],
                    }

                    #load associated historical variable
                    file_name_his = glob.glob(data_dir+'isimip/'+extreme+'/'+model.lower()+'/'+model.lower()+'*'+d_isimip_meta[i]['gcm']+'*_historical_*landarea*')[0]
                    da_AFA_his = open_dataarray_isimip(file_name_his)

                    # load GMT for rcp and historical period - note that these data are in different files
                    if d_isimip_meta[i]['gcm'] == 'hadgem2-es': # .upper() method doesn't work for HadGEM2-ES on linux server (only Windows works here)
                        file_names_gmt = glob.glob(data_dir+'isimip/DerivedInputData/globalmeans/tas/HadGEM2-ES/*.fldmean.yearmean.txt') # ignore running mean files
                    else:
                        file_names_gmt = glob.glob(data_dir+'isimip/DerivedInputData/globalmeans/tas/'+d_isimip_meta[i]['gcm'].upper()+'/*.fldmean.yearmean.txt') # ignore running mean files
                    file_name_gmt_fut = [s for s in file_names_gmt if d_isimip_meta[i]['rcp'] in s]
                    file_name_gmt_his = [s for s in file_names_gmt if '_historical_' in s]
                    file_name_gmt_pic = [s for s in file_names_gmt if '_piControl_' in s]

                    GMT_fut = pd.read_csv(
                        file_name_gmt_fut[0],
                        delim_whitespace=True,
                        skiprows=1,
                        header=None).rename(columns={0:'year',1:'tas'}).set_index('year')
                    GMT_his = pd.read_csv(
                        file_name_gmt_his[0],
                        delim_whitespace=True, 
                        skiprows=1, 
                        header=None).rename(columns={0:'year',1:'tas'}).set_index('year')
                    GMT_pic = pd.read_csv(
                        file_name_gmt_pic[0],
                        delim_whitespace=True, 
                        skiprows=1, 
                        header=None).rename(columns={0:'year',1:'tas'}).set_index('year')

                    # concatenate historical and future data
                    da_AFA = xr.concat([da_AFA_his,da_AFA_rcp], dim='time')
                    df_GMT = pd.concat([GMT_his,GMT_fut])

                    # convert GMT from absolute values to anomalies - use data from pic until 1861 and from his from then onwards
                    df_GMT = df_GMT - pd.concat([GMT_pic.loc[year_start_GMT_ref:np.min(GMT_his.index)-1,:], GMT_his.loc[:year_end_GMT_ref,:]]).mean()

                    # if needed, repeat mean of last 10 years until entire period of interest is covered
                    if da_AFA.time.max() < year_end: 
                        da_AFA_lastyear = da_AFA.sel(time=slice(da_AFA.time.max()-9,da_AFA.time.max())).mean(dim='time').expand_dims(dim='time',axis=0)
                        GMT_lastyear = df_GMT.iloc[-10:,:].mean() # mean of last 10 years to fill time span 

                        for year in range(da_AFA.time.max().values+1,year_end+1): 
                            da_AFA = xr.concat([da_AFA,da_AFA_lastyear.assign_coords(time = [year])], dim='time')
                            if len(df_GMT) < 439: # necessary to avoid this filling from 2100-2113 if GMTs already go to 2299
                                df_GMT = pd.concat([df_GMT,pd.DataFrame(data={'tas':GMT_lastyear['tas']},index=[year])])

                    # retain only period of interest
                    da_AFA = da_AFA.sel(time=slice(year_start,year_end))
                    df_GMT = df_GMT.loc[year_start:year_end,:]
                    
                    # rolling mean option
                    if flags['rm'] == 'no_rm':
                        
                        pass
                    
                    else:

                        if flags['rm_config'] =='21':
                        
                            df_GMT = df_GMT.rolling(window=21,min_periods=10,center=True,).mean()

                        if flags['rm_config'] =='11':
                            
                            df_GMT = df_GMT.rolling(window=11,min_periods=5,center=True,).mean()

                    # save GMT in metadatadict
                    d_isimip_meta[i]['GMT'] = df_GMT 

                    # recover the two GMT for the two scenario of interest in the STS pathways 

                    da_GMT_STS_ModAct = ds_GMT_STS['tas'].sel(
                        time=slice(1960, 2113),
                        percentile='50.0',
                        scenario='ModAct'
                        )

                    da_GMT_STS_Ren = ds_GMT_STS['tas'].sel(
                        time=slice(1960, 2113),
                        percentile='50.0',
                        scenario='Ren'
                        )
                    
                    

                    # get ISIMIP GMT indices closest to GMT trajectories        
                    RCP2GMT_diff_15 = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_15.values.transpose()), axis=0)
                    RCP2GMT_diff_20 = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_20.values.transpose()), axis=0)
                    RCP2GMT_diff_NDC = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_NDC.values.transpose()), axis=0)
                    RCP2GMT_diff_R26eval = np.min(np.abs(d_isimip_meta[i]['GMT'].values - d_isimip_meta[1]['GMT'].values.transpose()), axis=0)
                    RCP2GMT_diff_OS = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_OS.values.transpose()), axis=0)
                    RCP2GMT_diff_noOS = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_noOS.values.transpose()), axis=0)
                    RCP2GMT_diff_STS_ModAct = np.min(np.abs(d_isimip_meta[i]['GMT'].values - da_GMT_STS_ModAct.values.transpose()), axis=0)
                    RCP2GMT_diff_STS_Ren = np.min(np.abs(d_isimip_meta[i]['GMT'].values - da_GMT_STS_Ren.values.transpose()), axis=0)

                    ind_RCP2GMT_15 = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_15.values.transpose()), axis=0)
                    ind_RCP2GMT_20 = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_20.values.transpose()), axis=0)
                    ind_RCP2GMT_NDC = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_NDC.values.transpose()), axis=0)
                    ind_RCP2GMT_R26eval = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - d_isimip_meta[1]['GMT'].values.transpose()), axis=0)
                    ind_RCP2GMT_OS = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_OS.values.transpose()), axis=0)
                    ind_RCP2GMT_noOS = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_noOS.values.transpose()), axis=0)
                    ind_RCP2GMT_STS_ModAct = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - da_GMT_STS_ModAct.values.transpose()), axis=0)
                    ind_RCP2GMT_STS_Ren = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - da_GMT_STS_Ren.values.transpose()), axis=0)

                    # store GMT maxdiffs and indices in metadatadict
                    d_isimip_meta[i]['GMT_15_maxdiff'] = np.nanmax(RCP2GMT_diff_15)
                    d_isimip_meta[i]['GMT_20_maxdiff'] = np.nanmax(RCP2GMT_diff_20)
                    d_isimip_meta[i]['GMT_NDC_maxdiff'] = np.nanmax(RCP2GMT_diff_NDC)
                    d_isimip_meta[i]['GMT_R26eval_maxdiff'] = np.nanmax(RCP2GMT_diff_R26eval) 
                    d_isimip_meta[i]['GMT_OS_maxdiff'] = np.nanmax(RCP2GMT_diff_OS)
                    d_isimip_meta[i]['GMT_noOS_maxdiff'] = np.nanmax(RCP2GMT_diff_noOS)
                    d_isimip_meta[i]['GMT_STS_ModAct_maxdiff'] = np.nanmax(RCP2GMT_diff_STS_ModAct)
                    d_isimip_meta[i]['GMT_STS_Ren_maxdiff'] = np.nanmax(RCP2GMT_diff_STS_Ren)

                    d_isimip_meta[i]['GMT_15_valid'] = np.nanmax(RCP2GMT_diff_15) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['GMT_20_valid'] = np.nanmax(RCP2GMT_diff_20) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['GMT_NDC_valid'] = np.nanmax(RCP2GMT_diff_NDC) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['GMT_R26eval_valid'] = np.nanmax(RCP2GMT_diff_R26eval) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['GMT_OS_valid'] = np.nanmax(RCP2GMT_diff_OS) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['GMT_noOS_valid'] = np.nanmax(RCP2GMT_diff_noOS) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['GMT_STS_ModAct_valid'] = np.nanmax(RCP2GMT_diff_STS_ModAct) < RCP2GMT_maxdiff_threshold
                    d_isimip_meta[i]['GMT_STS_Ren_valid'] = np.nanmax(RCP2GMT_diff_STS_Ren) < RCP2GMT_maxdiff_threshold

                    d_isimip_meta[i]['ind_RCP2GMT_15'] = ind_RCP2GMT_15
                    d_isimip_meta[i]['ind_RCP2GMT_20'] = ind_RCP2GMT_20
                    d_isimip_meta[i]['ind_RCP2GMT_NDC'] = ind_RCP2GMT_NDC
                    d_isimip_meta[i]['ind_RCP2GMT_R26eval'] = ind_RCP2GMT_R26eval
                    d_isimip_meta[i]['ind_RCP2GMT_OS'] = ind_RCP2GMT_OS
                    d_isimip_meta[i]['ind_RCP2GMT_noOS'] = ind_RCP2GMT_noOS
                    d_isimip_meta[i]['ind_RCP2GMT_STS_ModAct'] = ind_RCP2GMT_STS_ModAct
                    d_isimip_meta[i]['ind_RCP2GMT_STS_Ren'] = ind_RCP2GMT_STS_Ren
                    
                    # run GMT mapping for stylized trajectories (repeat above but for dataframe of all trajectories)
                    d_isimip_meta[i]['GMT_strj_maxdiff'] = np.empty_like(np.arange(len(df_GMT_strj.columns)))
                    d_isimip_meta[i]['GMT_strj_valid'] = np.empty_like(np.arange(len(df_GMT_strj.columns)))
                    d_isimip_meta[i]['ind_RCP2GMT_strj'] = np.empty_like(df_GMT_strj.values)
                    
                    for step in range(len(df_GMT_strj.columns)):
                        RCP2GMT_diff = np.min(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_strj.loc[:,step].values.transpose()), axis=0)
                        d_isimip_meta[i]['ind_RCP2GMT_strj'][:,step] = np.argmin(np.abs(d_isimip_meta[i]['GMT'].values - df_GMT_strj.loc[:,step].values.transpose()), axis=0)
                        d_isimip_meta[i]['GMT_strj_maxdiff'][step] = np.nanmax(RCP2GMT_diff)
                        d_isimip_meta[i]['GMT_strj_valid'][step] = np.nanmax(RCP2GMT_diff) < RCP2GMT_maxdiff_threshold
                        
                    d_isimip_meta[i]['ind_RCP2GMT_strj'] = d_isimip_meta[i]['ind_RCP2GMT_strj'].astype(int)

                    # adding this to avoid duplicates of da_AFA_pic in pickles
                    if '{}_{}'.format(d_isimip_meta[i]['model'],d_isimip_meta[i]['gcm']) not in pic_list:

                        # load associated picontrol variables (can be from up to 4 files)
                        file_names_pic  = glob.glob(data_dir+'isimip/'+extreme+'/'+model.lower()+'/'+model.lower()+'*'+d_isimip_meta[i]['gcm']+'*_picontrol_*landarea*')

                        if  isinstance(file_names_pic, str): # single pic file 
                            da_AFA_pic  = open_dataarray_isimip(file_names_pic)
                        else: # concat pic files
                            das_AFA_pic = [open_dataarray_isimip(file_name_pic) for file_name_pic in file_names_pic]
                            da_AFA_pic  = xr.concat(das_AFA_pic, dim='time')
                            
                        # save AFA field as pickle

                        with open(data_dir+'{}/{}/isimip_AFA_pic_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'wb') as f: # added extreme to string of pickle
                            pk.dump(da_AFA_pic,f)
                            
                        pic_list.append('{}_{}'.format(d_isimip_meta[i]['model'],d_isimip_meta[i]['gcm']))
                        
                        # save metadata
                        d_pic_meta[i] = {
                            'model': d_isimip_meta[i]['model'], 
                            'gcm': d_isimip_meta[i]['gcm'],              
                            'extreme': file_name.split('_')[3], 
                            'years': str(len(da_AFA_pic.time)),
                        }
                            
                    # save AFA field as pickle

                    with open(data_dir+'{}/{}/isimip_AFA_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],str(i)), 'wb') as f: # added extreme to string of pickle
                        pk.dump(da_AFA,f)

                    # update counter
                    i += 1
        
            # save metadata dictionary as a pickle
            print('Saving metadata for {}'.format(extreme))

            if flags['rm'] == 'rm' and flags['rm_config'] =='11':

                with open(data_dir+'{}/rm_config/{}/isimip_metadata_{}_{}_{}.pkl'.format('pickles_sandbox',flags['extr'],flags['extr'],flags['gmt'],flags['rm']), 'wb') as f:
                    pk.dump(d_isimip_meta,f)
                with open(data_dir+'{}/rm_config/{}/isimip_pic_metadata_{}.pkl'.format('pickles_sandbox',flags['extr'],flags['extr']), 'wb') as f:
                    pk.dump(d_pic_meta,f) 
                with open(data_dir+'{}/rm_config/{}/df_GMT_rm_config_{}.pkl'.format('pickles_sandbox',flags['extr'],flags['rm_config']), 'wb') as f:
                    pk.dump(df_GMT,f) 
            
            elif flags['rm'] == 'rm' and flags['rm_config'] =='21':
        
                with open(data_dir+'{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],flags['gmt'],flags['rm']), 'wb') as f:
                    pk.dump(d_isimip_meta,f)
                with open(data_dir+'{}/{}/isimip_pic_metadata_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'wb') as f:
                    pk.dump(d_pic_meta,f)

                with open(data_dir+'{}/{}/df_GMT_rm_config_{}.pkl'.format(flags['version'],flags['extr'],flags['rm_config']), 'wb') as f:
                    pk.dump(df_GMT,f)

            elif flags['rm'] == 'no_rm':

                with open(data_dir+'{}/{}/df_GMT_no_rm.pkl'.format(flags['version'],flags['extr']), 'wb') as f:
                    pk.dump(df_GMT,f)

    else: 
        
        # loop over extremes
        print('⏳ Loading processed ISIMIP data')
        # loac pickled metadata for isimip and isimip-pic simulations

        if flags['rm'] == 'rm' and flags['rm_config'] =='11':
    
            with open(data_dir+'{}/rm_config/{}/isimip_metadata_{}_{}_{}.pkl'.format('pickles_sandbox',flags['extr'],flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
                d_isimip_meta = pk.load(f)
            with open(data_dir+'{}/rm_config/{}/isimip_pic_metadata_{}.pkl'.format('pickles_sandbox',flags['extr'],flags['extr']), 'rb') as f:
                d_pic_meta = pk.load(f) 
            with open(data_dir+'{}/rm_config/{}/df_GMT_rm_config_{}.pkl'.format('pickles_sandbox',flags['extr'],flags['rm_config']), 'rb') as f:
                df_GMT = pk.load(f) 
        
        elif flags['rm'] == 'rm' and flags['rm_config'] =='21':
            
            with open(pickles_dir+'{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
                d_isimip_meta = pk.load(f)
            with open(pickles_dir+'{}/{}/isimip_pic_metadata_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
                d_pic_meta = pk.load(f)
            with open(pickles_dir+'{}/{}/df_GMT_rm_config_{}.pkl'.format(flags['version'],flags['extr'],flags['rm_config']), 'rb') as f:
                df_GMT = pk.load(f)

        elif flags['rm'] == 'no_rm':

            with open(data_dir+'{}/{}/isimip_metadata_{}_{}_{}.pkl'.format(flags['version'],flags['extr'],flags['extr'],flags['gmt'],flags['rm']), 'rb') as f:
                d_isimip_meta = pk.load(f)
            with open(data_dir+'{}/{}/isimip_pic_metadata_{}.pkl'.format(flags['version'],flags['extr'],flags['extr']), 'rb') as f:
                d_pic_meta = pk.load(f)
            with open(data_dir+'{}/{}/df_GMT_no_rm.pkl'.format(flags['version'],flags['extr']), 'rb') as f:
                df_GMT = pk.load(f)            

    return d_isimip_meta,d_pic_meta