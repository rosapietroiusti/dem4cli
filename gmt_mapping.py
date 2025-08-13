"""
Adapted from L. Grant 2025 Unprecedented lifetime exposure

Amaury Laridon / Rosa Pietroiusti 

"""




def ar6_scen_grab(
    scens,
    df_GMT_all,
):
    """
    Load AR6 scenarios 
    """


    # for each line, additionally plot the candidate subsets and their names
    
    # start with upper line toward 4 degrees
    # convert to bools based on row max to find column with most maxes via idxmax
    maxes = pd.concat(
        [df_GMT_all.loc[:,c]==df_GMT_all.max(axis=1) for c in df_GMT_all.columns],
        axis=1,
    )
    df_GMT_40 = df_GMT_all.loc[:,df_GMT_all.columns[maxes.sum(axis=0).idxmax()]]
    
    # second line, 3 degrees
    # get all lines between target (3) and lower bound (first criteria)
    df_GMT_30 = df_GMT_all[
        df_GMT_all.columns[(df_GMT_all.max(axis=0)<scens['3.0'][1])&(df_GMT_all.max(axis=0)>scens['3.0'][0])]
    ]  
    # dfbools is new df with bool cells for years where series in df_GMT_30 are below the 4 deg line
    dfbools=pd.concat(
        [df_GMT_30.loc[:,c]<=df_GMT_40.loc[:] for c in df_GMT_30.columns],
        axis=1,
    )
    if len(df_GMT_30[df_GMT_30.columns[dfbools.all()]].columns) == 0: # if there's no columns fully beneath upper line, grab least overlapping
        minfalsecol = df_GMT_30.columns[dfbools.sum(axis=0).idxmax()]
        df_GMT_30 = df_GMT_30.loc[:,minfalsecol]    
    else: # otherwise, get column with most max years in subset
        maxes = pd.concat(
            [df_GMT_30.loc[:,c]==df_GMT_30.max(axis=1) for c in df_GMT_30[df_GMT_30.columns[dfbools.all()]].columns],
            axis=1,
        )
        maxes.columns = df_GMT_30[df_GMT_30.columns[dfbools.all()]].columns
        df_GMT_30 = df_GMT_30[df_GMT_30.columns[dfbools.all()]].loc[:,maxes.sum(axis=0).idxmax()]
        
    # third line, NDC (going for 2.7)
    df_GMT_NDC = df_GMT_all[
        df_GMT_all.columns[(df_GMT_all.max(axis=0)<scens['NDC'][1])&(df_GMT_all.max(axis=0)>scens['NDC'][0])]
    ]
    dfbools=pd.concat(
        [df_GMT_NDC.loc[:,c]<=df_GMT_30.loc[:] for c in df_GMT_NDC.columns],
        axis=1,
    )
    if len(df_GMT_NDC[df_GMT_NDC.columns[dfbools.all()]].columns) == 0: # if there's no columns fully beneath upper line, grab least overlapping
        minfalsecol = df_GMT_NDC.columns[dfbools.sum(axis=0).idxmax()]
        df_GMT_NDC = df_GMT_NDC.loc[:,minfalsecol]    
    else: # otherwise, get column with most max years in subset
        maxes = pd.concat(
            [df_GMT_NDC.loc[:,c]==df_GMT_NDC.max(axis=1) for c in df_GMT_NDC[df_GMT_NDC.columns[dfbools.all()]].columns],
            axis=1,
        )
        maxes.columns = df_GMT_NDC[df_GMT_NDC.columns[dfbools.all()]].columns
        df_GMT_NDC = df_GMT_NDC[df_GMT_NDC.columns[dfbools.all()]].loc[:,maxes.sum(axis=0).idxmax()]

    # 2 degree scen
    df_GMT_20 = df_GMT_all[
        df_GMT_all.columns[(df_GMT_all.max(axis=0)<scens['2.0'][1])&(df_GMT_all.max(axis=0)>scens['2.0'][0])]
    ]
    dfbools=pd.concat(
        [df_GMT_20.loc[:,c]<=df_GMT_NDC.loc[:] for c in df_GMT_20.columns],
        axis=1,
    )
    if len(df_GMT_20[df_GMT_20.columns[dfbools.all()]].columns) == 0:
        minfalsecol = df_GMT_20.columns[dfbools.sum(axis=0).idxmax()]
        df_GMT_20 = df_GMT_20.loc[:,minfalsecol]
    else:    
        maxes = pd.concat(
            [df_GMT_20.loc[:,c]==df_GMT_20.max(axis=1) for c in df_GMT_20[df_GMT_20.columns[dfbools.all()]].columns],
            axis=1,
        )
        maxes.columns = df_GMT_20[df_GMT_20.columns[dfbools.all()]].columns
        df_GMT_20 = df_GMT_20[df_GMT_20.columns[dfbools.all()]].loc[:,maxes.sum(axis=0).idxmax()]    

    # 1.5 degree scen
    df_GMT_15 = df_GMT_all[
        df_GMT_all.columns[(df_GMT_all.max(axis=0)<scens['1.5'][1])&(df_GMT_all.max(axis=0)>scens['1.5'][0])]
    ]
    dfbools=pd.concat(
        [df_GMT_15.loc[:,c]<=df_GMT_20.loc[:] for c in df_GMT_15.columns],
        axis=1,
    )
    if len(df_GMT_15[df_GMT_15.columns[dfbools.all()]].columns) == 0:
        minfalsecol = df_GMT_15.columns[dfbools.sum(axis=0).idxmax()]
        df_GMT_15 = df_GMT_15.loc[:,minfalsecol]
    else:    
        maxes = pd.concat(
            [df_GMT_15.loc[:,c]==df_GMT_15.max(axis=1) for c in df_GMT_15[df_GMT_15.columns[dfbools.all()]].columns],
            axis=1,
        )
        maxes.columns = df_GMT_15[df_GMT_15.columns[dfbools.all()]].columns
        df_GMT_15 = df_GMT_15[df_GMT_15.columns[dfbools.all()]].loc[:,maxes.sum(axis=0).idxmax()]
    
    # lower bound
    mins = pd.concat(
            [df_GMT_all.loc[:,c]==df_GMT_all.min(axis=1) for c in df_GMT_all.columns],
            axis=1,
    )
    df_GMT_lb = df_GMT_all.loc[:,df_GMT_all.columns[mins.sum(axis=0).idxmax()]] 

    return df_GMT_lb, df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_30, df_GMT_40




#%%

#%%---------------------------------------------------------------#
# Load global mean temperature projections and build              #
# stylized trajectories                                           #
# ----------------------------------------------------------------#

def load_GMT(
    year_start,
    year_end,
    year_range,
    flags,
):

    """
    Creation of stylized GMT trajectories based on loaded pathways
    """

    # ---------------------------------------------------------- #
    # Definition of the 1.5, 2.0 and NDC trajectories from SR15  #
    # This is the original scenarios used in Thiery et al.(2021) #                                      
    # ---------------------------------------------------------- #

    # Luke's comment : (wim's original scenarios; will use historical obs years from here, 1960-1999, but replace with ar6 trajectories)
    df_GMT_SR15 = pd.read_excel(data_dir+'temperature_trajectories_SR15/GMT_50pc_manualoutput_4pathways.xlsx', header=1);
    df_GMT_SR15 = df_GMT_SR15.iloc[:4,1:].transpose().rename(columns={
        0 : 'IPCCSR15_IMAGE 3.0.1_SSP1-26_GAS',
        1 : 'IPCCSR15_MESSAGE-GLOBIOM 1.0_ADVANCE_INDC_GAS',
        2 : 'IPCCSR15_MESSAGE-GLOBIOM 1.0_SSP2-19_GAS',
        3 : 'IPCCSR15_MESSAGEix-GLOBIOM 1.0_LowEnergyDemand_GAS'
    })

    if np.nanmax(df_GMT_SR15.index) < year_end: 
        # repeat average of last 10 years (i.e. end-9 to end ==> 2090:2099)
        GMT_last_10ymean = df_GMT_SR15.iloc[-10:,:].mean()
        for year in range(np.nanmax(df_GMT_SR15.index),year_end+1): 
            df_GMT_SR15 = pd.concat([df_GMT_SR15, pd.DataFrame(GMT_last_10ymean).transpose().rename(index={0:year})])

    # cut to analysis years
    # currently using hist from this earlier version of df_GMT_15 (df_GMT_15 gets remade under flags['gmt'] == 'ar6')
    df_GMT_15 = df_GMT_SR15.loc[year_start:year_end,'IPCCSR15_MESSAGEix-GLOBIOM 1.0_LowEnergyDemand_GAS']
    df_GMT_20 = df_GMT_SR15.loc[year_start:year_end,'IPCCSR15_IMAGE 3.0.1_SSP1-26_GAS']
    df_GMT_NDC = df_GMT_SR15.loc[year_start:year_end,'IPCCSR15_MESSAGE-GLOBIOM 1.0_ADVANCE_INDC_GAS']

    # check and drop duplicate years
    df_GMT_15 = df_GMT_15[~df_GMT_15.index.duplicated(keep='first')]
    df_GMT_20 = df_GMT_20[~df_GMT_20.index.duplicated(keep='first')]
    df_GMT_NDC = df_GMT_NDC[~df_GMT_NDC.index.duplicated(keep='first')]
    df_GMT_SR15 = df_GMT_SR15[~df_GMT_SR15.index.duplicated(keep='first')]

    # ---------------------------------------------------------- #
    # Definition of the OverShoot (OS) and no-OverShoot (noOS)   #
    # trajectories from .mat object of Thiery et al.(2021)       #
    # ---------------------------------------------------------- # 

    from scipy.io import loadmat

    # Load GMT_OS
    mat_data = loadmat(scripts_dir + '/references/lifetime_exposure_wim/lifetime_exposure_wim_v1/GMT_OS.mat', squeeze_me=True)
    GMT_OS = mat_data['GMT_OS'].flatten()
    years = np.arange(1960, 1960 + len(GMT_OS))
    df_GMT_OS = pd.Series(GMT_OS, index=years)
    df_GMT_OS.name = None
    df_GMT_OS.index.name = None

    # Load GMT_noOS
    mat_data = loadmat(scripts_dir + '/references/lifetime_exposure_wim/lifetime_exposure_wim_v1/GMT_noOS.mat', squeeze_me=True)
    GMT_noOS = mat_data['GMT_noOS'].flatten()
    years = np.arange(1960, 1960 + len(GMT_noOS))
    df_GMT_noOS = pd.Series(GMT_noOS, index=years)
    df_GMT_noOS.name = None
    df_GMT_noOS.index.name = None

    # ---------------------------------------------------------- #
    # Definition of the Stress Test Scenarios (STS)              #
    # by the SPARCCLE project                                    #
    # ---------------------------------------------------------- #

    # Open the NetCDF file
    ds_GMT_STS = xr.open_dataset(data_dir + '/temperature_trajectories_STS/GSAT_FaIR_SPARCCLE_STSv1.nc', engine='netcdf4')

    # ---------------------------------------------------------- #
    # Definition of stylized trajectories used in the BE         #
    # The definition of these trajectories depends on the value  #
    # of the flags['gmt'] to either used the 'original'          #
    # trajectories defined in Thiery et al.(2021) or the update  #
    # based on AR6 by Grant et al.(2025)                         #                                                                           
    # ---------------------------------------------------------- #

    if flags['gmt'] == 'original':
    
        GMT_max = 3.5
        GMT_fut_strtyr = int(df_GMT_15.index.where(df_GMT_15==df_GMT_20).max())+1
        ind_fut_strtyr = int(np.argwhere(np.asarray(df_GMT_15.index)==GMT_fut_strtyr))
        GMT_min = df_GMT_15.loc[GMT_fut_strtyr-1]
        GMT_steps = np.arange(0,GMT_max+GMT_inc/2,GMT_inc)
        GMT_steps = np.insert(GMT_steps[np.where(GMT_steps>GMT_min)],0,GMT_min)
        n_steps = len(GMT_steps)
        ind_15 = np.argmin(np.abs(GMT_steps-df_GMT_15.iloc[-1]))
        ind_20 = np.argmin(np.abs(GMT_steps-df_GMT_20.iloc[-1]))
        ind_NDC = np.argmin(np.abs(GMT_steps-df_GMT_NDC.iloc[-1]))
        n_years = len(year_range)
        trj = np.empty((n_years,n_steps))
        trj.fill(np.nan)
        trj[0:ind_fut_strtyr,:] = np.repeat(np.expand_dims(df_GMT_15.loc[:GMT_fut_strtyr-1].values,axis=1),n_steps,axis=1)
        trj[ind_fut_strtyr:,0] = GMT_min
        trj[ind_fut_strtyr:,-1] = np.interp(
            x=year_range[ind_fut_strtyr:],
            xp=[GMT_fut_strtyr,year_end],
            fp=[GMT_min,GMT_max],
        )
        trj[:,ind_15] = df_GMT_15.values
        trj[:,ind_20] = df_GMT_20.values
        trj[:,ind_NDC] = df_GMT_NDC.values
        trj_msk = np.ma.masked_invalid(trj)
        [xx, yy] = np.meshgrid(range(n_steps),range(n_years))
        x1 = xx[~trj_msk.mask]
        y1 = yy[~trj_msk.mask]
        trj_interpd = interpolate.griddata(
            (x1,y1), # only include coords with valid data
            trj[~trj_msk.mask].ravel(), # inputs are valid only, too
            (xx,yy), # then provide coordinates of ourput array, which include points where interp is required (not ravelled, so has 154x24 shape)
        )
        df_GMT_strj = pd.DataFrame(
            trj_interpd, 
            columns=range(n_steps), 
            index=year_range,
        )
        
    elif flags['gmt'] == 'ar6':
        
        # for alternative gmt mapping approaches, collect new ar6 scens from IASA explorer
        df_GMT_ar6 = pd.read_csv(data_dir+'temperature_trajectories_AR6/ar6_c1_c7_nogaps_2000-2100.csv',header=0)
        df_GMT_ar6.loc[:,'Model'] = df_GMT_ar6.loc[:,'Model']+'_'+df_GMT_ar6.loc[:,'Scenario']
        df_GMT_ar6 = df_GMT_ar6.drop(columns=['Scenario','Region','Variable','Unit']).transpose()
        df_GMT_ar6.columns=df_GMT_ar6.loc['Model',:]
        df_GMT_ar6.columns.name = None
        df_GMT_ar6 = df_GMT_ar6.drop(df_GMT_ar6.index[0])
        df_GMT_ar6 = df_GMT_ar6.dropna(axis=1)
        df_GMT_ar6.index = df_GMT_ar6.index.astype(int)
        df_hist_all = df_GMT_15.loc[1960:1999]
        df_hist_all = pd.concat([df_hist_all for i in range(len(df_GMT_ar6.columns))],axis=1)
        df_hist_all.columns = df_GMT_ar6.columns
        df_GMT_ar6 = pd.concat([df_hist_all,df_GMT_ar6],axis=0) # add historical values to additional scenarios
        
        if np.nanmax(df_GMT_ar6.index) < year_end: 
            # repeat average of last 10 years (i.e. end-9 to end ==> 2090:2099)
            GMT_last_10ymean = df_GMT_ar6.iloc[-10:,:].mean()
            for year in range(np.nanmax(df_GMT_ar6.index),year_end+1): 
                df_GMT_ar6 = pd.concat([df_GMT_ar6, pd.DataFrame(GMT_last_10ymean).transpose().rename(index={0:year})]) 
                
        # drop dups
        df_GMT_ar6 = df_GMT_ar6[~df_GMT_ar6.index.duplicated(keep='first')]

        # get new trajects
        df_GMT_lb, df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_30, df_GMT_40 = ar6_scen_grab(
            scen_thresholds,
            df_GMT_ar6,
        )        
        
        # GMT_max = df_GMT_40.loc[2100]
        GMT_max = df_GMT_40.iloc[-1]
        GMT_fut_strtyr = int(df_GMT_15.index.where(df_GMT_15==df_GMT_20).max())+1
        ind_fut_strtyr = int(np.argwhere(np.asarray(df_GMT_15.index)==GMT_fut_strtyr))
        GMT_min = df_GMT_lb.loc[GMT_fut_strtyr-1]
        GMT_steps = np.arange(0,GMT_max+0.05,GMT_inc)
        GMT_steps = np.insert(GMT_steps[np.where(GMT_steps>GMT_min)],0,GMT_min)
        n_steps = len(GMT_steps)
        ind_lb = np.argmin(np.abs(GMT_steps-df_GMT_lb.iloc[-1]))
        ind_15 = np.argmin(np.abs(GMT_steps-df_GMT_15.iloc[-1]))
        ind_20 = np.argmin(np.abs(GMT_steps-df_GMT_20.iloc[-1]))
        ind_NDC = np.argmin(np.abs(GMT_steps-df_GMT_NDC.iloc[-1]))
        ind_30 = np.argmin(np.abs(GMT_steps-df_GMT_30.iloc[-1]))
        ind_40 = np.argmin(np.abs(GMT_steps-df_GMT_40.iloc[-1]))
        indices=[ind_lb,ind_15,ind_20,ind_NDC,ind_30,ind_40]
        # year_range=np.arange(1960,2100+1)
        n_years = len(year_range)
        trj = np.empty((n_years,n_steps))
        trj.fill(np.nan)
        trj[0:ind_fut_strtyr,:] = np.repeat(np.expand_dims(df_GMT_15.loc[:GMT_fut_strtyr-1].values,axis=1),n_steps,axis=1)
        trj[ind_fut_strtyr:,0] = GMT_min
        trj[ind_fut_strtyr:,-1] = np.interp(
            x=year_range[ind_fut_strtyr:],
            xp=[GMT_fut_strtyr,year_end],
            fp=[GMT_min,GMT_max],
        )
        trj[:,ind_lb] = df_GMT_lb.values
        trj[:,ind_15] = df_GMT_15.values
        trj[:,ind_20] = df_GMT_20.values
        trj[:,ind_NDC] = df_GMT_NDC.values
        trj[:,ind_30] = df_GMT_30.values
        trj[:,ind_40] = df_GMT_40.values
        trj_msk = np.ma.masked_invalid(trj)
        [xx, yy] = np.meshgrid(range(n_steps),range(n_years))
        x1 = xx[~trj_msk.mask]
        y1 = yy[~trj_msk.mask]
        trj_interpd = interpolate.griddata(
            (x1,y1), # only include coords with valid data
            trj[~trj_msk.mask].ravel(), # inputs are valid only, too
            (xx,yy), # then provide coordinates of ourput array, which include points where interp is required (not ravelled, so has 154x24 shape)
        )
        df_GMT_strj = pd.DataFrame(
            trj_interpd, 
            columns=range(n_steps), 
            index=year_range,
        )
        
    elif flags['gmt'] == 'ar6_new':
        
        # ------------------------- This is original AR6 approach --------------------------
        # for alternative gmt mapping approaches, collect new ar6 scens from IASA explorer
        df_GMT_ar6 = pd.read_csv(data_dir+'temperature_trajectories_AR6/ar6_c1_c7_nogaps_2000-2100.csv',header=0)
        df_GMT_ar6.loc[:,'Model'] = df_GMT_ar6.loc[:,'Model']+'_'+df_GMT_ar6.loc[:,'Scenario']
        df_GMT_ar6 = df_GMT_ar6.drop(columns=['Scenario','Region','Variable','Unit']).transpose()
        df_GMT_ar6.columns=df_GMT_ar6.loc['Model',:]
        df_GMT_ar6.columns.name = None
        df_GMT_ar6 = df_GMT_ar6.drop(df_GMT_ar6.index[0])
        df_GMT_ar6 = df_GMT_ar6.dropna(axis=1)
        df_GMT_ar6.index = df_GMT_ar6.index.astype(int)
        df_hist_all = df_GMT_15.loc[1960:1999]
        df_hist_all = pd.concat([df_hist_all for i in range(len(df_GMT_ar6.columns))],axis=1)
        df_hist_all.columns = df_GMT_ar6.columns
        df_GMT_ar6 = pd.concat([df_hist_all,df_GMT_ar6],axis=0) # add historical values to additional scenarios
        
        # moved the extrapolation past 2100 to after creation of stylized traj 
        
        # if np.nanmax(df_GMT_ar6.index) < year_end: 
            
        #     if flags['gmt_extend'] == '10yrmean':
        #         # repeat average of last 10 years (i.e. end-9 to end ==> 2090:2099) # ORIGINAL
        #         # Rosa: not great, better to do this overwrite directly of the stylized trajectory at the end?
        #         GMT_last_10ymean = df_GMT_ar6.iloc[-10:,:].mean()
        #         for year in range(np.nanmax(df_GMT_ar6.index),year_end+1): 
        #             df_GMT_ar6 = pd.concat([df_GMT_ar6, pd.DataFrame(GMT_last_10ymean).transpose().rename(index={0:year})]) 
            
        #     elif flags['gmt_extend'] == 'lastyear':
        #         # ROSA: modify so the last single year value is repeated, not the last 10 years mean
        #         GMT_last_year = df_GMT_ar6.iloc[-1,:]    
        #         for year in range(np.nanmax(df_GMT_ar6.index),year_end+1): 
        #             df_GMT_last_year = pd.DataFrame(GMT_last_year).transpose()
        #             df_GMT_ar6 = pd.concat([df_GMT_ar6, df_GMT_last_year.rename(index={df_GMT_last_year.index[0]:year})]) 
        
        #     # ROSA testing: extrapolate last 10 year trend of each scenario
        #     elif flags['gmt_extend'] == '10yrtrend': 
        #         max_year = df_GMT_ar6.index.max()
        #         yrs = np.arange(max_year - 9, max_year + 1)
        #         trend = {s: np.polyfit(yrs, df_GMT_ar6.loc[yrs, s].astype(float), 1) for s in df_GMT_ar6.columns}
        #         future = {
        #             y: {s: np.polyval(trend[s], y) for s in df_GMT_ar6.columns}
        #             for y in range(max_year + 1, year_end + 1)
        #         }
        #         df_GMT_ar6 = pd.concat([df_GMT_ar6, pd.DataFrame.from_dict(future, orient="index")])
       
        # drop dups
        df_GMT_ar6 = df_GMT_ar6[~df_GMT_ar6.index.duplicated(keep='first')]

        # get new trajects
        df_GMT_lb, df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_30, df_GMT_40 = ar6_scen_grab(
            scen_thresholds,
            df_GMT_ar6,
        )        
        
        # GMT_max = df_GMT_40.loc[2100]
        GMT_max = df_GMT_40.iloc[-1]
        GMT_fut_strtyr = int(df_GMT_15.index.where(df_GMT_15==df_GMT_20).max())+1
        ind_fut_strtyr = int(np.argwhere(np.asarray(df_GMT_15.index)==GMT_fut_strtyr))
        GMT_min = df_GMT_lb.loc[GMT_fut_strtyr-1]
        GMT_steps = np.arange(0,GMT_max+0.05,GMT_inc)
        GMT_steps = np.insert(GMT_steps[np.where(GMT_steps>GMT_min)],0,GMT_min)
        n_steps = len(GMT_steps)
        ind_lb = np.argmin(np.abs(GMT_steps-df_GMT_lb.iloc[-1]))
        ind_15 = np.argmin(np.abs(GMT_steps-df_GMT_15.iloc[-1]))
        ind_20 = np.argmin(np.abs(GMT_steps-df_GMT_20.iloc[-1]))
        ind_NDC = np.argmin(np.abs(GMT_steps-df_GMT_NDC.iloc[-1]))
        ind_30 = np.argmin(np.abs(GMT_steps-df_GMT_30.iloc[-1]))
        ind_40 = np.argmin(np.abs(GMT_steps-df_GMT_40.iloc[-1]))
        indices=[ind_lb,ind_15,ind_20,ind_NDC,ind_30,ind_40]
        year_range=np.arange(1960,2100+1) # ROSA Testing 
        n_years = len(year_range)
        trj = np.empty((n_years,n_steps))
        trj.fill(np.nan)
        trj[0:ind_fut_strtyr,:] = np.repeat(np.expand_dims(df_GMT_15.loc[:GMT_fut_strtyr-1].values,axis=1),n_steps,axis=1)
        trj[ind_fut_strtyr:,0] = GMT_min
        trj[ind_fut_strtyr:,-1] = np.interp(
            x=year_range[ind_fut_strtyr:],
            xp=[GMT_fut_strtyr,year_range[-1]], # was year_end
            fp=[GMT_min,GMT_max],
        )
        trj[:,ind_lb] = df_GMT_lb.values
        trj[:,ind_15] = df_GMT_15.values
        trj[:,ind_20] = df_GMT_20.values
        trj[:,ind_NDC] = df_GMT_NDC.values
        trj[:,ind_30] = df_GMT_30.values
        trj[:,ind_40] = df_GMT_40.values
        trj_msk = np.ma.masked_invalid(trj)
        [xx, yy] = np.meshgrid(range(n_steps),range(n_years))
        x1 = xx[~trj_msk.mask]
        y1 = yy[~trj_msk.mask]
        trj_interpd = interpolate.griddata(
            (x1,y1), # only include coords with valid data
            trj[~trj_msk.mask].ravel(), # inputs are valid only, too
            (xx,yy), # then provide coordinates of ourput array, which include points where interp is required (not ravelled, so has 154x24 shape)
        )
        df_GMT_strj = pd.DataFrame(
            trj_interpd, 
            columns=range(n_steps), 
            index=year_range,
        )        


        # ------------------------- End of original AR6 approach --------------------------
        
        # ORIGINAL 

        # # Below we adapt for clean, 0.1 deg intervals between only 1.5 to 3.5 to speed up analysis
        # df_GMT_strj
        # GMT_min=1.5
        # GMT_max=3.5
        # GMT_steps = np.arange(GMT_min,GMT_max+0.05,GMT_inc)
        # n_steps = len(GMT_steps)
        # n_years = len(year_range)
        # trj = np.empty((n_years,n_steps))
        # trj.fill(np.nan)

        # GMT_fut_strtyr = int(df_GMT_15.index.where(df_GMT_15==df_GMT_20).max())+1
        # ind_fut_strtyr = int(np.argwhere(np.asarray(df_GMT_15.index)==GMT_fut_strtyr))

        # # new 1.5 degree as avg between pathways that hit 1.44 and 1.55 at 2100 and then fix 2100 year
        # df_GMT_15_new = df_GMT_strj.loc[:,5:6].mean(axis=1)
        # df_GMT_15_new[2100] = GMT_min

        # # new 3.5 degree
        # df_GMT_35_new = df_GMT_strj.loc[:,24]
        # df_GMT_35_new[2100] = GMT_max

        # trj[0:ind_fut_strtyr,:] = np.repeat(np.expand_dims(df_GMT_15.loc[:GMT_fut_strtyr-1].values,axis=1),n_steps,axis=1)
        # trj[:,0] = df_GMT_15_new
        # trj[:,-1] = df_GMT_35_new

        # trj_msk_new = np.ma.masked_invalid(trj)
        # [xx, yy] = np.meshgrid(range(n_steps),range(n_years))
        # x1 = xx[~trj_msk_new.mask]
        # y1 = yy[~trj_msk_new.mask]
        # trj_interpd_new = interpolate.griddata(
        #     (x1,y1), # only include coords with valid data
        #     trj[~trj_msk_new.mask].ravel(), # inputs are valid only, too
        #     (xx,yy), # then provide coordinates of ourput array, which include points where interp is required (not ravelled, so has 154x24 shape)
        # )
        # df_GMT_strj_new = pd.DataFrame(
        #     trj_interpd_new, 
        #     columns=range(n_steps), 
        #     index=year_range,
        # )       
        # df_GMT_strj = cp(df_GMT_strj_new) 

        
        # ------------------------- NEW ROSA --------------------------

        # # Rosa: note that forcing the GMT in 2100 causes jump!! discontinuity... why dont we just stick to original approach, remove the pathways we dont want 
        # # and call the one closest to 1.5 the 1.5 pathway? or i am sure there is a way to interpolate based on the temperature in 2100

        # modified it to fix this !! Below

        # Desired GMT range and steps
        GMT_min = 1.5
        GMT_max = 3.5
        GMT_steps = np.round(np.arange(GMT_min, GMT_max + 0.001, 0.1), 2)
        n_years = len(df_GMT_strj)

        # Extract the GMT values corresponding to each original column in df_GMT_strj
        # Suppose the original GMT levels are known — e.g.,:
        orig_GMT_levels = GMT_steps_all = np.round(df_GMT_strj.loc[2100].values, 2)  # or however you originally set them

        # Filter to retain only the columns within the 1.5–3.5°C range
        keep_cols = np.where((orig_GMT_levels >= GMT_min) & (orig_GMT_levels <= GMT_max))[0]
        df_GMT_strj_filtered = df_GMT_strj.iloc[:, keep_cols]
        orig_GMT_levels_filtered = orig_GMT_levels[keep_cols]

        # Interpolate across GMT dimension for each year
        df_GMT_strj_clean = pd.DataFrame(index=df_GMT_strj.index, columns=GMT_steps)
        for yr in df_GMT_strj.index:
            yvals = df_GMT_strj_filtered.loc[yr].values
            df_GMT_strj_clean.loc[yr] = np.interp(GMT_steps, orig_GMT_levels_filtered, yvals)

        # Now df_GMT_strj_clean has smooth trajectories that end at 1.5–3.5°C in 2100

        df_GMT_strj = cp(df_GMT_strj_clean)  

        # ROSA: Do the extrapolation past 2100 here after construction of stylized trajectories 
        # instead of on the original AR6 scenarios - better for 10yrtrend 

        df = df_GMT_strj
        if np.nanmax(df.index) < year_end: 
            
            if flags['gmt_extend'] == '10yrmean':
                # repeat average of last 10 years (i.e. end-9 to end ==> 2090:2099) # ORIGINAL
                # Rosa: not great, better to do this overwrite directly of the stylized trajectory at the end?
                GMT_last_10ymean = df.iloc[-10:,:].mean()
                for year in range(np.nanmax(df.index),year_end+1): 
                    df = pd.concat([df, pd.DataFrame(GMT_last_10ymean).transpose().rename(index={0:year})]) 
            
            elif flags['gmt_extend'] == 'lastyear':
                # ROSA: modify so the last single year value is repeated, not the last 10 years mean
                GMT_last_year = df.iloc[-1,:]    
                for year in range(np.nanmax(df.index),year_end+1): 
                    df_GMT_last_year = pd.DataFrame(GMT_last_year).transpose()
                    df = pd.concat([df, df_GMT_last_year.rename(index={df_GMT_last_year.index[0]:year})]) 
        
            # ROSA testing: extrapolate last 10 year trend of each scenario
            elif flags['gmt_extend'] == '10yrtrend': 
                max_year = df.index.max()
                yrs = np.arange(max_year - 9, max_year + 1)
                trend = {s: np.polyfit(yrs, df.loc[yrs, s].astype(float), 1) for s in df.columns}
                future = {
                    y: {s: np.polyval(trend[s], y) for s in df.columns}
                    for y in range(max_year + 1, year_end + 1)
                }
                df = pd.concat([df, pd.DataFrame.from_dict(future, orient="index")])


        df_GMT_strj = cp(df)
        


#     # pickles GMT #

#     if flags['gmt']=='ar6_new':

#         pass # Rosa commented out for permissions - could save it in my folder as pickle or netcdf

#         # with open(data_dir+'temperature_trajectories_AR6/df_GMT_strj.pkl', 'wb') as f:
#         #     pass # ROSA pass
#         #     #pk.dump(df_GMT_strj,f)

#         # with open(data_dir+'temperature_trajectories_STS/ds_GMT_STS.pkl', 'wb') as f:
#         #     pass # ROSA pass
#         #     #ds_GMT_STS.to_netcdf(data_dir+'temperature_trajectories_STS/ds_GMT_STS.nc')

#     if flags['gmt']=='original':

#         with open(data_dir+'temperature_trajectories_SR15/df_GMT_15.pkl', 'wb') as f:
#             pk.dump(df_GMT_15,f)

#         with open(data_dir+'temperature_trajectories_SR15/df_GMT_20.pkl', 'wb') as f:
#             pk.dump(df_GMT_20,f)

#         with open(data_dir+'temperature_trajectories_SR15/df_GMT_NDC.pkl', 'wb') as f:
#             pk.dump(df_GMT_NDC,f)
        
#         with open(data_dir+'temperature_trajectories_SR15/df_GMT_OS.pkl', 'wb') as f:
#             pk.dump(df_GMT_OS,f)
        
#         with open(data_dir+'temperature_trajectories_UVIC/df_GMT_noOS.pkl', 'wb') as f:
#             pk.dump(df_GMT_noOS,f)

#         with open(data_dir+'temperature_trajectories_UVIC/df_GMT_strj.pkl', 'wb') as f:
#             pk.dump(df_GMT_strj,f)

    return df_GMT_15, df_GMT_20, df_GMT_NDC, df_GMT_OS, df_GMT_noOS, ds_GMT_STS, df_GMT_strj
