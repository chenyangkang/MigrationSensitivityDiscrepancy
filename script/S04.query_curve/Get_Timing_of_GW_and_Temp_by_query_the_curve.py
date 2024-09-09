# %%

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
import h3pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle
import re
import sys
from functools import partial
from pygam import s,f,l,LinearGAM
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings('ignore')
tqdm.pandas()


# %%
# trophic_niche = str(sys.argv[1])#'all'#str(sys.argv[1]) #'all' #
# A_=int(sys.argv[2])
# B_=int(sys.argv[3])
# C_=int(sys.argv[4])
# D_=int(sys.argv[5])

trophic_niche = str(sys.argv[1])#'all'
# A_=10
# B_=40
# C_=20
# D_=50


# %% [markdown]
# ## 01.Read birdwave data

# %%
# # 01. Read Data
all_birdwave_data = pd.read_csv(f'../../data/D03.smoothed_peaks/all_smoothed_peaks_spring.csv')
all_birdwave_data = all_birdwave_data[all_birdwave_data['niche_or_level']==trophic_niche]
if 'geometry' in all_birdwave_data.columns:
    del all_birdwave_data['geometry']
    

# %%
all_birdwave_data


# %%
BW_peak_each_year = all_birdwave_data.copy()


# %% [markdown]
# ## 02.Read greenwave data

# %%
## Read Greenwave data
GW = []

for year in tqdm(range(2010,2021)):
    try:
        dat = pd.read_csv(f'../../data/D04.environmental_data/GreenWave_agg_h3_02_{year}.csv')
        dat['year'] = year
        GW.append(dat)

    except:
        continue
        
GW = pd.concat(GW, axis=0).reset_index(drop=True)

## calculate delta NDVI
def calc_delta_NDVI(df):
    df['delta_NDVI'] = np.gradient(df['mean_NDVI'].values)
    return df

GW = GW.groupby('h3_02').progress_apply(calc_delta_NDVI).reset_index(drop=True)



# %% [markdown]
# ## 03.Read temperature data

# %%
## Read Temperature data
temp = []

for year in tqdm(range(2010,2021)):
    try:
        dat = pd.read_csv(f'../../data/D04.environmental_data/h3_02_temperature_{year}.csv')
        dat['year'] = year
        temp.append(dat)

    except:
        continue
      
temp = pd.concat(temp, axis=0).reset_index(drop=True)

def calc_delta_Temp(df):
    df['delta_tmean'] = np.gradient(df['tmean'].values)
    return df

temp = temp.groupby('h3_02').progress_apply(calc_delta_Temp).reset_index(drop=True)


# %% [markdown]
# ## 04.Merge env data with birdwave peak

# %%
GW = GW.merge(
    BW_peak_each_year[['h3_02','mean_DOY_max', 'std_DOY_max', 'mean_DOY_peak', 'std_DOY_peak',
                   'source_model', 'season', 'year','niche_or_level','mean_seasonality',
                   'mu_ARR','std_ARR']], 
    on=['h3_02','year'], how='left'
)

temp = temp.merge(
    BW_peak_each_year[['h3_02','mean_DOY_max', 'std_DOY_max', 'mean_DOY_peak', 'std_DOY_peak',
                   'source_model', 'season', 'year','niche_or_level','mean_seasonality',
                   'mu_ARR','std_ARR']], 
    on=['h3_02','year'], how='left'
)


# %%
GW

# %%
BW_peak_each_year

# %% [markdown]
# ## 05.Define functions

# %%

def get_expected_env_value_at_arrival(df, var_, DOY_variable = 'mu_ARR_int'):
    try:
        mean_env_value_at_arrival = df[df['DOY']==df[DOY_variable]][var_].mean()
        return mean_env_value_at_arrival
        
    except Exception as e:
        print(e)
        return None
        
        
def get_exp_ARR_based_on_env(df, env_var, expected_env_var, DOY_variable = 'mu_ARR_int'):
    df = df.sort_values(by='DOY')
    target = df[expected_env_var].iloc[0]
    env_val = df[env_var].values
    mean_actual_arrival_date = np.nanmean(df[DOY_variable])
    
    candidates = []
    
    for i in range(len(env_val)-1):
        if (env_val[i] - target) * (env_val[i+1] - target)<0:
            if (i+1>=mean_actual_arrival_date-30) & (i+1<=mean_actual_arrival_date+30): #within 30 days

                if i+1>=183:
                    continue
                elif i+1<=1:
                    continue
                    
                candidates.append(i+1)
                
    if len(candidates)==0:
        return None
    else:
        the_expected_doy = candidates[0]
        return the_expected_doy

    

# %% [markdown]
# ## 06. Get GreenTrace Index

# %%
for var_ in ['mean_NDVI','delta_NDVI']:
    
    res_list = []
    for uncertainty_sample in tqdm(range(100)):
        
        tmp = GW[GW['season']=='spring']
        tmp['mu_ARR_int'] = np.random.normal(loc=tmp['mu_ARR'], scale=tmp['std_ARR'])
        tmp['mu_ARR_int'] = tmp['mu_ARR_int'].astype('int')
        
        expected_env_value = tmp.groupby('h3_02').progress_apply(partial(get_expected_env_value_at_arrival, var_=var_))
        expected_env_value = expected_env_value.reset_index(drop=False).rename(columns={0:f'expected_{var_}'})
        tmp = tmp.merge(expected_env_value[['h3_02',f'expected_{var_}']], how='left', on='h3_02')
        
        res = tmp.groupby(['h3_02','year']).progress_apply(partial(get_exp_ARR_based_on_env,
                                                            env_var=var_, 
                                                            expected_env_var = f'expected_{var_}'))
        res = res.reset_index(drop=False).rename(columns={0:f'expected_trace_by_{var_}'})
        res['season'] = 'spring'

        res_list.append(res)
        
    res_list = pd.concat(res_list, axis=0)
    res = res_list.groupby(['h3_02','year','season']).agg(
        mean_expected_trace=(f'expected_trace_by_{var_}', np.nanmean), 
        std_expected_trace=(f'expected_trace_by_{var_}', np.nanstd)
        ).reset_index(drop=False)
    res = res.rename(columns={f'mean_expected_trace':f'mean_expected_trace_by_{var_}',
                        f'std_expected_trace':f'std_expected_trace_by_{var_}'})
        
    del res_list
    
    all_birdwave_data = all_birdwave_data.merge(
        res, on=['h3_02','year','season'], how='left'
    )

    

# %%
all_birdwave_data


# %% [markdown]
# ## 07. Get ThermoTrace Index

# %%
for var_ in ['tmean','tmax','tmin']:
    
    res_list = []
    for uncertainty_sample in tqdm(range(100)):
        
        tmp = temp[temp['season']=='spring']
        tmp['mu_ARR_int'] = np.random.normal(loc=tmp['mu_ARR'], scale=tmp['std_ARR'])
        tmp['mu_ARR_int'] = tmp['mu_ARR_int'].astype('int')
        
        expected_env_value = tmp.groupby('h3_02').progress_apply(partial(get_expected_env_value_at_arrival, var_=var_))
        expected_env_value = expected_env_value.reset_index(drop=False).rename(columns={0:f'expected_{var_}'})
        tmp = tmp.merge(expected_env_value[['h3_02',f'expected_{var_}']], how='left', on='h3_02')
        
        res = tmp.groupby(['h3_02','year']).progress_apply(partial(get_exp_ARR_based_on_env,
                                                            env_var=var_, 
                                                            expected_env_var = f'expected_{var_}'))
        res = res.reset_index(drop=False).rename(columns={0:f'expected_trace_by_{var_}'})
        res['season'] = 'spring'

        res_list.append(res)
        
    res_list = pd.concat(res_list, axis=0)
    res = res_list.groupby(['h3_02','year','season']).agg(
        mean_expected_trace=(f'expected_trace_by_{var_}', np.nanmean), 
        std_expected_trace=(f'expected_trace_by_{var_}', np.nanstd)
        ).reset_index(drop=False)
    res = res.rename(columns={f'mean_expected_trace':f'mean_expected_trace_by_{var_}',
                        f'std_expected_trace':f'std_expected_trace_by_{var_}'})
        
    del res_list
    
    all_birdwave_data = all_birdwave_data.merge(
        res, on=['h3_02','year','season'], how='left'
    )

    

# %%
# all_birdwave_data.iloc[:,-12:].corr(method='spearman')


# %%
all_birdwave_data
print(all_birdwave_data.isnull().sum()/all_birdwave_data.shape[0])
all_birdwave_data.to_csv(f'../../data/D05.greentrace_thermotrace/{trophic_niche}_curve_query.csv', index=False)



