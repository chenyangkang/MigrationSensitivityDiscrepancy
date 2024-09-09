#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
import numpy
import math
import os
import warnings
import pickle
import time
import statsmodels.api as sm
from tqdm.auto import tqdm
import random
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from matplotlib import cm
import datetime
import re
import h3pandas
import json

##
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['PROJ_LIB'] = r'/usr/proj80/share/proj'

os.environ['GDAL_DATA'] = r'/beegfs/store4/chenyangkang/miniconda3/share'

warnings.filterwarnings("ignore")


# In[2]:


# for stemflow
from stemflow.model.AdaSTEM import AdaSTEM, AdaSTEMClassifier, AdaSTEMRegressor
from xgboost import XGBClassifier, XGBRegressor
from stemflow.model.Hurdle import Hurdle_for_AdaSTEM, Hurdle

from stemflow.model_selection import ST_train_test_split, ST_CV
from stemflow.utils.plot_gif import make_sample_gif


# In[3]:


scale='three_times'
type_='World'
import pickle
import gc


# In[4]:


tqdm.pandas(desc="Process: ")
from utiles.filter_data_script import *
from utiles.get_variables_dict import *


# In[5]:

type_ = 'World'
tro = str(sys.argv[1])
Di = str(sys.argv[2])
sp1 = sp = Di
year = int(sys.argv[3])
year_list = [year]

#P_lower_ = 10
#P_upper_ = 100
#P_step_ = 30
#P_bin_ = 80

P_lower_ = int(sys.argv[4])
P_upper_ = int(sys.argv[5])
P_step_ = int(sys.argv[6])
P_bin_ = int(sys.argv[7])

try:
    bb = sys.argv[8]
    if bb=='True':
        load_=True
    else:
        load_=False
except:
    load_=False

# In[6]:


x_names = get_variables_all()
x_names = x_names + ['mean_GM','std_GM','gradient_x_GM','gradient_y_GM']
# year_list = list(range(2015,2021))[::-1]
#year_list = [2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010]


# In[21]:


x_names = [i for i in x_names if not i in ['prec',
                                             'tmax',
                                             'tmin','year','month','week']]


def get_all_year_day_of_year_df(year):
    start_day = datetime.datetime.strptime(f'{year}-01-01','%Y-%m-%d')
    all_day_list = [start_day+datetime.timedelta(days=p) for p in range(366)]
    all_day_df = pd.DataFrame({'day':all_day_list})
    all_day_df['month'] = all_day_df.day.dt.month
    all_day_df['week'] = all_day_df.day.dt.isocalendar().week
    all_day_df['day_of_week'] = all_day_df.day.dt.day_of_week
    all_day_df['year'] = all_day_df.day.dt.year
    all_day_df = all_day_df[(all_day_df.year==year)]
    return all_day_df


# In[23]:


def pred_one_batch_day(model, pred_set, date_list, x_names):
    
    pred_set_list = []
    
    for date in date_list:
        ### add date
        pred_set['date'] = date

        ### if not doy in bins, continue
        the_doy = pred_set['date'].dt.day_of_year.values[0]

        ##### define pred set
        pred_set['duration_minutes'] = 60
        pred_set['effort_distance_km'] = 1
        pred_set['number_observers'] = 1
        pred_set['time_observation_started_minute_of_day']=420
        pred_set['obsvr_species_count']=1000
        pred_set['Traveling'] = 1
        pred_set['Stationary'] = 0
        pred_set['Area'] = 0
        pred_set['date'] = pd.to_datetime(pred_set['date'])
        pred_set['DOY'] = pred_set['date'].dt.day_of_year
        pred_set['month'] = pred_set['date'].dt.month
        pred_set['year'] = pred_set['date'].dt.year
        pred_set['week'] = pred_set['date'].dt.isocalendar().week

        pred_set = pred_set.reset_index(drop=True)
        pred_set_list.append(pred_set.copy())
     
    pred_set_list = pd.concat(pred_set_list, axis=0)
        
    pred_mean, pred_std = model.predict(pred_set_list[x_names], verbosity=0, return_std=True)
    pred_set_list['pred_mean'] = pred_mean
    pred_set_list['pred_std'] = pred_std
    
    df = pred_set_list[['h3_02','h3_05','h3_lng','h3_lat','longitude','latitude','elevation_mean',
                                       'year','month','week','DOY','date','pred_mean','pred_std']]
 
    return df


# In[49]:


class batch_generator():
    def __init__(self, wv, dv, batch_size=20):
        assert len(wv) == len(dv)
        self.batch_size = batch_size
        self.n = len(wv)
        self.wv = wv
        self.dv = dv
        self.count_ = 0
        self.current_wv_list = []
        self.current_dv_list = []

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.count_ >= self.n:
                if not len(self.current_wv_list)==0:
                    a = self.current_wv_list.copy()
                    b = self.current_dv_list.copy()
                    self.current_wv_list = []
                    self.current_dv_list = []
                    return (a, b)
                else:
                    raise StopIteration()
        
            if len(self.current_wv_list) < self.batch_size:
                self.current_wv_list.append(self.wv[self.count_])
                self.current_dv_list.append(self.dv[self.count_])
                self.count_+=1

            else:
                a = self.current_wv_list.copy()
                b = self.current_dv_list.copy()
                self.current_wv_list = []
                self.current_dv_list = []
                return (a, b)


# In[50]:


def predict_days(model,pred_set,year,sp1,pred_saving_path,x_names, batch=20):
    
    all_day_df = get_all_year_day_of_year_df(year)
    
    wv = all_day_df.week.values
    dv = all_day_df.day.values
    batch_generator_ = batch_generator(wv, dv, batch_size=batch)
    
    pred_dict = {sim:[] for sim in range(10)}
    for w_list,date_list in tqdm(batch_generator_, total=len(wv)//batch_generator_.batch_size):
        pred = pred_one_batch_day(model, pred_set, date_list, x_names)
        DOY = pred['DOY'].iloc[0]
        pred[['h3_05','year','elevation_mean','DOY','pred_mean','pred_std']].to_csv(os.path.join(pred_saving_path, 'raw', f'raw_pred_{sp1}_{year}_{tro}_DOY{DOY}_{P_lower_}_{P_upper_}_{P_step_}_{P_bin_}.csv'))
        

    return None


# In[51]:


def declare_adastem_model():
    model_hurdle_in_Ada = AdaSTEMRegressor(
        base_model=XGBRegressor(tree_method='hist',random_state=42, verbosity = 0, n_jobs=1),
        save_gridding_plot = True,
        ensemble_fold=20,
        min_ensemble_required=10,
        grid_len_lon_upper_threshold=P_upper_,
        grid_len_lon_lower_threshold=P_lower_,
        grid_len_lat_upper_threshold=P_upper_,
        grid_len_lat_lower_threshold=P_lower_,
        points_lower_threshold=50,
        temporal_start= 1, 
        temporal_end=366,
        temporal_step=P_step_,
        temporal_bin_interval=P_bin_,
        Spatio1='longitude',
        Spatio2 = 'latitude',
        Temporal1 = 'DOY',
        use_temporal_to_train=True,
        njobs=1                   
    )
    return model_hurdle_in_Ada



# In[52]:


def read_pred_set(year, x_names):
    p = f'/beegfs/store4/chenyangkang/06.ebird_data/43.Phenology_project_files/data/00.Subsampled_data/predset_data_{year}_NDVI_EMAG2.csv'
    dd = pd.read_csv(p, dtype = {dt:'float32' for dt in x_names + ['longitude','latitude']})
    dd[['max_NDVI','min_NDVI','median_NDVI','var_NDVI','max_NDVI_diff','min_NDVI_diff','median_NDVI_diff','var_NDVI_diff']] = dd[['max_NDVI','min_NDVI','median_NDVI','var_NDVI','max_NDVI_diff','min_NDVI_diff','median_NDVI_diff','var_NDVI_diff']].fillna(-9999)
    dd[['mean_GM','std_GM','gradient_x_GM','gradient_y_GM']] = dd[['mean_GM','std_GM','gradient_x_GM','gradient_y_GM']].fillna(99999)
    dd = dd.fillna(-1)
    return dd
    


# In[53]:

##### plot
def get_one_year_data(year, sp, Di, pred_saving_path, raw_pred_saving_path):

    indexing_lng_lat = None

    data = None
    for DOY in tqdm(range(1, 367)):
        
        try:
            
            tmp = pd.read_csv(os.path.join(raw_pred_saving_path, f'raw_pred_{sp}_{year}_{tro}_DOY{DOY}_{P_lower_}_{P_upper_}_{P_step_}_{P_bin_}.csv')).dropna()

            if indexing_lng_lat is None:
                sub = tmp.set_index('h3_05').h3.h3_to_geo_boundary().reset_index(drop=False)
                sub['h3_lng'] = sub.geometry.centroid.x
                sub['h3_lat'] = sub.geometry.centroid.y
                indexing_lng_lat = sub[['h3_05','h3_lng','h3_lat']]
                del sub

            new_dat = tmp[['h3_05','year','DOY','elevation_mean','pred_mean','pred_std']]
            del tmp
            
            if data is None:
                data = new_dat
                del new_dat
            else:
                data = data.append(new_dat, ignore_index=True)
                del new_dat
                
        except:
            continue

    data = data.merge(indexing_lng_lat, on='h3_05', how='left')
    
    if not Di == 'log_Raos_plus_1e-8':
        data['pred_mean'] = np.where(data['pred_mean']<0, 0, data['pred_mean'])
        
    data.columns = ['h3_05', 'year', 'DOY','elevation_mean', Di, 'pred_std', 'h3_lng', 'h3_lat']
    
    del indexing_lng_lat
    
    data.to_csv(os.path.join(pred_saving_path, f'{sp}_h3_05_year{year}_{tro}_{P_lower_}_{P_upper_}_{P_step_}_{P_bin_}.csv'), index=False)
    
    return data


              
def make_plot(data, sp, year, pred_saving_path, Di):
    from stemflow.utils.plot_gif import make_sample_gif
    make_sample_gif(data, os.path.join(pred_saving_path, f'{Di}_{year}_{tro}_{P_lower_}_{P_upper_}_{P_step_}_{P_bin_}.gif'),
                                col=Di, log_scale = False,
                                Spatio1='h3_lng', Spatio2='h3_lat', Temporal1='DOY',
                                figsize=(18,9), xlims=(-180, 180), ylims=(-90,90), grid=True,
                                xtick_interval=20, ytick_interval=20,
                                vmin = np.quantile(data[Di], 0.05), vmax = np.quantile(data[Di], 0.95),
                                lightgrey_under=False,
                                lng_size = 360, lat_size = 180, dpi=100, fps=30, cmap='viridis')




# path
model_saving_path = f'/beegfs/store4/chenyangkang/06.ebird_data/46.Seasonal_Diversity/02.SDM/01.model/{sp1}_{tro}'
metric_saving_path = f'/beegfs/store4/chenyangkang/06.ebird_data/46.Seasonal_Diversity/02.SDM/03.metrics/{sp1}_{tro}'
pred_saving_path = f'/beegfs/store4/chenyangkang/06.ebird_data/46.Seasonal_Diversity/02.SDM/02.pred/{sp1}_{tro}'
raw_pred_saving_path = f'/beegfs/store4/chenyangkang/06.ebird_data/46.Seasonal_Diversity/02.SDM/02.pred/{sp1}_{tro}/raw'

# mkdir if the path does not exist
for path_ in [model_saving_path, metric_saving_path, pred_saving_path, raw_pred_saving_path]:
    if not os.path.exists(path_):
        os.makedirs(path_)
        
        
# In[54]:


for year in year_list:
   
#    if f'{Di}_{year}_iter9.csv' in os.listdir(pred_saving_path):
#        if not f'{Di}_{year}.gif' in os.listdir(pred_saving_path):
#            try:
#                data = get_one_year_data(year, sp, Di)
#                make_plot(data, sp, year, pred_saving_path, Di)#
#
#            except Exception as e:
#                print(e)
#                pass
#        
#        continue
#
    ##### read sp data
    if not load_:
#    if not os.path.exists(os.path.join(model_saving_path, f'{sp1}_{year}_{tro}_adastem_model_{P_lower_}_{P_upper_}_{P_step_}_{P_bin_}.pkl')):

        try:
            sp_data = pd.read_csv(f'/beegfs/store4/chenyangkang/06.ebird_data/46.Seasonal_Diversity/01.diversity_index/{Di}_year{year}_{tro}.csv')
        except Exception as e:
            print(e)
            continue

        ### read checklist data
        checklist_data = pd.read_csv(f'/beegfs/store4/chenyangkang/06.ebird_data/43.Phenology_project_files/data/00.Subsampled_data/checklist_data_{year}_NDVI_EMAG2.csv', dtype = {**{dt:'float32' for dt in x_names + ['longitude','latitude']}, **{'sampling_event_identifier':'str'}})

        if Di in checklist_data.columns:
            del checklist_data[Di]

        print(checklist_data.columns)
        print(sp_data.columns)

        #### include species data
        checklist_data = pd.merge(checklist_data, sp_data,on='sampling_event_identifier',how='left')
        checklist_data = checklist_data.reset_index(drop=True)
        checklist_data[Di] = checklist_data[Di].fillna(0)
        checklist_data = checklist_data.dropna(subset=[Di])
        checklist_data[['max_NDVI','min_NDVI','median_NDVI','var_NDVI','max_NDVI_diff','min_NDVI_diff','median_NDVI_diff','var_NDVI_diff']] =checklist_data[['max_NDVI','min_NDVI','median_NDVI','var_NDVI','max_NDVI_diff','min_NDVI_diff','median_NDVI_diff','var_NDVI_diff']].fillna(-9999)
        checklist_data[['mean_GM','std_GM','gradient_x_GM','gradient_y_GM']] = checklist_data[['mean_GM','std_GM','gradient_x_GM','gradient_y_GM']].fillna(99999)
        del sp_data

        #### manipulations
        checklist_data['year']=year

        ##### Get the X and y
        print('Get the X and y')
        X = checklist_data[['longitude','latitude']+x_names]
        y = checklist_data[Di].values
        del checklist_data

        ##### now start training!
        print('now start training!')
        X_train, X_test, y_train, y_test = ST_train_test_split(X, y, Spatio1='longitude', Spatio2='latitude', Temporal1='DOY', 
                            Spatio_blocks_count=100, Temporal_blocks_count=100, test_size=0.2)

        # declare model
        model = declare_adastem_model()

        #### 1. training model
        model.fit(X_train.reset_index(drop=True), y_train, verbosity=1)


        # save model
        with open(os.path.join(model_saving_path, f'{sp1}_{year}_{tro}_adastem_model_{P_lower_}_{P_upper_}_{P_step_}_{P_bin_}.pkl'),'wb') as f:
            pickle.dump(model, f)

        try:
            print('Feature Importances')
            print(model.feature_importances_)
        except:
            pass

        #### 2. evaluation
        # evaluation
        pred = model.predict(X_test, verbosity=1)

        # missing prediction
        perc = np.sum(np.isnan(pred.flatten()))/len(pred.flatten())
        print(f'Percentage not predictable {round(perc*100, 2)}%')

        # eval metrics
        if Di=='log_Raos_plus_1e-8':
            pred_df = pd.DataFrame({
                'y_true':y_test.flatten(),
                'y_pred':pred.flatten(),
                'DOY':X_test['DOY'].values,
            }).dropna()
        else:
            pred_df = pd.DataFrame({
                'y_true':y_test.flatten(),
                'y_pred':np.where(pred.flatten()<0, 0, pred.flatten()),
                'DOY':X_test['DOY'].values,
            }).dropna()

        metrics_dict = AdaSTEM.eval_STEM_res('regression', pred_df.y_true, pred_df.y_pred)
        print(metrics_dict)

        # save metrics
        with open(os.path.join(metric_saving_path, f'{sp1}_{year}_{tro}_metrics_{P_lower_}_{P_upper_}_{P_step_}_{P_bin_}.json'), 'w') as j:
            json.dump(metrics_dict, j)

        try:
            ##bining DOY
            DOY_metrics = {}
            pred_df['DOY_bin'] = np.digitize(pred_df['DOY'].values,np.linspace(1,367,30))
            for DOY_bin in sorted(pred_df['DOY_bin'].unique()):
                tmp = pred_df[pred_df['DOY_bin'] == DOY_bin]
                tmp_metrics = AdaSTEM.eval_STEM_res('regression', tmp.y_true, tmp.y_pred)
                DOY_metrics[str(DOY_bin)] = tmp_metrics
                print(f'DOY_bin {DOY_bin}: {tmp_metrics}')

            # save metrics
            with open(os.path.join(metric_saving_path, f'{sp1}_{year}_{tro}_metrics_{P_lower_}_{P_upper_}_{P_step_}_{P_bin_}_by_DOY_bins.json'), 'w') as j:
                json.dump(DOY_metrics, j)

        except Exception as e:
            print(e)
            pass

    else:
        with open(os.path.join(model_saving_path, f'{sp1}_{year}_{tro}_adastem_model_{P_lower_}_{P_upper_}_{P_step_}_{P_bin_}.pkl'), 'rb') as f:
            model = pickle.load(f)
            model.ensemble_models_disk_saver = False

    #### 3. predict
    # read predset
    pred_set = read_pred_set(year, x_names)
    
    # predset
    _ = predict_days(model,pred_set,year,sp1,pred_saving_path,['longitude','latitude'] + x_names, batch=1)
        
    try:
        del pred_df, pred_set, metrics_dict, pred, model, X_train, X_test, y_train, y_test, X, y
    except:
        pass
    
    # plot
    try:
        data = get_one_year_data(year, sp, Di, pred_saving_path, raw_pred_saving_path)
        make_plot(data, sp, year, pred_saving_path, Di)

    except Exception as e:
        print(e)
        continue


