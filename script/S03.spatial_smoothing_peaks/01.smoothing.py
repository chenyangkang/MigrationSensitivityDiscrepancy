# %%
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
import h3pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pickle
import arviz as az
import pymc as pm
import pymc.sampling.jax as pmjax
import jax

from fastprogress.fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import force_console_behavior
master_bar, progress_bar = force_console_behavior()
import pytensor.tensor as pt
import re
from warnings import filterwarnings
import matplotlib.colors as colors
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd 
import sys
import os

filterwarnings('ignore')
tqdm.pandas()

from utils import *


# %%
target_niche = sys.argv[1] #'all'
target_season = 'spring'
if not os.path.exists('./compile_dir'):
    os.makedirs('./compile_dir')
    
COMP_DIR=f'./compile_dir/{target_niche}_{target_season}'
COMP_FORM=f"compiledir_{target_niche}_{target_season}_mcmc"  # compilation directory name format
PYT_FLG=f"PYTENSOR_FLAGS='compiledir_format=${COMP_FORM},base_compiledir=${COMP_DIR}'"
PYTENSOR_FLAGS=f'compiledir_format=${COMP_FORM},base_compiledir=${COMP_DIR}'
os.environ['COMP_DIR'] = COMP_DIR
os.environ['COMP_FORM'] = COMP_FORM
os.environ['PYT_FLG'] = PYT_FLG
os.environ['PYTENSOR_FLAGS'] = PYTENSOR_FLAGS


# %%
## 01.Load data
data = pd.read_csv('../../data/D02.wave_peak/all_birdwave_greenwave_peak.csv')
data = data[(data['niche_or_level']==target_niche) & (data['season']==target_season)]
data = data[data['lat']>0] # North Hemisphere only

data = data.dropna(subset = ['mean_DOY_peak','std_DOY_peak'])
data = data[(data['std_DOY_peak'] > 0) & (data['std_DOY_peak'] <= 40)]



# %%
data = data.set_index('h3_02').h3.h3_to_geo_boundary().reset_index(drop=False)
data = data[data.area<200]


# %%
# Get univ cell
univ_cell = get_univ_cell(data)
univ_cell = univ_cell[(univ_cell.lng>=data.lng.min()) & (univ_cell.lng<=data.lng.max()) & \
        (univ_cell.lat>=data.lat.min()) & (univ_cell.lat<=data.lat.max())]
data = data[data['h3_02'].isin(set(univ_cell.index))]

# Get adj mat
adj_mat = calculat_adj_mat(univ_cell)
univ_cell = univ_cell.reset_index(drop=False)

# Encode cell index and year index
cell_encoder = LabelEncoder().fit(univ_cell['h3_02'])
univ_cell['cell_index'] = cell_encoder.transform(univ_cell['h3_02'])
data['cell_index'] = cell_encoder.transform(data['h3_02'])
year_encoder = LabelEncoder().fit(data['year'])
data['year_index'] = year_encoder.transform(data['year'])

# transform adj mat to pairs
univ_pairs_adj_df = get_pair_df_from_df(univ_cell, adj_mat)


# %% [markdown]
# ## 02.Spatial smoothing the migration arrival

# %%
model = build_model_smoothing(data, univ_cell, univ_pairs_adj_df)
idata = Run_Model(model, saving_path=f'../../data/D03.smoothed_peaks/{target_niche}_{target_season}.pkl', max_iter=8, max_tune=100000, SAMPLE_SIZE=1000, TUNES=1000)


#%%
with model:
    post_samples = pm.sample_posterior_predictive(idata)
    

# %%
data['mu_ARR'] = np.concatenate(post_samples.posterior_predictive['ARR_obs'], axis=0).mean(axis=0)
data['std_ARR'] = np.concatenate(post_samples.posterior_predictive['ARR_obs'], axis=0).std(axis=0)

# %%
data.to_csv(f'../../data/D03.smoothed_peaks/smoothed_{target_niche}_{target_season}.csv')


# %%


# %%



