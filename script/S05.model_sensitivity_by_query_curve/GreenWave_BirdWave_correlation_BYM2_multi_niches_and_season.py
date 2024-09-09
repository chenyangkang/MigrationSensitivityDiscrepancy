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
# import tensorflow_probability.substrates.jax as tfp
# jax.scipy.special.erfcx = tfp.math.erfcx
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

from utils_GreenWave_BirdWave_corr_BYM2 import *
# from utils_longterm_phenological_shift_BYM2 import build_model_longterm_trend


trophic_niche = sys.argv[1] #'all'#sys.argv[1]
season = 'spring'
target_var = sys.argv[2] #'delta_NDVI'

comb = pd.read_csv(f'../../data/D05.greentrace_thermotrace/{trophic_niche}_curve_query.csv')
comb = comb[comb['season']==season]
comb = comb.set_index('h3_02').h3.h3_to_geo_boundary().reset_index(drop=False)
comb = comb[comb.area<200]

comb = comb.dropna(subset=['mu_ARR', 'std_ARR',
                           f'mean_expected_trace_by_{target_var}',f'std_expected_trace_by_{target_var}'])


# %%
def get_centered(df, var_='mu_ARR'):
    df[f'centered_{var_}'] = df[var_]-df[var_].mean()
    return df

comb = comb.groupby('h3_02').apply(get_centered, var_='mu_ARR').reset_index(drop=True)
comb = comb.groupby('h3_02').apply(get_centered, var_=f'mean_expected_trace_by_{target_var}').reset_index(drop=True)

#############
univ_cell = get_univ_cell(comb)

###
univ_cell = univ_cell[(univ_cell.lng>=comb.lng.min()) & (univ_cell.lng<=comb.lng.max()) & \
        (univ_cell.lat>=comb.lat.min()) & (univ_cell.lat<=comb.lat.max())]
comb = comb[comb['h3_02'].isin(set(univ_cell.index))]

###
adj_mat = calculat_adj_mat(univ_cell)
univ_cell = univ_cell.reset_index(drop=False)
cell_encoder = LabelEncoder().fit(univ_cell['h3_02'])
univ_cell['cell_index'] = cell_encoder.transform(univ_cell['h3_02'])
comb['cell_index'] = cell_encoder.transform(comb['h3_02'])
year_encoder = LabelEncoder().fit(comb['year'])
comb['year_index'] = year_encoder.transform(comb['year'])
univ_pairs_adj_df = get_pair_df_from_df(univ_cell, adj_mat)




# %%
model = build_model(comb, univ_cell, univ_pairs_adj_df, target_var)

# with model:
#     idata = pm.sampling.jax.sample_numpyro_nuts(1000, #random_seed = 42,
#                                     chains=4,tune=1000, 
#                                     progressbar=True)

idata = Run_Model(model, saving_path=f'../../data/D06.BYM2_results_query_curve/{trophic_niche}_{target_var}_sensitivity.pkl',
                  max_iter=8, max_tune=100000, SAMPLE_SIZE=2000, TUNES=2000,
                must_converge_var=['mu_alpha','mu_beta','logit_rho_beta_BYM2_component','beta_lat',
                                             'scaling_factor_ICAR_beta_BYM2_component','sigma_alpha',
                                             'beta_sigma_phi_BYM2','sigma_error','rho_ICAR_beta_BYM2_component'], 
                almost_converge_var=['env_date','theta_alpha',
                                        'phi_ICAR_beta_BYM2_component','theta_ICAR_beta_BYM2_component','theta_error',
                                        'alpha','beta','convolved_re_beta_BYM2_component',
                                        'arrival_true'])

# Save the whole package of variables for modeling
with open(f'../../data/D06.BYM2_results_query_curve/modeling_package_{trophic_niche}_{target_var}_sensitivity.pkl','wb') as f:
    pickle.dump([comb, univ_cell, univ_pairs_adj_df],f)



