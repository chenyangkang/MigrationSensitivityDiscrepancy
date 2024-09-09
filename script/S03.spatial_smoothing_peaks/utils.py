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
filterwarnings('ignore')

tqdm.pandas()


def get_pair_df_from_df(tmp, mat):
    ### get pairwise adjacent information
    adj_pairs_df = []
    for index1, line1 in tqdm(tmp[['cell_index', 'h3_02']].drop_duplicates().reset_index(drop=True).iterrows(), total=len(tmp)):
        for index2, line2 in tmp[['cell_index', 'h3_02']].drop_duplicates().reset_index(drop=True).iloc[index1:,:].iterrows():
            if (line1['h3_02']!=line2['h3_02']) and (mat.loc[line1['h3_02'], line2['h3_02']]==1):
                adj_pairs_df.append({
                    'cell1':line1['cell_index'],
                    'cell2':line2['cell_index'],
                })
                
    adj_pairs_df = pd.DataFrame(adj_pairs_df)        
    return adj_pairs_df   
    
    
def get_univ_cell(comb, plot=False):
    ####### get universal adj matrix
    ### simulate the north hemisphere grid
    lon_grid, lat_grid = np.meshgrid(
        np.linspace(-180,180,1000),
        np.linspace(-90,90,1000),
    )

    univ_cell = pd.DataFrame({
        'lng':lon_grid.flatten(),
        'lat':lat_grid.flatten()
    }).h3.geo_to_h3(2)

    univ_cell = univ_cell.groupby('h3_02').first()
    univ_cell = univ_cell.h3.h3_to_geo_boundary()
    univ_cell = univ_cell[univ_cell.area<200]
    
    univ_cell['dist'] = np.where(univ_cell.index.isin(comb.h3_02.values), 1, 0)
    
    return univ_cell


def calculat_adj_mat(univ_cell):
    ### calculating adj_mat
    adj_mat = pd.DataFrame(
        np.zeros([len(univ_cell),len(univ_cell)]),
        columns=list(univ_cell.index),
        index=list(univ_cell.index)
    )
    for index, line in tqdm(univ_cell.iterrows(), total=len(univ_cell)):
        neighbors = univ_cell[~univ_cell.geometry.disjoint(line.geometry)]
        for index2 in list(neighbors.index):
            adj_mat.loc[index, index2] = adj_mat.loc[index2, index] = 1
    return adj_mat


def get_pair_df_from_df(tmp, mat):
    ### get pairwise adjacent information
    adj_pairs_df = []
    for index1, line1 in tqdm(tmp[['cell_index', 'h3_02']].drop_duplicates().reset_index(drop=True).iterrows(), total=len(tmp)):
        for index2, line2 in tmp[['cell_index', 'h3_02']].drop_duplicates().reset_index(drop=True).iloc[index1:,:].iterrows():
            if (line1['h3_02']!=line2['h3_02']) and (mat.loc[line1['h3_02'], line2['h3_02']]==1):
                adj_pairs_df.append({
                    'cell1':line1['cell_index'],
                    'cell2':line2['cell_index'],
                })
                
    adj_pairs_df = pd.DataFrame(adj_pairs_df)        
    return adj_pairs_df  


def decide_pass_or_not(idata):
    rhat_ = az.rhat(idata)
    ess_ = az.ess(idata)
    ######
    vars_ = list(dict(rhat_.variables).keys())
    vars_ = [i for i in vars_ if not 'dim' in i and not 'index' in i]

    ######
    max_rhat_ = np.max([float(rhat_[i].max()) for i in vars_])
    Fail1 = True if max_rhat_>1.03 else False

    ######  
    min_ess_ = np.min([float(ess_[i].min()) for i in vars_])
    Fail2 = True if min_ess_<400 else False
    
    return Fail1, max_rhat_, Fail2, min_ess_



def Run_Model(model1, saving_path='./idata.pkl', max_iter=5, max_tune=40000, SAMPLE_SIZE=3000, TUNES=1000):
    # ##### modeling
    # model1 = build_model(HM_df, univ_cell, univ_pairs_adj_df)

    ####### sampling
    SAMPLE_CHAINS=4
    SAMPLE_CORES=4
    
    with model1:
        idata = pm.sampling.jax.sample_numpyro_nuts(SAMPLE_SIZE, #random_seed = 42,
                                        chains=SAMPLE_CHAINS,tune=TUNES, 
                                        progressbar=True)
        
    Fail1, max_rhat_, Fail2, min_ess_ = decide_pass_or_not(idata)

    for try_ in range(max_iter):
        if Fail1 or Fail2:
            
            if TUNES>max_tune:
                raise AttributeError('Convergence Fail')
            
            if Fail1 & Fail2:
                sign_ = f'Both RHAT ({max_rhat_}) and ESS ({min_ess_})' 
            elif Fail1:
                sign_ = f'RHAT ({max_rhat_})'
            elif Fail2:
                sign_ = f'ESS ({min_ess_})'
                
            TUNES*=2
            
            if (Fail2 and (not Fail1)):
                SAMPLE_SIZE*=2
                
            print(f'Fail {sign_}. Try Again... Tuning: {TUNES}, Sampling: {SAMPLE_SIZE}')
            
            with model1:
                idata = pm.sampling.jax.sample_numpyro_nuts(SAMPLE_SIZE,random_seed = 42,
                                        chains=SAMPLE_CHAINS,tune=TUNES, 
                                        progressbar=True)
            
            Fail1, max_rhat_, Fail2, min_ess_ = decide_pass_or_not(idata)
        
        else:
            print('Converged!')
            with open(saving_path, 'wb') as f:
                pickle.dump(idata, f)
            
            break
        
    if Fail1 or Fail2:
        raise RuntimeError('Model fitting failed!')
        
    return idata



def ICAR_prior(sshape, name, node1, node2):

    def pairwise_diff(phi_, node1_, node2_):
        return -0.5 * pm.math.sum((phi_[node1_] - phi_[node2_])**2)

    phi_ICAR = pm.Normal(f'phi_ICAR_{name}', mu=0, sigma=1, shape=sshape)
    pm.Potential(f"spatial_diff_{name}", pairwise_diff(phi_ICAR, node1, node2))
    ICAR_sum_to_zero = pm.Normal.dist(mu=0, sigma=0.001 * sshape)
    pm.Potential(f"tt_sum1_{name}", pm.logp(
        ICAR_sum_to_zero, pt.sum(phi_ICAR)
    ))

    return phi_ICAR


def prior_convolved_RE(sshape, name, node1, node2):

    #### prior for phi
    phi_ICAR = ICAR_prior(sshape=sshape, name=name, node1 = node1, node2 = node2)

    #### prior for rho
    logit_rho = pm.Normal(f'logit_rho_{name}', mu=0, sigma=1)
    rho_ICAR = pm.Deterministic(f'rho_ICAR_{name}', pm.math.invlogit(logit_rho))

    #### prior for theta
    theta_ICAR = pm.Normal(f'theta_ICAR_{name}', mu=0, sigma=1, shape=sshape)

    #### prior for scaling_factor
    scaling_factor_ICAR = pm.Uniform(f'scaling_factor_ICAR_{name}', lower=0)

    #### spatial random + spatial structured
    convolved_re =  pm.Deterministic(f'convolved_re_{name}', pm.math.sqrt(1 - rho_ICAR) * theta_ICAR + pm.math.sqrt(rho_ICAR / scaling_factor_ICAR) * phi_ICAR)

    # convolved_re = phi_ICAR
    return convolved_re


## Smoothing
def build_model_smoothing(data, univ_cell, univ_pairs_adj_df):
    with pm.Model() as model:

        ## data
        unique_year_count = data['year'].unique().shape[0]
        sshape = unique_cell_count = int(univ_cell['cell_index'].values.max()+1)
        
        node1 = univ_pairs_adj_df['cell1'].values
        node2 = univ_pairs_adj_df['cell2'].values
        
        cell_index = data['cell_index'].values
        year_index = data['year_index'].values
        
        lat = univ_cell['lat'].values - univ_cell['lat'].values.mean()
        
        # arrival_temp = data['centered_arrival_temp'].values
        arrival_date = data['mean_DOY_peak'].values
        arrival_date_error = data['std_DOY_peak'].values
        
        ## Mean arrival date
        mu_ARR = pt.stack([pm.Normal(f'mu_ARR_{year}',mu=120, sigma=40) for year in range(unique_year_count)])
        # Latitude trend
        beta_lat = pm.HalfNormal('beta_lat', sigma=5)
        # 
        beta_BYM2_component = pt.stack([prior_convolved_RE(sshape, f'beta_BYM2_component_year{year}', node1, node2) for year in range(unique_year_count)])
        beta_sigma_phi_BYM2 = pm.Uniform(f'beta_sigma_phi_BYM2', lower=0, upper=10)
        ARR = pm.Deterministic('ARR', mu_ARR[year_index] + beta_lat * lat[cell_index] + beta_BYM2_component[year_index, cell_index] * beta_sigma_phi_BYM2)
        ARR_obs = pm.Normal('ARR_obs', mu=ARR, sigma=arrival_date_error, observed=arrival_date)
        
    return model
        
        
        