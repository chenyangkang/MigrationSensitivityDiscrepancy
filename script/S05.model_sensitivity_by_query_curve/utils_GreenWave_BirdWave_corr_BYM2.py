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
# import jax
# import tensorflow_probability.substrates.jax as tfp
# jax.scipy.special.erfcx = tfp.math.erfcx
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


#
def decide_pass_or_not(idata, must_converge_var=None, almost_converge_var=None):

    rhat_ = az.rhat(idata)
    ess_ = az.ess(idata)
    
    all_var = list(rhat_.data_vars.variables.keys())
    
    if almost_converge_var is None:
        almost_converge_var = []
    else:
        for i in almost_converge_var:
            if not i in all_var:
                raise ValueError(f'{i} not in idata!')
        
    if must_converge_var is None:
        must_converge_var = list(set(all_var) - set(almost_converge_var))
    else:
        for i in must_converge_var:
            if not i in all_var:
                raise ValueError(f'{i} not in idata!')
            
    current_all_var = set(almost_converge_var) | set(must_converge_var)
    rest = set(all_var) - current_all_var
    rest = list(rest)
    if len(rest)>0:
        print(f'{rest} not in current convergent target, add them to "must converge list".')
        must_converge_var.extend(rest)
    
    ######
    if len(must_converge_var) == 0:
        max_rhat_1 = 1.0
        must_converge_Fail1 = False
    else:
        max_rhat_1 = np.max([float(rhat_[i].max()) for i in must_converge_var])
        must_converge_Fail1 = True if max_rhat_1>1.03 else False

    ######
    if len(almost_converge_var) == 0:
        max_rhat_2 = 1.0
        almost_converge_Fail1 = False
    else:
        max_rhat_2 = np.max([float(rhat_[i].quantile(0.99)) for i in almost_converge_var])
        almost_converge_Fail1 = True if max_rhat_2>1.03 else False

    ######  
    if len(must_converge_var) == 0:
        min_ess_1 = 1e8
        must_converge_Fail2 = False
    else:
        min_ess_1 = np.min([float(ess_[i].min()) for i in must_converge_var])
        must_converge_Fail2 = True if min_ess_1<400 else False

    ######
    if len(almost_converge_var) == 0:
        min_ess_2 = 1e8
        almost_converge_Fail2 = False
    else:
        min_ess_2 = np.min([float(ess_[i].quantile(0.01)) for i in almost_converge_var])
        almost_converge_Fail2 = True if min_ess_2<400 else False
    
    return must_converge_Fail1 or almost_converge_Fail1, max([max_rhat_1, max_rhat_2]), must_converge_Fail2 or almost_converge_Fail2, min([min_ess_1, min_ess_2])


def Run_Model(model1, saving_path='./idata.pkl', max_iter=5, max_tune=40000, SAMPLE_SIZE=3000, TUNES=1000, must_converge_var=None, almost_converge_var=None):
    # ##### modeling
    # model1 = build_model(HM_df, univ_cell, univ_pairs_adj_df)

    ####### sampling
    SAMPLE_CHAINS=4
    SAMPLE_CORES=4
    
    with model1:
        idata = pm.sampling.jax.sample_numpyro_nuts(SAMPLE_SIZE, #random_seed = 42,
                                        chains=SAMPLE_CHAINS,tune=TUNES, 
                                        progressbar=True)
        
    Fail1, max_rhat_, Fail2, min_ess_ = decide_pass_or_not(idata, must_converge_var=must_converge_var, almost_converge_var=almost_converge_var)

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
            
            Fail1, max_rhat_, Fail2, min_ess_ = decide_pass_or_not(idata, must_converge_var=must_converge_var, almost_converge_var=almost_converge_var)
        
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
    

def build_model(comb, univ_cell, univ_pairs_adj_df, target_var):
    with pm.Model() as model:

        ## data
        unique_year_count = comb['year'].unique().shape[0]
        sshape = unique_cell_count = int(univ_cell['cell_index'].values.max()+1)
        
        node1 = univ_pairs_adj_df['cell1'].values
        node2 = univ_pairs_adj_df['cell2'].values
        
        cell_index = comb['cell_index'].values
        year_index = comb['year_index'].values
        
        env_date_mean = comb[f'centered_mean_expected_trace_by_{target_var}'].values
        env_date_std = np.where(comb[f'std_expected_trace_by_{target_var}'].values==0, 
                                np.mean(comb[f'std_expected_trace_by_{target_var}'].values),
                                comb[f'std_expected_trace_by_{target_var}'].values)
        BirdWave_date_mean = comb['centered_mu_ARR'].values
        BirdWave_date_std = comb['std_ARR'].values
        lat = univ_cell['lat'].values - univ_cell['lat'].values.mean()
        
        #
        env_date = pm.Normal('env_date', mu=env_date_mean, sigma=env_date_std)

        # params
        mu_alpha = pm.Normal('mu_alpha',mu=0, sigma=1)
        theta_alpha = pm.Normal('theta_alpha', mu=0, sigma=1, shape=sshape)
        sigma_alpha = pm.HalfNormal('sigma_alpha',sigma=1)
        alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * theta_alpha)
        
        # beta
        mu_beta = pm.Normal('mu_beta',mu=0, sigma=1)
        beta_BYM2_component = prior_convolved_RE(sshape, 'beta_BYM2_component', node1, node2)
        beta_sigma_phi_BYM2 = pm.HalfNormal(f'beta_sigma_phi_BYM2', sigma=1)
        beta_lat = pm.Normal('beta_lat', mu=0, sigma=0.1)
        beta = pm.Deterministic('beta', mu_beta + beta_lat * lat + beta_BYM2_component * beta_sigma_phi_BYM2)
        
        theta_error = pm.Normal('theta_error', mu=0, sigma=1, shape=len(comb))
        sigma_error = pm.HalfNormal('sigma_error',sigma=5)
        arrival_true = pm.Deterministic('arrival_true', alpha[cell_index] + beta[cell_index] * env_date + theta_error * sigma_error)
        BirdWave_true = pm.Normal('BirdWave_true', mu=arrival_true, sigma=BirdWave_date_std, observed=BirdWave_date_mean) #
                                
    return model



def build_model_two_explaining(comb, univ_cell, univ_pairs_adj_df):
    with pm.Model() as model:

        ## data
        unique_year_count = comb['year'].unique().shape[0]
        sshape = unique_cell_count = int(univ_cell['cell_index'].values.max()+1)
        
        node1 = univ_pairs_adj_df['cell1'].values
        node2 = univ_pairs_adj_df['cell2'].values
        
        cell_index = comb['cell_index'].values
        year_index = comb['year_index'].values
        
        env_date_mean1 = comb[f'centered_mean_expected_trace_by_delta_NDVI'].values
        env_date_std1 = np.where(comb[f'std_expected_trace_by_delta_NDVI'].values==0, 
                                np.mean(comb[f'std_expected_trace_by_delta_NDVI'].values),
                                comb[f'std_expected_trace_by_delta_NDVI'].values)
        
        env_date_mean2 = comb[f'centered_mean_expected_trace_by_tmin'].values
        env_date_std2 = np.where(comb[f'std_expected_trace_by_tmin'].values==0, 
                                np.mean(comb[f'std_expected_trace_by_tmin'].values),
                                comb[f'std_expected_trace_by_tmin'].values)
        
        
        BirdWave_date_mean = comb['centered_mu_ARR'].values
        BirdWave_date_std = comb['std_ARR'].values
        lat = univ_cell['lat'].values - univ_cell['lat'].values.mean()
        
        #
        env_date1 = pm.Normal('env_date1', mu=env_date_mean1, sigma=env_date_std1)
        env_date2 = pm.Normal('env_date2', mu=env_date_mean2, sigma=env_date_std2)

        # params
        mu_alpha = pm.Normal('mu_alpha',mu=0, sigma=1)
        theta_alpha = pm.Normal('theta_alpha', mu=0, sigma=1, shape=sshape)
        sigma_alpha = pm.HalfNormal('sigma_alpha',sigma=1)
        alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * theta_alpha)
        
        # beta
        mu_beta1 = pm.Normal('mu_beta1',mu=0, sigma=1)
        beta1_BYM2_component = prior_convolved_RE(sshape, 'beta1_BYM2_component', node1, node2)
        beta1_sigma_phi_BYM2 = pm.HalfNormal(f'beta1_sigma_phi_BYM2', sigma=1)
        beta1_lat = pm.Normal('beta1_lat', mu=0, sigma=0.1)
        beta1 = pm.Deterministic('beta1', mu_beta1 + beta1_lat * lat + beta1_BYM2_component * beta1_sigma_phi_BYM2)
        
        mu_beta2 = pm.Normal('mu_beta2',mu=0, sigma=1)
        beta2_BYM2_component = prior_convolved_RE(sshape, 'beta2_BYM2_component', node1, node2)
        beta2_sigma_phi_BYM2 = pm.HalfNormal(f'beta2_sigma_phi_BYM2', sigma=1)
        beta2_lat = pm.Normal('beta2_lat', mu=0, sigma=0.1)
        beta2 = pm.Deterministic('beta2', mu_beta2 + beta2_lat * lat + beta2_BYM2_component * beta2_sigma_phi_BYM2)
        
        theta_error = pm.Normal('theta_error', mu=0, sigma=1, shape=len(comb))
        sigma_error = pm.HalfNormal('sigma_error',sigma=5)
        arrival_true = pm.Deterministic('arrival_true', alpha[cell_index] + beta1[cell_index] * env_date1 + beta2[cell_index] * env_date2 + theta_error * sigma_error)
        BirdWave_true = pm.Normal('BirdWave_true', mu=arrival_true, sigma=BirdWave_date_std, observed=BirdWave_date_mean) #
                                
    return model


def build_model_year_effect(comb, univ_cell, univ_pairs_adj_df, target_var):
    with pm.Model() as model:

        ## data
        unique_year_count = comb['year'].unique().shape[0]
        sshape = unique_cell_count = int(univ_cell['cell_index'].values.max()+1)
        
        node1 = univ_pairs_adj_df['cell1'].values
        node2 = univ_pairs_adj_df['cell2'].values
        
        cell_index = comb['cell_index'].values
        year_index = comb['year_index'].values
        # year_norm = #comb['year_norm'].values
        year_norm = np.sort(comb['year_index'].unique()) - comb['year_index'].unique().mean()
        
        env_date_mean = comb[f'centered_mean_expected_trace_by_{target_var}'].values
        env_date_std = np.where(comb[f'std_expected_trace_by_{target_var}'].values==0, 
                                np.mean(comb[f'std_expected_trace_by_{target_var}'].values),
                                comb[f'std_expected_trace_by_{target_var}'].values)
        BirdWave_date_mean = comb['centered_mu_ARR'].values
        BirdWave_date_std = comb['std_ARR'].values
        lat = univ_cell['lat'].values - univ_cell['lat'].values.mean()
        
        #
        env_date = pm.Normal('env_date', mu=env_date_mean, sigma=env_date_std)

        # params
        mu_alpha = pm.Normal('mu_alpha',mu=0, sigma=1)
        theta_alpha = pm.Normal('theta_alpha', mu=0, sigma=1, shape=sshape)
        sigma_alpha = pm.HalfNormal('sigma_alpha',sigma=1)
        alpha = pm.Deterministic('alpha', mu_alpha + sigma_alpha * theta_alpha)
        
        # beta
        mu_beta = pm.Normal('mu_beta',mu=0, sigma=1)
        beta_BYM2_component = prior_convolved_RE(sshape, 'beta_BYM2_component', node1, node2)
        beta_sigma_phi_BYM2 = pm.HalfNormal(f'beta_sigma_phi_BYM2', sigma=1)
        beta_lat = pm.Normal('beta_lat', mu=0, sigma=0.1)
        
        beta_year = pm.Normal('beta_year', mu=0, sigma=0.1)
        
        # beta = pm.Deterministic('beta', mu_beta + beta_year * year_norm[...,np.newaxis] + beta_lat * lat[np.newaxis,...] + beta_BYM2_component[np.newaxis,...] * beta_sigma_phi_BYM2)
        beta_spatial = pm.Deterministic('beta_spatial', mu_beta + beta_lat * lat + beta_BYM2_component * beta_sigma_phi_BYM2)
        mu_beta_temporal = pm.Deterministic('mu_beta_temporal', beta_year * year_norm)
        theta_beta_temporal = pm.Normal('theta_beta_temporal', mu=0, sigma=0.1, shape=len(year_norm))
        sigma_beta_temporal = pm.HalfNormal('sigma_beta_temporal',sigma=1)
        beta_temporal = pm.Deterministic('beta_temporal', mu_beta_temporal + theta_beta_temporal * sigma_beta_temporal)

        theta_error = pm.Normal('theta_error', mu=0, sigma=1, shape=len(comb))
        sigma_error = pm.HalfNormal('sigma_error',sigma=5)
        arrival_true = pm.Deterministic('arrival_true', alpha[cell_index] + (beta_spatial[cell_index] + beta_temporal[year_index]) * env_date + theta_error * sigma_error)
        BirdWave_true = pm.Normal('BirdWave_true', mu=arrival_true, sigma=BirdWave_date_std, observed=BirdWave_date_mean) #
                                
    return model
