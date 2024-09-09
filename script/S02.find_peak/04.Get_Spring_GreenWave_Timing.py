import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')
from pygam import LinearGAM, s
from scipy.signal import find_peaks
from tqdm.auto import tqdm

tqdm.pandas()

import h3pandas

def get_peak(df):
    sub_ = df.dropna(subset=['NDVI_diff','DOY'])
    sub_ = sub_[sub_['DOY']<=183]
    try:
        res = find_peaks(sub_['NDVI_diff'].values, prominence=0.3)
        peak_index = res[0][np.argmax(res[1]['prominences'])]
        peak_DOY = sub_['DOY'].values[peak_index]
    except:
        peak_DOY = np.nan 
    return peak_DOY

# def get_argmax(df):
#     sub_ = df.dropna(subset=['NDVI_diff','DOY'])
#     try:
#         argmax_DOY = sub_['DOY'].values[np.argmax(sub_['NDVI_diff'].values)]
#     except Exception as e:
#         print(e)
#         argmax_DOY = np.nan
#     return argmax_DOY

def make_one(year):
    
    import h3pandas
    from tqdm.auto import tqdm

    tqdm.pandas()

    with open(f'/beegfs/store4/chenyangkang/06.ebird_data/10.predictor_data/37.MODIS_NDVI/NDVI_h3_05_year{year}_smoothed_with_diff.pkl','rb') as f:
        NDVI = pickle.load(f)
        
    NDVI = pd.DataFrame(NDVI)
    peaks = NDVI.groupby('h3_05').progress_apply(get_peak)
    seasonality = NDVI.groupby('h3_05')['NDVI'].std()
    
    peaks = peaks.reset_index(drop=False)
    # peaks['argmax_DOY'] = argmax.values
    peaks['seasonality'] = seasonality.values
    peaks = peaks.set_index('h3_05').h3.h3_to_geo_boundary()
    peaks['lng'] = peaks.geometry.centroid.x
    peaks['lat'] = peaks.geometry.centroid.y

    peaks = peaks.reset_index(drop=False)
    peaks.columns = ['h3_05','peak','seasonality','geometry','lng','lat']
    del peaks['geometry']
    peaks = pd.DataFrame(peaks)
    
    ## Aggregate to h3_02
    peaks = peaks.h3.geo_to_h3(2).reset_index(drop=False)
    def get_mean_std(df):
        
        return pd.Series({
            'mean_DOY_peak':df['peak'].mean(),
            'std_DOY_peak':df['peak'].std(),
            'mean_seasonality':df['seasonality'].mean()
        })

    peaks = peaks.groupby('h3_02')[['peak','seasonality']].apply(get_mean_std).reset_index(drop=False)
    peaks.to_csv(f'../../data/D02.wave_peak/Green_Wave_Peaks_{year}_spring.csv', index=False)
    
        
# %%
from joblib import Parallel, delayed
results = Parallel(n_jobs=11)(delayed(make_one)(year) for year in tqdm(list(range(2010, 2021)), desc="Processing year"))




