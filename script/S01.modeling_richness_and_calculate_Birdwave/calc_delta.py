import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import h3pandas
from scipy.interpolate import splrep, BSpline
from tqdm.auto import tqdm
tqdm.pandas()

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import h3pandas
from multiprocessing import Pool
import rioxarray as rxr
import pyproj
from pyproj import Transformer
# transformer = Transformer.from_crs("ESRI:54008", "EPSG:4326",always_xy=True)
from pygam import s,f,l,LinearGAM
import sys
import warnings
import pickle

warnings.filterwarnings('ignore')

sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
wgs84 = pyproj.Proj("+init=EPSG:4326")
transformer = Transformer.from_proj(sinu, wgs84,always_xy=True)


Di = str(sys.argv[1])
tro= str(sys.argv[2])
year = int(sys.argv[3])
year_list = [year]

A_=int(sys.argv[4])
B_=int(sys.argv[5])
C_=int(sys.argv[6])
D_=int(sys.argv[7])

    
def generate_smoothed(df, ns=30, Di=Di):

    xnew = np.array(list(range(1, 367)))
    
    try:
        df = df.sort_values(by=[Di]).dropna()
        x = df['DOY'].values
        y = df[Di].values

        header_x = np.array(x[-30:] - 366)
        header_y = np.array(y[-30:])

        tail_x = np.array(x[:30] + 366)
        tail_y = np.array(y[:30])

        x = np.concatenate([header_x, x, tail_x])
        y = np.concatenate([header_y, y, tail_y])

#         tck = splrep(x, y)
#         smoothed = BSpline(*tck)(xnew)
        model = LinearGAM(s(0,ns), max_iter=500).fit(x, y)
        smoothed = model.predict(X=xnew.reshape(-1,1))
    
        
    except Exception as e:
        smoothed = [np.nan for i in range(366)]
    
    return pd.Series(smoothed, index=xnew)


def get_diff(df, Di=Di):
    dd = df.sort_values(by='DOY').set_index('DOY')[Di].values
    dd = np.gradient(dd, 1)
    return pd.Series(dd, index=range(1,367))



data = pd.read_csv(f'02.SDM/02.pred/{Di}_{tro}/{Di}_h3_05_year{year}_{tro}_{A_}_{B_}_{C_}_{D_}.csv')


##
from tqdm.auto import tqdm
tqdm.pandas()
# unique_cells_df = data.groupby('h3_05')[['h3_lng','h3_lat']].first().reset_index(drop=False)


## Interpolation
data = data.dropna(subset=[Di])
smoothed_res = data.sort_values(by=['h3_05']).groupby('h3_05').progress_apply(generate_smoothed)

smoothed_res = smoothed_res.reset_index(drop=False).melt(id_vars='h3_05',value_name=Di,var_name='DOY')

smoothed_res = data[['h3_05','h3_lng','h3_lat','year','elevation_mean','pred_std']].drop_duplicates(subset=['h3_05']).merge(
    smoothed_res, on=['h3_05'], how='right'
)

smoothed_res['h3_lng'] = smoothed_res['h3_lng'].round(6)
smoothed_res['h3_lat'] = smoothed_res['h3_lat'].round(6)
smoothed_res['elevation_mean'] = smoothed_res['elevation_mean'].round(2)

smoothed_res = smoothed_res.drop_duplicates().reset_index(drop=True)

if not (Di=='shannon_H' or Di=='Raos'):
    smoothed_res[Di] = np.exp(smoothed_res[Di])

The_diff = smoothed_res.groupby('h3_05').progress_apply(get_diff)
The_diff = The_diff.reset_index(drop=False).melt(id_vars=['h3_05'])
smoothed_res[f'delog_{Di}_diff'] = The_diff['value'].values.round(4)

if not (Di=='shannon_H' or Di=='Raos'):
    smoothed_res[Di] = np.log(smoothed_res[Di])

smoothed_res.to_csv(f'02.SDM/02.pred/{Di}_{tro}/{Di}_h3_05_year{year}_{tro}_{A_}_{B_}_{C_}_{D_}_smoothed_with_diff.csv', index=False)


## Make plot
import os
from stemflow.utils.plot_gif import make_sample_gif
make_sample_gif(smoothed_res, os.path.join(f'02.SDM/02.pred/{Di}_{tro}/', f'{Di}_{year}_{tro}_{A_}_{B_}_{C_}_{D_}_diff.gif'),
                            col=f'delog_{Di}_diff', log_scale = False,
                            Spatio1='h3_lng', Spatio2='h3_lat', Temporal1='DOY',
                            figsize=(18,9), xlims=(-180, 180), ylims=(-90,90), grid=True,
                            xtick_interval=20, ytick_interval=20,
                            vmin = np.quantile(smoothed_res[f'delog_{Di}_diff'][~np.isnan(smoothed_res[f'delog_{Di}_diff'])], 0.05),
                            vmax = np.quantile(smoothed_res[f'delog_{Di}_diff'][~np.isnan(smoothed_res[f'delog_{Di}_diff'])], 0.95),
                            lightgrey_under = False,
                            lng_size = 360, lat_size = 180, dpi=100, fps=30, cmap='viridis')


