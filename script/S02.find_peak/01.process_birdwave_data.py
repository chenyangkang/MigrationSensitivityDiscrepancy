import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
import h3pandas
import sys
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

tqdm.pandas()

trophic_niche = str(sys.argv[1]) #'all'#
stemflow_params = str(sys.argv[2])


def get_peaks(tmp):
    
    try:
        tmp = tmp.sort_values(by='DOY')
        val = tmp['delog_log_richness_plus_1_diff'].values
        peaks = find_peaks((val - np.nanmean(val))/np.nanstd(val), 
                           prominence=0.5, width=15)[0]
        if len(peaks)==0:
            return np.nan
        else:
            if val[peaks[0]]<0:
                return np.nan
            else:
                return tmp['DOY'].values[peaks[0]]
    except:
        return np.nan
    
    
    
    
def max_x(x):
    try:
        a = x['DOY'].values[np.argmax(x['delog_log_richness_plus_1_diff'].values)]
        return a
                            
    except:
        return np.nan
                            
                                    
for year in tqdm(range(2010,2021)):
    # Read birdwave
    data_path = f'/beegfs/store4/chenyangkang/06.ebird_data/46.Seasonal_Diversity/02.SDM/02.pred/log_richness_plus_1_{trophic_niche}/log_richness_plus_1_h3_05_year{year}_{trophic_niche}_{stemflow_params}_smoothed_with_diff.csv'
    BirdWave = pd.read_csv(data_path)

    # process by season
    for season in ['spring','fall']:

        if season == 'spring':
            tmp_season_Birdwave = BirdWave[BirdWave['DOY']<=183]
        elif season=='fall':
            tmp_season_Birdwave = BirdWave[BirdWave['DOY']>=183]
            # if it is fall, we get the date that the birdwave decrease the fastest
            tmp_season_Birdwave['delog_log_richness_plus_1_diff'] = -tmp_season_Birdwave['delog_log_richness_plus_1_diff']

        try:
            season_Birdwave_max = tmp_season_Birdwave[tmp_season_Birdwave['delog_log_richness_plus_1_diff']>0].groupby('h3_05').progress_apply(max_x)

            season_Birdwave_peak = tmp_season_Birdwave.groupby('h3_05').progress_apply(
                get_peaks
            )
        except Exception as e:
            print(e)
            continue
        
        season_Birdwave = pd.concat([season_Birdwave_max, season_Birdwave_peak], axis=1)
        
        season_Birdwave = pd.DataFrame(season_Birdwave)
        season_Birdwave.columns = ['DOY_max','DOY_peak']

        season_Birdwave = season_Birdwave[~((season_Birdwave['DOY_peak']==1) | (season_Birdwave['DOY_peak']==356) | (season_Birdwave['DOY_peak']==366)| (season_Birdwave['DOY_peak']==183))]

        
        season_Birdwave = season_Birdwave.h3.h3_to_geo()
        season_Birdwave['lng'] = season_Birdwave.geometry.centroid.x
        season_Birdwave['lat'] = season_Birdwave.geometry.centroid.y
        del season_Birdwave['geometry']

        ### Aggregate to h3_02
        season_Birdwave = pd.DataFrame(season_Birdwave).reset_index(drop=True)
        season_Birdwave = season_Birdwave.h3.geo_to_h3(2)
        
        def get_mean_std(df):
            
            return pd.Series({
                'mean_DOY_max':df['DOY_max'].mean(),
                'std_DOY_max':df['DOY_max'].std(),
                'mean_DOY_peak':df['DOY_peak'].mean(),
                'std_DOY_peak':df['DOY_peak'].std(),
            })

        season_Birdwave = season_Birdwave.groupby('h3_02').apply(get_mean_std)
        geom = season_Birdwave.h3.h3_to_geo_boundary().geometry
        season_Birdwave['lng'] = geom.centroid.x
        season_Birdwave['lat'] = geom.centroid.y
        season_Birdwave = season_Birdwave.reset_index(drop=False)
        season_Birdwave.to_csv(f'../../data/D02.wave_peak/{season}_max_birdwave_day_{trophic_niche}_{year}_{stemflow_params}.csv', index=False)

        


