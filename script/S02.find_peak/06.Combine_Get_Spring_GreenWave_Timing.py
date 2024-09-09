# %%
import glob
import pandas as pd
import h3pandas

# %%
all_data = []
for year in range(2010, 2021):
    data = pd.read_csv(f'../../data/D02.wave_peak/Green_Wave_Peaks_{year}_spring.csv')
    data['year'] = int(year)
    all_data.append(data)
    
all_data = pd.concat(all_data, axis=0).reset_index(drop=True)


# %%
all_data = all_data.set_index('h3_02').h3.h3_to_geo_boundary()
all_data['lng'] = all_data.geometry.centroid.x
all_data['lat'] = all_data.geometry.centroid.y
del all_data['geometry']
all_data = all_data.reset_index(drop=False)
all_data['season'] = 'spring'

# %%
all_data.to_csv('../../data/D02.wave_peak/all_greenwave_peak.csv')


# %%
data


# %%


# %%


# %%


# %%


# %%



