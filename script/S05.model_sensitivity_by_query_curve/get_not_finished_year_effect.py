import glob
import os

var_list = ["mean_NDVI", "delta_NDVI", "tmean", "tmax", "tmin"]
# var_list = ['year_effect']


out_file = open('not_finished_list.txt','w')

with open('../trophic_niches.txt','r') as f:
    niche_list = f.read()
    niche_list = niche_list.strip().split('\n')
    print(niche_list)
    
for trophic_niche in niche_list:
    for target_var in var_list:
        if ((not os.path.exists(f'../../data/D06.BYM2_results_query_curve/{trophic_niche}_{target_var}_sensitivity_year_effect.pkl'))
            or 
            (not os.path.exists(f'../../data/D06.BYM2_results_query_curve/modeling_package_{trophic_niche}_{target_var}_sensitivity_year_effect.pkl'))):
            out_file.write(f"{trophic_niche}\t{target_var}\n")
            
out_file.close()

