import glob
all_files = glob.glob('./log_file/*')
out_file = open('fail_list.txt','w')

fail_list = []
for file in all_files:
    with open(file,'r') as f:
        data = f.read()
        
        if 'Convergence Fail' in data:
            Flag = 1
        else:
            if 'Converged!' not in data:
                Flag=1
            else:
                Flag=0
        
        if Flag:
            file_name = file.split('/')[-1]
            if 'NDVI' in file_name:
                niche = '_'.join(file_name.split('_')[:-2])
                var_ = '_'.join(file_name.split('_')[-2:]).split('.log')[0]
            elif (
                ('tmin' in file_name) or
                ('tmax' in file_name) or
                ('tmean' in file_name)
            ):
                niche = '_'.join(file_name.split('_')[:-1])
                var_ = '_'.join(file_name.split('_')[-1:]).split('.log')[0]
                
            # fail_list.append((niche, var_))
            out_file.write(f"{niche}\t{var_}\n")
            
out_file.close()
            