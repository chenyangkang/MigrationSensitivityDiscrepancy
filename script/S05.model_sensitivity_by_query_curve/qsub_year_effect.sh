# Read niches into an array
mapfile -t niches < ../trophic_niches.txt
# new_item="peak_IRG"
# niches+=($new_item)

# Define node specifications for 4 different chunks
nodes_specifications=("nodes=cu03:ppn=4,mem=50G" "nodes=cu04:ppn=4,mem=50G" "nodes=cu05:ppn=4,mem=50G" "nodes=cu06:ppn=4,mem=50G" "nodes=cu07:ppn=4,mem=50G" "nodes=cu08:ppn=4,mem=50G" "nodes=cu09:ppn=4,mem=50G")
target_var_list=("delta_NDVI" "tmin" "mean_NDVI" "tmean" "tmax")
# Initialize counter
counter=0

# Loop through each niche and create job scripts with different node specifications
for target_var in "${target_var_list[@]}"
    do
        for niche in "${niches[@]}"
            do
                # Determine the node specification based on the counter
                node_spec="${nodes_specifications[$((counter % 7))]}"
                
                # Create the job script
                echo """
                cd \$PBS_O_WORKDIR
                python -u GreenWave_BirdWave_correlation_BYM2_multi_niches_and_season_year_effect.py ${niche} ${target_var} > ./log_file/${niche}_${target_var}_year_effect.log 2>&1
                """ > ${niche}_${target_var}_year_effect.sh
                
                # Set permissions for the job script
                chmod 755 ${niche}_${target_var}_year_effect.sh
                
                # Submit the job script with the specified node configuration
                qsub -l ${node_spec} ${niche}_${target_var}_year_effect.sh
                
                # Remove the job script after submission
                rm ${niche}_${target_var}_year_effect.sh
                
                # Increment the counter
                counter=$((counter + 1))
            done
    done





