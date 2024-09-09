python get_not_finished_two_explaining.py
wait
sleep 1


# Define node specifications for 4 different chunks
nodes_specifications=("nodes=cu03:ppn=4,mem=50G" "nodes=cu04:ppn=4,mem=50G" "nodes=cu05:ppn=4,mem=50G" "nodes=cu06:ppn=4,mem=50G" "nodes=cu07:ppn=4,mem=50G" "nodes=cu08:ppn=4,mem=50G")

# Path to your file
file_path="not_finished_list.txt"

# Initialize counter
counter=0

# Reading the file line by line
while IFS=$'\t' read -r niche target_var; do

    # Determine the node specification based on the counter
    node_spec="${nodes_specifications[$((counter % 6))]}"
    
    # Create the job script
    echo """
    cd \$PBS_O_WORKDIR
    python -u GreenWave_BirdWave_correlation_BYM2_multi_niches_and_season_two_explaining.py ${niche} ${target_var} > ./log_file/${niche}_${target_var}_two_explaining.log 2>&1
    """ > ${niche}_${target_var}.sh
    
    # Set permissions for the job script
    chmod 755 ${niche}_${target_var}.sh
    
    # Submit the job script with the specified node configuration
    qsub -l ${node_spec} ${niche}_${target_var}.sh
    
    # Remove the job script after submission
    rm ${niche}_${target_var}.sh
    
    # Increment the counter
    counter=$((counter + 1))

done < "$file_path"



