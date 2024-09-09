python get_not_finished.py
wait
sleep 1


# Define node specifications for 4 different chunks
nodes_specifications=("nodes=cu03:ppn=4,mem=60G" "nodes=cu04:ppn=4,mem=60G" "nodes=cu05:ppn=4,mem=60G" "nodes=cu06:ppn=4,mem=60G" "nodes=cu07:ppn=4,mem=60G" "nodes=cu09:ppn=4,mem=60G")

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
    python -u GreenWave_BirdWave_correlation_BYM2_multi_niches_and_season.py ${niche} > ./log_file/${niche}_two_explaining.log 2>&1
    """ > ${niche}_two_explaining.sh
    
    # Set permissions for the job script
    chmod 755 ${niche}_two_explaining.sh
    
    # Submit the job script with the specified node configuration
    qsub -l ${node_spec} ${niche}_two_explaining.sh
    
    # Remove the job script after submission
    rm ${niche}_two_explaining.sh
    
    # Increment the counter
    counter=$((counter + 1))

done < "$file_path"



