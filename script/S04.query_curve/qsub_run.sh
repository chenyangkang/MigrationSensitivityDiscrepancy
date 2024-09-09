
mapfile -t niches < ../trophic_niches.txt

# Define node specifications for 4 different chunks
nodes_specifications=("nodes=cu04:ppn=4,mem=2G" "nodes=cu05:ppn=4,mem=2G")
# nodes_specifications=("nodes=cu07:ppn=4,mem=40G")

# Initialize counter
counter=0

# Loop through each niche and create job scripts with different node specifications
for niche in "${niches[@]}"
do
    # Determine the node specification based on the counter
    node_spec="${nodes_specifications[$((counter % 2))]}"
    
    # Create the job script
    echo """
    cd \$PBS_O_WORKDIR
    python -u Get_Timing_of_GW_and_Temp_by_query_the_curve.py ${niche} > ./log_file/${niche}.log 2>&1
    """ > ${niche}.sh
    
    # Set permissions for the job script
    chmod 755 ${niche}.sh
    
    # Submit the job script with the specified node configuration
    qsub -l ${node_spec} ${niche}.sh
    
    # Remove the job script after submission
    rm ${niche}.sh
    
    # Increment the counter
    counter=$((counter + 1))
done





