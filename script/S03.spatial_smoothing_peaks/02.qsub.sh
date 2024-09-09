# Read niches into an array
mapfile -t niches < ../trophic_niches.txt
new_item="peak_IRG"
niches+=($new_item)

# Define node specifications for 4 different chunks
nodes_specifications=("nodes=cu04:ppn=4,mem=40G" "nodes=cu07:ppn=4,mem=40G" "nodes=cu06:ppn=4,mem=40G")

# Initialize counter
counter=0

# Loop through each niche and create job scripts with different node specifications
for niche in "${niches[@]}"
do
    # Determine the node specification based on the counter
    node_spec="${nodes_specifications[$((counter % 3))]}"
    
    # Create the job script
    echo """
    cd \$PBS_O_WORKDIR
    python -u 01.smoothing.py ${niche} spring > ./log_file/${niche}_smoothing.log 2>&1
    """ > ${niche}_smoothing.sh
    
    # Set permissions for the job script
    chmod 755 ${niche}_smoothing.sh
    
    # Submit the job script with the specified node configuration
    qsub -l ${node_spec} ${niche}_smoothing.sh
    
    # Remove the job script after submission
    rm ${niche}_smoothing.sh
    
    # Increment the counter
    counter=$((counter + 1))

    sleep 60
    
done


