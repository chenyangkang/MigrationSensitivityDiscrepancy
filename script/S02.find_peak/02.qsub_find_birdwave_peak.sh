

for niche in $(<../trophic_niches.txt)
    do
        for stemflow_params in 10_40_20_50 10_100_30_80
            do
                echo """
                cd \$PBS_O_WORKDIR
                python -u process_birdwave_data.py $niche $stemflow_params > ${niche}_${stemflow_params}.log 2>&1
                """ > ${niche}_${stemflow_params}_peak.sh
                qsub -l nodes=1:ppn=1,mem=50G ${niche}_${stemflow_params}_peak.sh
                rm ${niche}_${stemflow_params}_peak.sh
            done
    done
