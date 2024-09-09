
A_=10
B_=40
C_=20
D_=50

for year in {2010..2020}
do
{
#    for Di in log_abundance_plus_1 log_richness_plus_1 shannon_H Raos log_biomass_plus_1
    for Di in log_richness_plus_1
    do
    {
    for tro in all Trophic_Level_Herbivore Trophic_Level_Omnivore Trophic_Level_Scavenger Trophic_Level_Carnivore Trophic_Niche_Aquatic_predator Trophic_Niche_Frugivore Trophic_Niche_Granivore Trophic_Niche_Herbivore_aquatic Trophic_Niche_Herbivore_terrestrial Trophic_Niche_Invertivore Trophic_Niche_Nectarivore Trophic_Niche_Omnivore Trophic_Niche_Scavenger Trophic_Niche_Vertivore
    do
    {
    echo '''
    cd /beegfs/store4/chenyangkang/06.ebird_data/46.Seasonal_Diversity
    tro='"${tro}"'
    Di='${Di}'
    year='"${year}"'
    A_='"${A_}"'
    B_='"${B_}"'
    C_='"${C_}"'
    D_='"${D_}"'
    python -u SDM_stemflow_larger.py ${tro} ${Di} ${year} ${A_} ${B_} ${C_} ${D_} > stemflow_log/${tro}_${Di}_${year}_${A_}_${B_}_${C_}_${D_}.log 2>&1

    wait

    python -u calc_delta.py ${Di} ${tro} ${year} ${A_} ${B_} ${C_} ${D_} > smooth_and_diff_log/${Di}_${tro}_${year}_${A_}_${B_}_${C_}_${D_}.log 2>&1

    wait

    ''' > ${tro}_${Di}_${year}_${A_}_${B_}_${C_}_${D_}.sh

    chmod 755 ${tro}_${Di}_${year}_${A_}_${B_}_${C_}_${D_}.sh

    while true
    do
        {
            myjobcount=`qstat |grep "chenyangkang"|wc -l`
            if [[ $myjobcount -lt 1700 ]];then
                qsub -q cu -l nodes=1:ppn=1,mem=60G ${tro}_${Di}_${year}_${A_}_${B_}_${C_}_${D_}.sh
                break
            fi
            sleep 5
        }
    done
    rm ${tro}_${Di}_${year}_${A_}_${B_}_${C_}_${D_}.sh
    sleep 0.5

    }
    done

    }
    done
}
done
