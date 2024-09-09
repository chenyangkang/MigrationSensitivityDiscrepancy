echo """
cd $\PBS_O_WORKDIR
python -u Get_Spring_GreenWave_Timing.py > Get_Spring_GreenWave_Timing.log 2>&1
""" > Get_Spring_GreenWave_Timing.sh
chmod 755 Get_Spring_GreenWave_Timing.sh
qsub -l nodes=cu06:ppn=11,mem=200G Get_Spring_GreenWave_Timing.sh
rm Get_Spring_GreenWave_Timing.sh
