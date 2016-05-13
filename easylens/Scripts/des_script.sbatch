#!/bin/bash -l

#SBATCH --partition=dphys_compute
#SBATCH --time=24:00:00
#SBATCH --job-name="DES"
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --output=/users/sibirrer/Logs/des.%j.o
#SBATCH --error=/users/sibirrer/Logs/des.%j.e
#SBATCH --exclusive
#======START===============================
cd $HOME
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Running on $SLURM_JOB_CPUS_PER_NODE cpus per node."
echo "Current working directory is `pwd`"
#module load python/2.7.5-intel-14.0.1

module load sextractor
export PYTHONPATH=/apps/monch/openmpi/1.7.5/gcc/4.7.3/python_2.7.5/site-packages/:$PYTHONPATH
export LD_LIBRARY_PATH=/apps/monch/openmpi/1.7.5/gcc/4.7.3/lib:$LD_LIBRARY_PATH
export PATH=/apps/monch/openmpi/1.7.5/gcc/4.7.3/bin:$PATH

module load python/2.7.6-gcc-4.8.1


module list

which mpirun
which python


cd /users/sibirrer/Software/easylens/easylens/Scripts/
mpirun python des_script.py $'/mnt/lnec/sibirrer/des_input.txt'
#======END=================================
