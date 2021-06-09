#!/bin/bash

#SBATCH --output=slurmjob.out
#SBATCH --constraint=icx8360Y
#SBATCH --job-name="thermal_mpi"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

cd $SLURM_SUBMIT_DIR

source /opt/intel/oneAPI/latest/setvars.sh 
source /opt/crtdc/mpich/3.2.1-gcc-8.3.1/vars.sh

mkdir -p data
mpirun -n 72 -bind-to core:1 ./bin/thermal_mpi 8 8 8 32 32 32 10 >> data/results.csv 
