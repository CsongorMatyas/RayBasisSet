#!/bin/bash
#PBS -S /bin/bash
#PBS -l mem=2000MB
#PBS -l nodes=1:ppn=2
#PBS -l walltime=23:50:00

#PBS -A xdg-523-aa
#PBS -m bea
#PBS -M aia638@mun.ca

#  Gaussian job script

cd $PBS_O_WORKDIR
echo "Current working directory is `pwd`"
echo "Running on `hostname`"
echo "Starting run at: `date`"

module load gaussian

# Run g09
g09 < z.gjf > z.out
