#!/bin/sh
# Script for running gaussian

#PBS -l walltime=23:00:00
#PBS -l mem=2000mb
#PBS -r n
#PBS -l nodes=1:ppn=2

cd $PBS_O_WORKDIR #makes it easy to return to the directory from which you submitted the job:
module load gaussian
g09 < /home/iawad/RayBasisSet/glo/opt_IRC/Tb3lyp/IRCR_init_boats/./6/glo_1w_beta_boat6_B3LYP_631Gd_iopt.com > /home/iawad/RayBasisSet/glo/opt_IRC/Tb3lyp/IRCR_init_boats/./6/glo_1w_beta_boat6_B3LYP_631Gd_iopt.out
