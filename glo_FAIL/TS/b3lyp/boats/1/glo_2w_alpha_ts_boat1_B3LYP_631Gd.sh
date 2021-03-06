#!/bin/sh
# Script for running gaussian

#PBS -l walltime=47:00:00
#PBS -l mem=2000mb
#PBS -r n
#PBS -l nodes=1:ppn=2

cd $PBS_O_WORKDIR #makes it easy to return to the directory from which you submitted the job:

module load gaussian
g09 < glo_2w_alpha_ts_boat1_B3LYP_631Gd.com > glo_2w_alpha_ts_boat1_B3LYP_631Gd.out
