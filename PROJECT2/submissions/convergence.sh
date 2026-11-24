#!/bin/bash
#BSUB -J conv[1-5]
#BSUB -o outputs/conv_%J_%I.out
#BSUB -e outputs/conv_%J_%I.err
#BSUB -q hpc
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -W 01:00
#BSUB -R "rusage[mem=1GB]"
#BSUB -n 8
#BSUB -R "span[hosts=1]"

source modules.sh
cd src/sequential

SIZES=(8 16 24 32 48)
N=${SIZES[$((LSB_JOBINDEX - 1))]}

mpirun -n 8 python3 main.py $N 100000 1e-4 conv 0
