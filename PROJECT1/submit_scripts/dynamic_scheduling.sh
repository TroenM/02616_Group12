#!/bin/bash
#BSUB -J dynamic_scheduling
#BSUB -o ../outputs/out/test_dynamic_scheduling%J.out
#BSUB -e ../outputs/err/test_dynamic_scheduling%J.err
#BSUB -q hpcintro
#BSUB -W 00:05
## #BUSB -M 128MB
#BSUB -R "rusage[mem=1GB]"
#BSUB -n 4
#BSUB -R "span[ptile=4]"

###               hpcintro queue only has XeonE5_2650v4
###                       vvvvvvvvvvvvv
## #BSUB -R "select[model==XeonE5_2650v4]"

## OBS!!!: This script has to executed from the PROJECT1/src folder
## for the paths to work out
## From src folder execute line below:
## bsub < ../submit_scripts/template.sh

source ../modules.sh
source ../.venv/bin/activate

mpirun -n 4 python3 Nonblocking/test_dynamic_scheduling.py
