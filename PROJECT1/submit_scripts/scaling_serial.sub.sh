#!/bin/bash
#BSUB -J Serial
#BSUB -o ../outputs/out/Serial%J.out
#BSUB -e ../outputs/err/Serial%J.err
#BSUB -q hpcintro
#BSUB -W 00:30
## #BUSB -M 128MB
#BSUB -R "rusage[mem=1GB]"
#BSUB -n 1
#BSUB -R "span[hosts=1]"


###               hpcintro queue only has XeonE5_2650v4
###                       vvvvvvvvvvvvv
## #BSUB -R "select[model==XeonE5_2650v4]"

## OBS!!!: This script has to executed from the PROJECT1/src folder
## for the paths to work out
## From src folder execute line below:
## bsub < ../submit_scripts/scaling_run_serial.sub.sh

source ../modules.sh

echo "=== Serial template (1 rank) ==="
python3 Mandelbrot_Template_scaling.py
ls -l ../results/Serial
