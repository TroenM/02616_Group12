#!/bin/bash
#BSUB -J template
#BSUB -o ../outputs/out/template%J.out
#BSUB -e ../outputs/err/template%J.err
#BSUB -q hpcintro
#BSUB -W 00:05
## #BUSB -M 128MB
#BSUB -R "rusage[mem=128MB]"
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
# source ../.venv/bin/activate

mpirun python3 Mandelbrot_Template.py 10
