#!/bin/bash
#BSUB -J Nonblocking
#BSUB -o ../outputs/out/Nonblocking%J.out
#BSUB -e ../outputs/err/Nonblocking%J.err
#BSUB -q hpcintro
#BSUB -W 00:05
## #BUSB -M 128MB
#BSUB -R "rusage[mem=1GB]"
#BSUB -n 24
#BSUB -R "span[hosts=1]"


###               hpcintro queue only has XeonE5_2650v4
###                       vvvvvvvvvvvvv
## #BSUB -R "select[model==XeonE5_2650v4]" 

## OBS!!!: This script has to executed from the PROJECT1/src folder
## for the paths to work out
## From src folder execute line below: 
## bsub < ../submit_scripts/template.sh

source ../modules.sh

for n in 2 4 6 8 10 12 14 16; do
    echo "=== Nonblocking with $n ranks==="
    mpirun -np $n python3 Nonblocking/Mandelbrot_Nonblocking.py
done
ls -l ../results/Nonblocking
