#!/bin/bash

#BSUB -J plot
#BSUB -o ./outputs/plotting/plot%J.out
#BSUB -e ./outputs/plotting/plot%J.err

#BSUB -q hpcintro
#BSUB -W 00:05
#BSUB -R "rusage[mem=256MB]"
#BSUB -R "span[ptile=1]"
#BSUB -n 1

source modules.sh

python3 src/Testsuite/plots.py --results results/memory_bandwidth --figs figs/memory_bandwidth