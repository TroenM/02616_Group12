#!/bin/bash

#BSUB -J lscpu
#BSUB -o ../outputs/lscpu.out
#BSUB -e ../outputs/lscpu.err

#BSUB -q hpcintro
#BSUB -W 00:01
#BSUB -R "rusage[mem=128MB]"
#BSUB -R "span[hosts=1]"
#BSUB -n 24

lscpu -C
lscpu
lstopo-no-graphics

