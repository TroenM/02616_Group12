#!/bin/bash

#BSUB -J sleep
#BSUB -o ../outputs/sleep.out
#BSUB -e ../outputs/sleep.err

#BSUB -q hpcintro
#BSUB -W 00:01
#BSUB -R "rusage[mem=128MB]"
#BSUB -R "span[ptile=24]"
#BSUB -n 96

sleep 0.1