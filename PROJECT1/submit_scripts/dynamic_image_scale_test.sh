#!/bin/bash
#BSUB -J dynamic_image_scale_test[1-5]
#BSUB -o ../outputs/out/dynamic_image_scale_test%J_%I.out
#BSUB -e ../outputs/err/dynamic_image_scale_test%J_%I.err
#BSUB -q hpc
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -W 00:20
#BSUB -R "rusage[mem=2GB]"
#BSUB -n 256
#BSUB -R "span[ptile=24]"

source ../modules.sh
source ../.venv/bin/activate

# Define the image sizes using standard Bash array syntax (space separated)
IMAGE_SIZES=(5000 15000 25000 50000 93000)

# LSB_JOBINDEX runs 1-5, but Bash arrays are 0-indexed. Calculate the offset.
INDEX=$((LSB_JOBINDEX - 1))
CURRENT_SIZE=${IMAGE_SIZES[$INDEX]}

# Use exactly 256 for the chunks-per-axis argument to maintain the 256x256 grid
mpirun -n 256 python3 Nonblocking/test_dynamic_scheduling.py 256 $CURRENT_SIZE
