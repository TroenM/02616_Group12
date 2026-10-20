#!/bin/bash
#BSUB -J assymetric[1-3]
#BSUB -o ../outputs/out/assymetric%J_%I.out
#BSUB -e ../outputs/err/assymetric%J_%I.err
#BSUB -q hpcintro
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -W 00:50
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 72
#BSUB -R "span[ptile=24]"

source ../modules.sh
source ../.venv/bin/activate

# Define the image sizes using standard Bash array syntax (space separated)
IMAGE_SIZES=(5000 15000 25000 50000 93000)

# Calculate zero-based index
INDEX=$((LSB_JOBINDEX - 1))
CURRENT_SIZE=${IMAGE_SIZES[$INDEX]}

# Format the string as WxH
FORMATTED_SIZE="${CURRENT_SIZE}x${CURRENT_SIZE}"

echo $CURRENT_SIZE

mpirun python3 Nonblocking/dynamic_rows.py 2500 $FORMATTED_SIZE -2.2:0 0:1.3

mpirun python3 Nonblocking/dynamic_scheduling.py 50 $FORMATTED_SIZE -2.2:0 0:1.3
