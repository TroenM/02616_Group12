#!/bin/bash
#BSUB -J rows_image_scale_test[1-5]
#BSUB -o ../outputs/out/rows_image_scale_test%J_%I.out
#BSUB -e ../outputs/err/rows_image_scale_test%J_%I.err
#BSUB -q hpcintro
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -W 03:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 16
#BSUB -R "span[ptile=16]"

source ../modules.sh
#source ../.venv/bin/activate

# Define the image sizes using standard Bash array syntax (space separated)
IMAGE_SIZES=(5000 15000 25000 50000 93000)

# Calculate zero-based index
INDEX=$((LSB_JOBINDEX - 1))
CURRENT_SIZE=${IMAGE_SIZES[$INDEX]}

# Format the string as WxH
FORMATTED_SIZE="${CURRENT_SIZE}x${CURRENT_SIZE}"

# The chunk-size argument is unused by the row version (one row strip per rank),
# but it is positional, so a placeholder is needed before the image size
echo $CURRENT_SIZE
mpirun -n 16 python3 Blocking/Mandelbrot_Blocking.py 1 $FORMATTED_SIZE
