#!/bin/bash
#BSUB -J grid_image_scale_test[5-5]
#BSUB -o ../outputs/out/grid_image_scale_test%J_%I.out
#BSUB -e ../outputs/err/grid_image_scale_test%J_%I.err
#BSUB -q hpcintro
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -W 02:00
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

# The grid version needs a perfect-square number of ranks (16 -> 4x4 grid).
# The chunk-size argument is unused, but positional, so a placeholder is needed
echo $CURRENT_SIZE
mpirun -n 72 python3 Blocking/Mandelbrot_Grid.py 256 $FORMATTED_SIZE
