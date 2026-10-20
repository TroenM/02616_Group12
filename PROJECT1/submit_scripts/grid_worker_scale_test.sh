#!/bin/bash
# Worker scaling test for the static grid version (25000x25000 image).
#BSUB -J grid_worker_scale_tests[1-3]
#BSUB -o ../outputs/out/grid_worker_scale_tests_%J_%I.out
#BSUB -e ../outputs/err/grid_worker_scale_tests_%J_%I.err
#BSUB -q hpcintro
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -W 02:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 72
#BSUB -R "span[ptile=24]"

IMAGE_SIZE="25000x25000"
WORKERS=(24 48 72)


source ../modules.sh
#source ../.venv/bin/activate

# Use exactly 256 for the chunks-per-axis argument to maintain the 256x256 grid
# Calculate zero-based index
INDEX=$((LSB_JOBINDEX - 1))
CURRENT_N_WORKER=${WORKERS[$INDEX]}

# The chunk-size argument is unused, but positional, so a placeholder is needed
echo $CURRENT_N_WORKER
mpirun -n ${CURRENT_N_WORKER} python3 Blocking/Mandelbrot_Grid_numwork.py 256 ${IMAGE_SIZE}
