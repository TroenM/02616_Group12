#!/bin/bash
# Worker scaling test for the dynamic scheduler (25000x25000 image).
# LSF job arrays can't change the core count per element, so this script

IMAGE_SIZE="25000x25000"
WORKERS=(25 49 64 100 121)

for N in "${WORKERS[@]}"; do
bsub <<JOB
#!/bin/bash
#BSUB -J dynamic_worker_scale_test_N${N}
#BSUB -o ../outputs/out/dynamic_worker_scale_test_N${N}_%J.out
#BSUB -e ../outputs/err/dynamic_worker_scale_test_N${N}_%J.err
#BSUB -q hpc
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -W 02:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -n ${N}
#BSUB -R "span[ptile=24]"

source ../modules.sh
#source ../.venv/bin/activate

# Use exactly 256 for the chunks-per-axis argument to maintain the 256x256 grid
echo ${N}
mpirun -n ${N} python3 Nonblocking/dynamic_scheduling_numwork.py 256 ${IMAGE_SIZE}
JOB
done
