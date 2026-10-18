#!/bin/bash
# Worker scaling test for the static row version (25000x25000 image).

IMAGE_SIZE="25000x25000"
WORKERS=(25 49 64 100 121)

for N in "${WORKERS[@]}"; do
bsub <<JOB
#!/bin/bash
#BSUB -J rows_worker_scale_test_N${N}
#BSUB -o ../outputs/out/rows_worker_scale_test_N${N}_%J.out
#BSUB -e ../outputs/err/rows_worker_scale_test_N${N}_%J.err
#BSUB -q hpcintro
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -W 02:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -n ${N}
#BSUB -R "span[ptile=24]"

source ../modules.sh
#source ../.venv/bin/activate

echo ${N}
mpirun -n ${N} python3 Blocking/Mandelbrot_Blocking_numwork.py 1 ${IMAGE_SIZE}
JOB
done
