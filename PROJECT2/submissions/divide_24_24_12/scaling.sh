N=200
MAXIT=5
PS=(1 2 4 8 12 24 48 64)

mkdir -p outputs

mkdir -p results/divide_work_24_24_12

for P in "${PS[@]}"; do
    # wall time: ~30 min per iteration on 1 core, divided by P, plus margin
    MIN=$(( 30 * MAXIT / P + 15 ))
    WALL=$(printf "%02d:%02d" $(( MIN / 60 )) $(( MIN % 60 )))
    # memory per core: ~6 GB total at N=500, split over P ranks, plus 1 GB
    MEM=$(( 6000 / P + 1000 ))
    # up to 24 ranks fit on one node, above that fill whole nodes
    if [ "$P" -le 24 ]; then SPAN="span[hosts=1]"; else SPAN="span[ptile=24]"; fi

bsub <<JOB
#!/bin/bash
#BSUB -J strong_N${N}_P${P}[1-3]
#BSUB -o outputs/strong_N${N}_P${P}_%J_%I.out
#BSUB -e outputs/strong_N${N}_P${P}_%J_%I.err
#BSUB -q hpc
#BSUB -R "select[model==XeonE5_2650v4]"
#BSUB -W ${WALL}
#BSUB -R "rusage[mem=${MEM}MB]"
#BSUB -n ${P}
#BSUB -R "${SPAN}"

source modules.sh
source .venv/bin/activate
cd src/disjoint_boundaries


REP=\$((LSB_JOBINDEX - 1))
echo "$LSB_MCPU_HOSTS"    
mpirun -n ${P} python3 main.py -N ${N} --maxit ${MAXIT} --tol 0 --tag scaling --rep \$REP --outdir ../../results/divide_work_24_24_12/
JOB
done
