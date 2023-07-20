#!/bin/csh
#SBATCH --time=12:00:00 # walltime, abbreviated by -t
#SBATCH -o log/transposed_gemm_timing_%j.sout # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e log/transposed_gemm_timing_%j.serr # name of the stderr, using job and first node values
#SBATCH --ntasks=2    # number of MPI tasks, abbreviated by -n
#SBATCH --account=soc-gpu-np     # account - abbreviated by -A
#SBATCH --partition=soc-gpu-np  # partition, abbreviated by -p
#SBATCH --gres=gpu:a100:1
echo "$1x$2"
echo "$2x$3"
echo "$1x$3"
module load cuda
ncu python3 ./transposed_gemm_gpu.py timing $1 $2 $3 | grep 'Duration' | awk '{print $3 " " $2}' | awk '{max = (max < $1) ? $1 : max} END {print "Execution time of this operator: " max " " $2}'z &> "evaluate/transposed_gemm_M$1_K$2_N$3_sm_80.txt"
