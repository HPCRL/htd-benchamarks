#!/bin/csh
#SBATCH --time=12:00:00 # walltime, abbreviated by -t
#SBATCH -o log/conv2d_nchw_eval_%j.sout # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e log/conv2d_nchw_eval_%j.serr # name of the stderr, using job and first node values
#SBATCH --ntasks=2    # number of MPI tasks, abbreviated by -n
#SBATCH --account=soc-gpu-np     # account - abbreviated by -A
#SBATCH --partition=soc-gpu-np  # partition, abbreviated by -p
#SBATCH --gres=gpu:a100:1  # partition, abbreviated by -p
module load cuda
ncu  python3 ./conv2d_nchw.py timing $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} |  grep 'Duration' | awk '{print $3 " " $2}' | awk '{max = (max < $1) ? $1 : max} END {print "Execution time of this operator: " max " " $2}'z
#python3 ./conv2d_nchw.py timing $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} 
