#!/bin/csh
#SBATCH --time=12:00:00 # walltime, abbreviated by -t
#SBATCH -o log/conv2d_nchw_eval_%j.sout # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e log/conv2d_nchw_eval_%j.serr # name of the stderr, using job and first node values
#SBATCH --ntasks=2    # number of MPI tasks, abbreviated by -n
#SBATCH --account=soc-gpu-np     # account - abbreviated by -A
#SBATCH --partition=soc-gpu-np  # partition, abbreviated by -p
#SBATCH --gres=gpu:a100:1  # partition, abbreviated by -p

module load cuda
python3 ./conv2d_nchw.py resume $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10}