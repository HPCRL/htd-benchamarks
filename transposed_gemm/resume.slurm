#!/bin/csh
#SBATCH --time=12:00:00 # walltime, abbreviated by -t
#SBATCH -o log/transposed_gemm_resume_%j.sout # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e log/transposed_gemm_resume_%j.serr # name of the stderr, using job and first node values
#SBATCH --ntasks=2    # number of MPI tasks, abbreviated by -n
#SBATCH --account=soc-gpu-np     # account - abbreviated by -A
#SBATCH --partition=soc-gpu-np  # partition, abbreviated by -p
#SBATCH --gres=gpu:a100:1

module load cuda
python3 ./transposed_gemm_gpu.py resume $1 $2 $3
