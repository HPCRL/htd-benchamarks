#!/bin/csh
#SBATCH --time=12:00:00 # walltime, abbreviated by -t
#SBATCH -o log/matvec_tune_%j.sout # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e log/matvec_tune_%j.serr # name of the stderr, using job and first node values
#SBATCH --ntasks=2    # number of MPI tasks, abbreviated by -n
#SBATCH --account=soc-gpu-np     # account - abbreviated by -A
#SBATCH --partition=soc-gpu-np  # partition, abbreviated by -p
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64000
#SBATCH --exclude notch369,notch370

module load cuda python3
python3 ./matvec_gpu.py tune $1 $2 $3