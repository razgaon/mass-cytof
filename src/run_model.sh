#!/bin/bash
#SBATCH --job-name=run_vae_24h
#SBATCH --output=/nobackup/users/ifrah/mass-cytof/src/results/fulldata/runvae_nonpd1_output.log
#SBATCH --error=/nobackup/users/ifrah/mass-cytof/src/results/fulldata/runvae_nonpd1_error.log
#SBATCH --mem=1T
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --nodes=1

#SBATCH --time 12:00:00

####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG


USER=$(whoami)
HOME2=/nobackup/users/$USER
PYTHON_VIRTUAL_ENVIRONMENT=cytof
cd $HOME2/projects/condas/$PYTHON_VIRTUAL_ENVIRONMENT
export PATH="pwd/miniconda3/bin:$PATH"
. miniconda3/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited
echo "running!"

cd /nobackup/users/ifrah/mass-cytof/src
python3 -u /nobackup/users/ifrah/mass-cytof/src/run_model.py
echo "finished!"