#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-oneill
#SBATCH --nodes=2
#SBATCH --ntasks=65
#SBATCH --cpus-per-task=2
#SBATCH --job-name run_sim

module load NiaEnv/2022a python/3.11.5 intel/2021u4 intelmpi/2021u4

source ~/.virtualenvs/SimEnv/bin/activate

srun python run_model_MPI.py

deactivate
