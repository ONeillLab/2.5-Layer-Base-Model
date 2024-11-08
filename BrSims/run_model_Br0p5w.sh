#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --account=def-oneill
#SBATCH --nodes=3
#SBATCH --ntasks=65
#SBATCH --cpus-per-task=2
#SBATCH --mem=187G
#SBATCH --job-name=run_sim
#SBATCH --constraint=[skylake|cascade]

module load python/3.11.5 intel/2023.2.1 mpi4py StdEnv/2023 scipy-stack

source ~/.virtualenvs/SimEnv/bin/activate

srun python run_model_MPI2_Br0p5w.py

deactivate
