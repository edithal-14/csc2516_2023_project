#!/usr/bin/env bash
#SBATCH --partition=biggpunodes
#SBATCH --nodelist=gpunode23
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=/u/edithal/work/git_repos/csc2516_2023_project/code/slurm_output_train_census_synthesizer_0.txt 
#SBATCH --error=/u/edithal/work/git_repos/csc2516_2023_project/code/slurm_output_train_census_synthesizer_0.txt
source /u/edithal/work/pyenvs/pytorch/bin/activate
srun -u python /u/edithal/work/git_repos/csc2516_2023_project/code/train_census_synthesizer.py
