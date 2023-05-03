#!/usr/bin/env bash
#SBATCH --partition=smallgpunodes
#SBATCH --nodelist=gpunode26
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=/u/edithal/work/git_repos/csc2516_2023_project/code/slurm_output_train_news_synthesizer_3.txt 
#SBATCH --error=/u/edithal/work/git_repos/csc2516_2023_project/code/slurm_output_train_news_synthesizer_3.txt
source /u/edithal/work/pyenvs/pytorch/bin/activate
srun -u python /u/edithal/work/git_repos/csc2516_2023_project/code/train_news_synthesizer.py
