#!/usr/bin/env bash
source /u/edithal/pyenvs/pytorch/bin/activate
python -u /u/edithal/work/git_repos/csc2516_2023_project/code/train_mnist_ctgan.py > /u/edithal/work/git_repos/csc2516_2023_project/code/slurm_output_train_mnist_ctgan.txt 2>&1
