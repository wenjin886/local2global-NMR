#!/bin/bash

# for portal
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE
unset SLURM_TASKS_PER_NODE

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1


DATA_PATH=data/uspto/preprocessed 
python -m src.train -cn train_uspto_fragment



