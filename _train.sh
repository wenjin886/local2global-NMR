#!/bin/bash

# for portal
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE
unset SLURM_TASKS_PER_NODE

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# DATA_PATH=data/uspto/preprocessed 

SRC_DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/END_NMR/data/uspto/download/data/multimodal_spectroscopic_dataset
export DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/uspto/preprocessed
# Preprocesss Dataset
# python -m preprocess.uspto_nmr_preprocess \
#   --parquet_dir $SRC_DATA_PATH \
#   --save_dir $DATA_PATH \
#   --seed 0 \
#   --keep_duplicate_records
  
# Train Model
python -m src.train -cn train_uspto_fragment



