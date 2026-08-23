#!/bin/bash

# for portal
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE
unset SLURM_TASKS_PER_NODE

source /rds/projects/c/chenlv-ai-and-chemistry/wuwj/anaconda3/etc/profile.d/conda.sh
conda activate un-nmr

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1



DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/uspto/preprocessed
exp_path=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/exp/local2global/uspto/step-1-nmr2fragment/joint-bixt-fragment-d512_2026-07-28_23-39-06/
python -m preprocess.audit_fragment_carbon_valence \
  --config ${exp_path}/logs/.hydra/config.yaml \
  --checkpoint ${exp_path}/checkpoints/epoch=030.ckpt \
  --data-dir ${DATA_PATH} \
  --split val \
  --output ${exp_path}/fragment_carbon_valence_val_epoch=030.json