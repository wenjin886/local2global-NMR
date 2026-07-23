#!/bin/bash

# for portal
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE
unset SLURM_TASKS_PER_NODE

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

# test data
# SRC_DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/local2global/data/uspto/exp_data
# export DATA_PATH=data/uspto/preprocessed 

# train data
SRC_DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/END_NMR/data/uspto/download/data/multimodal_spectroscopic_dataset
export DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/uspto/preprocessed

# Train Model
# python -m local2geo_module.train

# Evaluate Model

python -m local2geo_module.eval \
  --checkpoint /rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/exp/local2global/uspto/local2geo/bs512d-dim512_2026-07-23_11-08-41/checkpoints/epoch=000.ckpt \
  --config /rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/exp/local2global/uspto/local2geo/bs512d-dim512_2026-07-23_11-08-41/logs/local2geo/runs/2026-07-23_11-08-41/.hydra/config.yaml \
  --smiles "O=C(CC(F)(F)F)NC[C@H]1CN(c2ccc3c(c2)CCCc2cn[nH]c2-3)C(=O)O1"  \
  --output xyz_out/step256.xyz \
  --relax-steps 256

