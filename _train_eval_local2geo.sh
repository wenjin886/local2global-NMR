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

# # train data
# SRC_DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/END_NMR/data/uspto/download/data/multimodal_spectroscopic_dataset
# export DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/uspto/preprocessed

# # Train Model
# python -m local2geo_module.train

# # Evaluate Model

# python -m local2geo_module.eval \
#   --smiles \
#   "CCCCCCCCCCCCC" \
#   --input-mode clean-soft \
#   --num-steps 256 \
#   --unbonded-distance-scale 0.85 \
#   --unbonded-weight 3.0 \
#   --anti-torsion-weight 10.0 \
#   --output-dir xyz_out \
#   --device cpu \
#   --write-sdf

# python -m local2geo_module.eval \
#   --smiles "C=C(CSCCCSc1ccc(C(=O)C(C)(C)N2CCOCC2)cc1)C(=O)OC" \
#   --input-mode clean-soft \
#   --num-steps 256 \
#   --unbonded-distance-scale 0.85 \
#   --unbonded-weight 3.0 \
#   --anti-torsion-weight 1.0 \
#   --ring-bond-weight 2.0 \
#   --conjugated-weight 2.0 \
#   --chain-extension-weight 10.0 \
#   --output-dir xyz_out \
#   --device cpu \
#   # --write-sdf

# python -m local2geo_module.eval \
#   --smiles "CCCCCCC" \
#   --seed-mode mds \
#   --num-steps 256 \
#   --output-dir xyz_out/mds \
#   --device cpu \
#   --write-sdf

# python -m local2geo_module.eval_prior \
  # --smiles "C=C(CSCCCSc1ccc(C(=O)C(C)(C)C(=O)OC)cc1)CN1CCOCC1" "CCCCCCC" "CCCCCCCCC" \
  # --input-mode corrupted-soft \
  # --noise-std 0.1 \
  # --output-dir xyz_out/corrupted-4

exp_path="/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/exp/local2global/uspto/step-3-geoinit/dim512-bs256_2026-07-27_17-23-06"
python -m local2geo_module.eval_hybrid \
  --checkpoint ${exp_path}/checkpoints/last.ckpt \
  --smiles "C=C(CSCCCSc1ccc(C(=O)C(C)(C)C(=O)OC)cc1)CN1CCOCC1" "CCCC" "CCCCCCCCC" \
  --seed-mode soft_stress \
  --soft-stress-steps 96 \
  --num-steps 0 \
  --output-dir ${exp_path}/xyz_out