#!/bin/bash

# for portal
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE
unset SLURM_TASKS_PER_NODE

export TMPDIR=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/tmp
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

python -c "import matplotlib; print(matplotlib.__version__)"

# test data
# SRC_DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/local2global/data/uspto/exp_data
# SAVE_DATA_PATH=data/uspto
# export DATA_PATH=data/uspto/preprocessed 

# train data
# SRC_DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/END_NMR/data/uspto/download/data/multimodal_spectroscopic_dataset
SAVE_DATA_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/uspto
export DATA_PATH=${SAVE_DATA_PATH}/preprocessed

# === Preprocesss Dataset ===
# python -m preprocess.uspto_nmr_preprocess \
#   --parquet_dir $SRC_DATA_PATH \
#   --save_dir $DATA_PATH \
#   --seed 0 \
#   --keep_duplicate_records

# python -m preprocess.count_graph_connectivity \
#   --data_dir $DATA_PATH \
#   --output $DATA_PATH/graph_connectivity_stats.json
# echo "Data Preprocessing Done. Starting Training..."


# === Train NMR-->2D ===
# python -m src.train -cn train_uspto_smiles
# python -m src.train -cn train_uspto_fragment
# python -m src.train -cn train_uspto_graph

# === Pretrain Initializer ===
# python -m local2geo_module.train

# ==================================
# === Preprocess 3D --> NMR dataset ===
COORD_PATH=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/END_NMR/data/uspto/download/preprocessed
# python -m preprocess.uspto_3d_nmr build \
#   --nmr-dir ${DATA_PATH} \
#   --coords-dir ${COORD_PATH} \
#   --output-dir ${SAVE_DATA_PATH}/3d2shift \
#   --index-cache /rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/data/uspto/3d2shift/index_cache.pt

# === PreTrain ===
# python -m shift3d_module.train

# === END2END ===
# 1. prior-only correction
export NMR2GRAPH_CKPT=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/exp/local2global/uspto/step-2-nmr2graph-edge/joint-bixt-graph-d512-nonebondw0.8-smiw1.0-neighborw1.0-Cvalw0.25_2026-08-23_09-14-09/checkpoints/last.ckpt
export TOPOLOGY_PRIOR_CKPT=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/exp/local2global/uspto/step-3-geoinit/dim512-bs32-fixnumatomsbug_2026-07-27_19-23-13/checkpoints/last.ckpt
python -m end2end_module.train_prior

# export SHIFT3D_CKPT=/rds/projects/c/chenlv-ai-and-chemistry/wuwj/Unsupervised_NMR/exp/local2global/uspto/step-4-3d2nmr/dim512-bs64_2026-08-24_15-36-51/checkpoints/last.ckpt
# python -m end2end_module.train

# For Portal
# ps -fu $USER | grep -E "python|wandb"

