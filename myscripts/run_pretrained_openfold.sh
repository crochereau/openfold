#!/bin/bash

source scripts/activate_conda_env.sh

python3 run_pretrained_openfold.py \
../function_pred/data/fastas/ \
../function_pred/data/pdb_mmcif/mmcif_files/ \
--use_precomputed_alignments ../function_pred/data/alignments/ \
--output_dir ../function_pred/data/alphafold/test/ \
--model_device "cuda:0" \
--jax_param_path ../../../projects/resources/alphafold_params/alphafold_params/params_model_1.npz \
--save_single