#!/bin/bash

source scripts/activate_conda_env.sh

python3 run_pretrained_openfold.py \
../function_pred/data/fastas/0_to_128 \
../function_pred/data/pdb_mmcif/mmcif_files/ \
--use_precomputed_alignments ../function_pred/data/alignments/ \
--output_dir ../function_pred/data/alphafold/tmp \
--model_device "cuda:0" \
--save_all_recycles \
# --jax_param_path ../../../projects/resources/alphafold_params/alphafold_params/params_model_1.npz \
