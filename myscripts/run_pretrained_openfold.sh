#!/bin/bash

source scripts/activate_conda_env.sh

python3 run_pretrained_openfold.py ../function_pred/data/fastas/0_to_128 ../function_pred/data/pdb_mmcif/mmcif_files/ \
--use_precomputed_alignments ../function_pred/data/alignments/ --output_dir ../function_pred/data/alphafold --save_all_recycles \
--model_device "cuda:0";\
python3 run_pretrained_openfold.py ../function_pred/data/fastas/129_to_256 ../function_pred/data/pdb_mmcif/mmcif_files/ \
--use_precomputed_alignments ../function_pred/data/alignments/ --output_dir ../function_pred/data/alphafold --save_all_recycles \
--model_device "cuda:1";\
python3 run_pretrained_openfold.py ../function_pred/data/fastas/257_to_512_1 ../function_pred/data/pdb_mmcif/mmcif_files/ \
--use_precomputed_alignments ../function_pred/data/alignments/ --output_dir ../function_pred/data/alphafold --save_all_recycles \
--model_device "cuda:2";\
python3 run_pretrained_openfold.py ../function_pred/data/fastas/257_to_512_2 ../function_pred/data/pdb_mmcif/mmcif_files/ \
--use_precomputed_alignments ../function_pred/data/alignments/ --output_dir ../function_pred/data/alphafold --save_all_recycles \
--model_device "cuda:3";\
python3 run_pretrained_openfold.py ../function_pred/data/fastas/257_to_512_3 ../function_pred/data/pdb_mmcif/mmcif_files/ \
--use_precomputed_alignments ../function_pred/data/alignments/ --output_dir ../function_pred/data/alphafold --save_all_recycles \
--model_device "cuda:4";\
python3 run_pretrained_openfold.py ../function_pred/data/fastas/257_to_512_4 ../function_pred/data/pdb_mmcif/mmcif_files/ \
--use_precomputed_alignments ../function_pred/data/alignments/ --output_dir ../function_pred/data/alphafold --save_all_recycles \
--model_device "cuda:5";\
python3 run_pretrained_openfold.py ../function_pred/data/fastas/512_to_1000 ../function_pred/data/pdb_mmcif/mmcif_files/ \
--use_precomputed_alignments ../function_pred/data/alignments/ --output_dir ../function_pred/data/alphafold --save_all_recycles \
--model_device "cuda:6"
