# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import numpy as np
import os
from pathlib import Path

import random
import sys
import time
import torch
import tqdm

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import (
    import_jax_weights_,
)

from scripts.utils import add_data_args


def main(args):

    for file in tqdm(glob.glob(args.fasta_path + '*.fasta')):
        # Gather input sequences
        with open(file, "r") as fp:
            lines = [l.strip() for l in fp.readlines()]
        tags, seqs = lines[::2], lines[1::2]
        tags = [l[1:] for l in tags]
    
        for tag, seq in zip(tags, seqs):
            print(tag)
            outfile = tag + '.npy'
            outdir_single = os.path.join(args.output_dir, 'single')
            outdir_pair = os.path.join(args.output_dir, 'pair')
            outpath_single = os.path.join(outdir_single, outfile)
            outpath_pair = os.path.join(outdir_pair, outfile)
        
            if not os.path.isfile(outpath_single) or not os.path.isfile(outpath_pair):
                try:
                    config = model_config(args.model_name)
                    model = AlphaFold(config)
                    model = model.eval()
                    import_jax_weights_(model, args.param_path, version=args.model_name)
                
                    model = model.to(args.model_device)
                    print('model on device')
                    template_featurizer = templates.TemplateHitFeaturizer(
                        mmcif_dir=args.template_mmcif_dir,
                        max_template_date=args.max_template_date,
                        max_hits=config.data.predict.max_templates,
                        kalign_binary_path=args.kalign_binary_path,
                        release_dates_path=args.release_dates_path,
                        obsolete_pdbs_path=args.obsolete_pdbs_path
                    )
                
                    use_small_bfd = (args.bfd_database_path is None)
                
                    data_processor = data_pipeline.DataPipeline(
                        template_featurizer=template_featurizer,
                    )
                
                    random_seed = args.data_random_seed
                    if random_seed is None:
                        random_seed = random.randrange(sys.maxsize)
                    feature_processor = feature_pipeline.FeaturePipeline(config.data)
                    
                    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                    Path(args.outdir_single).mkdir(parents=True, exist_ok=True)
                    Path(args.outdir_pair).mkdir(parents=True, exist_ok=True)
                        
                    if(args.use_precomputed_alignments is None):
                        alignment_dir = os.path.join(args.output_dir, "alignments")
                    else:
                        alignment_dir = args.use_precomputed_alignments
                
                    fasta_path = os.path.join(args.output_dir, f"{tag}_tmp.fasta")
                    with open(fasta_path, "w") as fp:
                        fp.write(f">{tag}\n{seq}")
            
                    logging.info("Generating features...")
                    local_alignment_dir = os.path.join(alignment_dir, tag)

                    if(args.use_precomputed_alignments is None):
                        if not os.path.exists(local_alignment_dir):
                            os.makedirs(local_alignment_dir)
                        
                        alignment_runner = data_pipeline.AlignmentRunner(
                            jackhmmer_binary_path=args.jackhmmer_binary_path,
                            hhblits_binary_path=args.hhblits_binary_path,
                            hhsearch_binary_path=args.hhsearch_binary_path,
                            uniref90_database_path=args.uniref90_database_path,
                            mgnify_database_path=args.mgnify_database_path,
                            bfd_database_path=args.bfd_database_path,
                            uniclust30_database_path=args.uniclust30_database_path,
                            pdb70_database_path=args.pdb70_database_path,
                            use_small_bfd=use_small_bfd,
                            no_cpus=args.cpus,
                        )
                        alignment_runner.run(
                            fasta_path, local_alignment_dir
                        )

                    feature_dict = data_processor.process_fasta(
                        fasta_path=fasta_path, alignment_dir=local_alignment_dir
                    )
            
                    # Remove temporary FASTA file
                    os.remove(fasta_path)
                
                    processed_feature_dict = feature_processor.process_features(
                        feature_dict, mode='predict',
                    )
                
                    logging.info("Executing model...")
                    batch = processed_feature_dict
                    with torch.no_grad():
                        batch = {
                            k:torch.as_tensor(v, device=args.model_device)
                            for k,v in batch.items()
                        }
                    
                        t = time.perf_counter()
                        out = model(batch)
                        logging.info(f"Inference time: {time.perf_counter() - t}")
                        
                        # Save embeddings
                        np.save(outpath_single, out['single'].cpu().detach().numpy())
                        np.save(outpath_pair, out['pair'].cpu().detach().numpy())

                except RuntimeError:
                    print(tag, 'CUDA OOM')
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_path", type=str,
    )
    parser.add_argument(
        "--template_mmcif_dir", type=str,
    )
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction.""",
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--model_name", type=str, default="model_1",
        help="""Name of a model config. Choose one of model_{1-5} or 
             model_{1-5}_ptm, as defined on the AlphaFold GitHub."""
    )
    parser.add_argument(
        "--param_path", type=str, default=None,
        help="""Path to model parameters. If None, parameters are selected
             automatically according to the model name from 
             openfold/resources/params"""
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        '--preset', type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        '--data_random_seed', type=str, default=None
    )
    add_data_args(parser)
    args = parser.parse_args()

    if(args.param_path is None):
        args.param_path = os.path.join(
            "openfold", "resources", "params", 
            "params_" + args.model_name + ".npz"
        )

    if(args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    main(args)
