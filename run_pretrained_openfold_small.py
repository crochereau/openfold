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
import logging
import math
import numpy as np
import os
from pathlib import Path

from openfold.data import parsers
from openfold.data.data_transforms import make_seq_mask, cast_to_64bit_ints
from openfold.utils.script_utils import load_models_from_command_line, parse_fasta, run_model, prep_output, \
    update_timings, relax_protein

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import pickle

import random
import time
import torch
import tqdm
from typing import Mapping, Dict
import re

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if(
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import residue_constants, protein
import openfold.np.relax.relax as relax

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)
from scripts.utils import add_data_args


TRACING_INTERVAL = 50

FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]

def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def generate_feature_dict_(
    tags,
    seqs,
    args,
):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
    assert len(seqs) == 1
    tag = tags[0]
    seq = seqs[0]
    with open(tmp_fasta_path, "w") as fp:
        fp.write(f">{tag}\n{seq}")

    feature_dict = process_fasta_(
        fasta_path=tmp_fasta_path
    )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict

def process_fasta_(
    fasta_path: str,
) -> FeatureDict:
    """Assembles features for a single sequence in a FASTA file""" 
    with open(fasta_path) as f:
        fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(fasta_str)
    if len(input_seqs) != 1:
        raise ValueError(
            f"More than one input sequence found in {fasta_path}."
        )
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    num_res = len(input_sequence)

    sequence_features = make_sequence_features_(
        sequence=input_sequence,
        description=input_description,
        num_res=num_res,
    )
    
    return {
        **sequence_features
    }

def make_sequence_features_(
    sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features

def process_aatype(feats: FeatureDict, iters: int) -> torch.Tensor:
    aatype = torch.tensor(feats['aatype'], dtype=torch.int64)
    aatype = torch.argmax(aatype, dim=-1).unsqueeze(-1).expand((aatype.shape[0], iters))
    return aatype

def make_sequence_mask(feats: FeatureDict, iters: int) -> torch.Tensor:
    mask = torch.ones((len(feats['seq_length']), iters), dtype=torch.int64)
    return mask

def process_features(feats: FeatureDict, iters: int) -> TensorDict:
    processed_feats = {}
    aatype = process_aatype(feats=feats, iters=iters)
    seq_mask = make_sequence_mask(feats=feats, iters=iters)
    processed_feats["aatype"] = aatype
    processed_feats["seq_mask"] = seq_mask
    processed_feats = cast_to_64bit_ints(processed_feats) 
    return processed_feats

def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    config = model_config(args.config_preset, long_sequence_inference=args.long_sequence_inference)

    if(args.trace_model):
        if(not config.data.predict.fixed_size):
            raise ValueError(
                "Tracing requires that fixed_size mode be enabled in the config"
            )

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    feats_dir = Path(args.feats_dir)
    single_dir = feats_dir / Path(args.single_dir)
    pair_dir = feats_dir / Path(args.pair_dir)

    tag_list = []
    seq_list = []
    for fasta_file in list_files_with_extensions(args.fasta_dir, (".fasta", ".fa")):
        # Gather input sequences
        with open(os.path.join(args.fasta_dir, fasta_file), "r") as fp:
            data = fp.read()

        tags, seqs = parse_fasta(data)
        # assert len(tags) == len(set(tags)), "All FASTA tags must be unique"
        tag = '-'.join(tags)

        tag_list.append((tag, tags))
        seq_list.append(seqs)

    seq_sort_fn = lambda target: sum([len(s) for s in target[1]])
    sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)
    feature_dicts = {}
    model_generator = load_models_from_command_line(
        config,
        args.model_device,
        args.openfold_checkpoint_path,
        args.jax_param_path,
        args.output_dir)

    for model, output_directory in model_generator:
        cur_tracing_interval = 0
        for (tag, tags), seqs in tqdm.tqdm(sorted_targets):

            single_rep_file = single_dir / Path(tag) / Path(f"{tag}_single_{args.iter}.pt")
            pair_rep_file = pair_dir / Path(tag) / Path(f"{tag}_pair_{args.iter}.pt")

            si = torch.load(single_rep_file)
            zij = torch.load(pair_rep_file)
            import pdb; pdb.set_trace()

            output_name = f'{tag}_{args.config_preset}'
            if args.output_postfix is not None:
                output_name = f'{output_name}_{args.output_postfix}'

            feature_dict = feature_dicts.get(tag, None)

            if(feature_dict is None):
                generate_feature_dict_
                feature_dict = generate_feature_dict_(
                    tags,
                    seqs,
                    args,
                )

                if(args.trace_model):
                    n = feature_dict["aatype"].shape[-2]
                    rounded_seqlen = round_up_seqlen(n)
                    feature_dict = pad_feature_dict_seq(
                        feature_dict, rounded_seqlen,
                    )

                feature_dicts[tag] = feature_dict
            
            # feature dict with only seq_mask, aatype
            num_iters = config.data.common.max_recycling_iters + 1
            processed_feature_dict = process_features(feature_dict, num_iters)

            # add saved reps to feature dict
            processed_feature_dict['single'] = si
            processed_feature_dict['pair'] = zij
            import pdb; pdb.set_trace()

            
            processed_feature_dict = {
                k:torch.as_tensor(v, device=args.model_device)
                for k,v in processed_feature_dict.items()
            }

            if(args.trace_model):
                if(rounded_seqlen > cur_tracing_interval):
                    logger.info(
                        f"Tracing model at {rounded_seqlen} residues..."
                    )
                    t = time.perf_counter()
                    trace_model_(model, processed_feature_dict)
                    tracing_time = time.perf_counter() - t
                    logger.info(
                        f"Tracing time: {tracing_time}"
                    )
                    cur_tracing_interval = rounded_seqlen
            import pdb; pdb.set_trace()
            out = run_model(model, processed_feature_dict, tag, args.output_dir, use_small_model=True)

            # Toss out the recycling dimensions --- we don't need them anymore
            processed_feature_dict = tensor_tree_map(
                lambda x: np.array(x[..., -1].cpu()),
                processed_feature_dict
            )
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

            # TODO
            # sanity check: save states reps on a few examples
            # to see if match previously saved reps

            if args.save_structure:
                unrelaxed_protein = prep_output(
                    out,
                    processed_feature_dict,
                    feature_dict,
                    feature_processor,
                    args
                )
    
                unrelaxed_output_path = os.path.join(
                    output_directory, f'{output_name}_unrelaxed.pdb'
                )
    
                with open(unrelaxed_output_path, 'w') as fp:
                    fp.write(protein.to_pdb(unrelaxed_protein))
    
                logger.info(f"Output written to {unrelaxed_output_path}...")
    
                if not args.skip_relaxation:
                    amber_relaxer = relax.AmberRelaxation(
                        use_gpu=(args.model_device != "cpu"),
                        **config.relax,
                    )
    
                    # Relax the prediction.
                    logger.info(f"Running relaxation on {unrelaxed_output_path}...")
                    t = time.perf_counter()
                    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
                    if "cuda" in args.model_device:
                        device_no = args.model_device.split(":")[-1]
                        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
                    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
                    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
                    relaxation_time = time.perf_counter() - t

                    logger.info(f"Relaxation time: {relaxation_time}")
                    update_timings({"relaxation": relaxation_time}, os.path.join(args.output_dir, "timings.json"))

                    # Save the relaxed PDB.
                    relaxed_output_path = os.path.join(
                        output_directory, f'{output_name}_relaxed.pdb'
                    )
                    with open(relaxed_output_path, 'w') as fp:
                        fp.write(relaxed_pdb_str)

                    logger.info(f"Relaxed output written to {relaxed_output_path}...")

                if args.save_outputs:
                    output_dict_path = os.path.join(
                        output_directory, f'{output_name}_output_dict.pkl'
                    )
                    with open(output_dict_path, "wb") as fp:
                        pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

                    logger.info(f"Model output written to {output_dict_path}...")
                        
            #except RuntimeError:
                #continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_dir", type=str, default="../function_pred/data/fastas/0_to_128",
        help="Path to directory of FASTA files, one sequence per file"
    )
    parser.add_argument(
        "--feats_dir", type=str, default="../function_pred/data/alphafold",
        help="Path to AF data directory."
    )
    parser.add_argument(
        "--single_dir", type=str, default="single",
        help="Path to single representation directory."
    )
    parser.add_argument(
        "--pair_dir", type=str, default="pair",
        help="Path to pair representation directory."
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--iter", type=int, default=3, choices=[0,1,2,3],
        help="""Number of iteration at which to predict the structure"""
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--config_preset", type=str, default="model_1",
        help="""Name of a model config preset defined in openfold/config.py"""
    )
    parser.add_argument(
        "--jax_param_path", type=str, default=None,
        help="""Path to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params"""
    )
    parser.add_argument(
        "--openfold_checkpoint_path", type=str, default=None,
        help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file"""
    )
    parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    parser.add_argument(
        "--save_structure", action="store_true", default=False,
        help="Whether to save the output structure."
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        "--preset", type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        "--output_postfix", type=str, default=None,
        help="""Postfix for output prediction filenames"""
    )
    parser.add_argument(
        "--data_random_seed", type=str, default=None
    )
    parser.add_argument(
        "--skip_relaxation", action="store_true", default=False,
    )
    parser.add_argument(
        "--multimer_ri_gap", type=int, default=200,
        help="""Residue index offset between multiple sequences, if provided"""
    )
    parser.add_argument(
        "--trace_model", action="store_true", default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs."""
    )
    parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    add_data_args(parser)
    args = parser.parse_args()

    if(args.jax_param_path is None and args.openfold_checkpoint_path is None):
        args.jax_param_path = os.path.join(
            "openfold", "resources", "params",
            "params_" + args.config_preset + ".npz"
        )

    if (args.model_device == "cpu" and torch.cuda.is_available()):
        logging.warning(
            """The model is being run on CPU. Consider specifying
            --model_device for better performance"""
        )

    main(args)
