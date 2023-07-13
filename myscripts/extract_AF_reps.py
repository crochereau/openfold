import argparse
import glob
import os
from os.path import join
from pathlib import Path 
from tqdm import tqdm

import torch


# TODO: test script

def main(args):
    # create output subdirectories
    single_dir = join(args.input_dir, "single")
    pair_dir = join(args.input_dir, "pair")
    states_dir = join(args.input_dir, "states")
    Path(single_dir).mkdir(parents=True, exist_ok=True)
    Path(pair_dir).mkdir(parents=True, exist_ok=True)
    Path(states_dir).mkdir(parents=True, exist_ok=True)

    for file in tqdm(glob.glob(join(args.input_dir, "*.pt"))):
        reps = torch.load(file)
        for name, embed in reps.items():
            # save representations from the input tensor into separate subdirectories
            protein_name = Path(file).stem
            output_name = f"{protein_name}_{name}.pt"
            if "single" in name:
                single_prot_dir = join(single_dir, protein_name)
                Path(single_prot_dir).mkdir(parents=True, exist_ok=True)
                outfile = join(single_prot_dir, output_name)
            elif "pair" in name:
                pair_prot_dir = join(pair_dir, protein_name)
                Path(pair_prot_dir).mkdir(parents=True, exist_ok=True)
                outfile = join(pair_prot_dir, output_name)
            elif "states" in name:
                states_prot_dir =join(states_dir, protein_name)
                Path(states_prot_dir).mkdir(parents=True, exist_ok=True)
                outfile = join(states_prot_dir, output_name)
            else:
                raise Exception("Unknown representation name.")
            torch.save(embed, outfile)
        # remove input tensor
        os.remove(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, default="../../function_pred/data/alphafold", help="Directory where input representations are saved.")
    parser.add_argument("output_dir", type=str, default="../../function_pred/data/alphafold", help="Directory where output representations are saved.")
    args = parser.parse_args()
    main(args)