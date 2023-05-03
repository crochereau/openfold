import os
from os.path import join
from pathlib import Path 
import torch
from tqdm import tqdm

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
            # separately save each representation saved together in the input tensor
            output_name = f"{name}.pt"
            if "single" in name:
                outfile = join(single_dir, output_name)
            elif "pair" in name:
                outfile = join(pair_dir, output_name)
            elif "states" in name:
                outfile = join(states_dir, output_name)
            else:
                raise Exception("Representation name not referenced.")
            torch.save(embed, outfile)
        # remove input tensor
        os.remove(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, required=True, help="Directory where input representations are saved.")
    parser.add_argument("output_dir", type=str, required=True, help="Directory where output representations are saved.")
    args = parser.parse_args()
    main(args)