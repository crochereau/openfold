import argparse
import glob
import json
import os
from pathlib import Path


def main(args):
    with open(os.path.join(args.alignment_dir, args.index_file), 'rb') as f:
        indices = json.load(f)
    
    for file in glob.glob(args.fasta_dir + '/*.fasta'):
        protein = Path(file).stem
        uniprot, chain = protein.split('-')
        name = uniprot.lower() + '_' + chain
        
        # create subdirectory for MSAs
        if not os.path.exists(f"{args.out_dir}/{protein}"):
            os.makedirs(f"{args.out_dir}/{protein}")
        
        # get alignments from db files
        alignments_indices = indices[name]
        fp = open(os.path.join(args.alignment_dir, alignments_indices["db"]), "rb")
        
        def read_msa(start, size):
            fp.seek(start)
            msa = fp.read(size).decode("utf-8")
            return msa
        
        def read_template(start, size):
            fp.seek(start)
            return fp.read(size).decode("utf-8")
        
        for (name, start, size) in alignments_indices["files"]:
            ext = os.path.splitext(name)[-1]
            outfile = f"{args.out_dir}/{protein}/{name}"
            
            if not os.path.exists(outfile):
                if (ext == ".a3m") or (ext == ".sto"):
                    msa = read_msa(start, size)
                    with open(outfile, 'w') as f:
                        f.write(msa)
                
                elif (ext == ".hhr"):
                    hits = read_template(start, size)
                    with open(outfile, 'w') as f:
                        f.write(hits)
                else:
                    continue
        fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alignment_dir", type=str, default='data/pdb_alignment_dbs',
        help="""Path to alignments database."""
    )
    parser.add_argument(
        "--index_file", type=str, default='super.index',
        help="""Name of file where PDB chain ids are mapped to an index in the alignments database."""
    )
    parser.add_argument(
        "--fasta_dir", type=str, default='data/deepfri_ec/val/fastas',
        help="""Path to fasta files of target sequences."""
    )
    parser.add_argument(
        "--out_dir", type=str, default='data/deepfri_ec/val/alignments',
        help="""Name of directory in which to output the alignments."""
    )
    args = parser.parse_args()

    main(args)
