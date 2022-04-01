import argparse

from openfold.data import parsers


def main(args):
    with open(args.input_fasta, 'r') as f:
        fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(fasta_str)
    for seq, desc in zip(input_seqs, input_descs):
        with open(f"{args.out_dir}/{desc}.fasta", 'w') as f:
            f.write('>' + desc)
            f.write('\n')
            f.write(seq)
            f.write('\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_fasta", type=str, default='data/deepfri_ec/test/deepfri_final_ec_test.fasta',
        help="""Name of fasta file with target sequences."""
    )
    parser.add_argument(
        "--out_dir", type=str, default='data/deepfri_ec/test/fastas',
        help="""Name of directory in which to output the individual sequence fasta files."""
    )
    args = parser.parse_args()

    main(args)


