from openfold.data import parsers

input_fasta = 'data/deepfri_ec/test/deepfri_final_ec_test.fasta'
outdir = 'data/deepfri_ec/test/fastas'


if __name__ == "__main__":
	with open(input_fasta, 'r') as f:
		fasta_str = f.read()
	input_seqs, input_descs = parsers.parse_fasta(fasta_str)
	for seq, desc in zip(input_seqs, input_descs):
		with open(f"{outdir}/{desc}.fasta", 'w') as f:
			f.write('>' + desc)
			f.write('\n')
			f.write(seq)
			f.write('\n')
