import json
import os

#protein = '192L-A'
fasta_dir = 'data/deepfri_ec/val/fastas'
index_file = 'super.index'
alignment_dir = 'data/pdb_alignment_dbs'
outdir = 'data/deepfri_ec/val/alignments'


if __name__ == "__main__":
	with open(os.path.join(alignment_dir, index_file), 'rb') as f:
		indices = json.load(f)

	# process protein example
	uniprot, chain = protein.split('-')
	name = uniprot.lower() + '_' + chain

	# create subdirectory for MSAs
	if not os.path.exists(f"{outdir}/{protein}"):
		os.makedirs(f"{outdir}/{protein}")

	# get alignments from db files
	alignments_indices = indices[name]
	fp = open(os.path.join(alignment_dir, alignments_indices["db"]), "rb")

	def read_msa(start, size):
		fp.seek(start)
		msa = fp.read(size).decode("utf-8")
		return msa

	def read_template(start, size):
		fp.seek(start)
		return fp.read(size).decode("utf-8")


	for (name, start, size) in alignments_indices["files"]:
		ext = os.path.splitext(name)[-1]
		outfile = f"{outdir}/{protein}/{name}"

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
