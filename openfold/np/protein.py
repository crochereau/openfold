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

"""Protein data type."""
import dataclasses
import io
from typing import Any, Sequence, Mapping, Optional
import re
import string

from openfold.np import residue_constants
from Bio.PDB import PDBParser
import numpy as np


FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01

@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB 
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    if chain_id is not None:
        chain = model[chain_id]
    else:
        chains = list(model.get_chains())
        if len(chains) != 1:
            raise ValueError(
                "Only single chain PDBs are supported when chain_id not specified. "
                f"Found {len(chains)} chains."
            )
        else:
            chain = chains[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []

    for res in chain:
        if res.id[2] != " ":
            raise ValueError(
                f"PDB contains an insertion code at chain {chain.id} and residue "
                f"index {res.id[1]}. These are not supported."
            )
        res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num
        )
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.0
            res_b_factors[
                residue_constants.atom_order[atom.name]
            ] = atom.bfactor
        if np.sum(mask) < 0.5:
            # If no known atom positions are reported for the residue then skip it.
            continue
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)

    parents = None
    if("PARENT" in pdb_str):
        for l in pdb_str.split("\n"):
            if("PARENT" in l and not "N/A" in l):
                parents = l.split()[1:]
                break

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        b_factors=np.array(b_factors),
        parents=parents,
    )


def from_proteinnet_string(proteinnet_str: str) -> Protein:
    tag_re = r'(\[[A-Z]+\]\n)'
    tags = [
        tag.strip() for tag in re.split(tag_re, proteinnet_str) if len(tag) > 0
    ]
    groups = zip(tags[0::2], [l.split('\n') for l in tags[1::2]])
   
    atoms = ['N', 'CA', 'C']
    aatype = None
    atom_positions = None
    atom_mask = None
    for g in groups:
        if("[PRIMARY]" == g[0]):
            seq = g[1][0].strip()
            for i in range(len(seq)):
                if(seq[i] not in residue_constants.restypes):
                    seq[i] = 'X'
            aatype = np.array([
                residue_constants.restype_order.get(
                    res_symbol, residue_constants.restype_num
                ) for res_symbol in seq
            ])
        elif("[TERTIARY]" == g[0]):
            tertiary = []
            for axis in range(3):
                tertiary.append(list(map(float, g[1][axis].split())))
            tertiary_np = np.array(tertiary)
            atom_positions = np.zeros(
                (len(tertiary[0])//3, residue_constants.atom_type_num, 3)
            ).astype(np.float32)
            for i, atom in enumerate(atoms):
                atom_positions[:, residue_constants.atom_order[atom], :] = (
                    np.transpose(tertiary_np[:, i::3])
                )
            atom_positions *= PICO_TO_ANGSTROM
        elif("[MASK]" == g[0]):
            mask = np.array(list(map({'-': 0, '+': 1}.get, g[1][0].strip())))
            atom_mask = np.zeros(
                (len(mask), residue_constants.atom_type_num,)
            ).astype(np.float32)
            for i, atom in enumerate(atoms):
                atom_mask[:, residue_constants.atom_order[atom]] = 1
            atom_mask *= mask[..., None]

    return Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=np.arange(len(aatype)),
        b_factors=None,
    )


def get_pdb_headers(prot: Protein, chain_id: int = 0) -> Sequence[str]:
    pdb_headers = []

    remark = prot.remark
    if(remark is not None):
        pdb_headers.append(f"REMARK {remark}")

    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if(parents_chain_index is not None):
        parents = [
            p for i, p in zip(parents_chain_index, parents) if i == chain_id
        ]

    if(parents is None or len(parents) == 0):
        parents = ["N/A"]

    pdb_headers.append(f"PARENT {' '.join(parents)}")

    return pdb_headers


def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
    restypes = residue_constants.restypes + ["X"]
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], "UNK")
    atom_types = residue_constants.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    headers = get_pdb_headers(prot)
    if(len(headers) > 0):
        pdb_lines.extend(headers)

    n = aatype.shape[0]
    atom_index = 1
    prev_chain_index = 0
    chain_tags = string.ascii_uppercase
    # Add all atom sites.
    for i in range(n):
        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask, b_factor in zip(
            atom_types, atom_positions[i], atom_mask[i], b_factors[i]
        ):
            if mask < 0.5:
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""
    
            chain_tag = "A"
            if(chain_index is not None):
                chain_tag = chain_tags[chain_index[i]]

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        should_terminate = (i == n - 1)
        if(chain_index is not None):
            if(i != n - 1 and chain_index[i + 1] != prev_chain_index):
                should_terminate = True
                prev_chain_index = chain_index[i + 1]

        if(should_terminate):
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_1to3(aatype[i]):>3} "
                f"{chain_tag:>1}{residue_index[i]:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

            if(i != n - 1):
                # "prev" is a misnomer here. This happens at the beginning of
                # each new chain.
                pdb_lines.extend(get_pdb_headers(prot, prev_chain_index))

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Computes an ideal atom mask.

    `Protein.atom_mask` typically is defined according to the atoms that are
    reported in the PDB. This function computes a mask according to heavy atoms
    that should be present in the given sequence of amino acids.

    Args:
      prot: `Protein` whose fields are `numpy.ndarray` objects.

    Returns:
      An ideal atom mask.
    """
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    chain_index: Optional[np.ndarray] = None,
    remark: Optional[str] = None,
    parents: Optional[Sequence[str]] = None,
    parents_chain_index: Optional[Sequence[int]] = None
) -> Protein:
    """Assembles a protein from a prediction.

    Args:
      features: Dictionary holding model inputs.
      result: Dictionary holding model outputs.
      b_factors: (Optional) B-factors to use for the protein.
      chain_index: (Optional) Chain indices for multi-chain predictions
      remark: (Optional) Remark about the prediction
      parents: (Optional) List of template names
    Returns:
      A protein instance.
    """
    if b_factors is None:
        b_factors = np.zeros_like(result["final_atom_mask"])

    return Protein(
        aatype=features["aatype"],
        atom_positions=result["final_atom_positions"],
        atom_mask=result["final_atom_mask"],
        residue_index=features["residue_index"] + 1,
        b_factors=b_factors,
        chain_index=chain_index,
        remark=remark,
        parents=parents,
        parents_chain_index=parents_chain_index,
    )
