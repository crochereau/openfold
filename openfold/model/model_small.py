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
from functools import partial
import weakref

import torch
import torch.nn as nn

from openfold.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    TemplateAngleEmbedder,
    TemplatePairEmbedder,
    ExtraMSAEmbedder,
)
from openfold.model.evoformer import EvoformerStack, ExtraMSAStack
from openfold.model.heads import AuxiliaryHeadsSmall
from openfold.model.structure_module import StructureModule
from openfold.model.template import (
    TemplatePairStack,
    TemplatePointwiseAttention,
    embed_templates_average,
    embed_templates_offload,
)
import openfold.np.residue_constants as residue_constants
from openfold.utils.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    atom14_to_atom37,
)
from openfold.utils.loss import (
    compute_plddt,
)
from openfold.utils.tensor_utils import (
    add,
    dict_multimap,
    tensor_tree_map,
)

# TODO: turn back to False
SAVE_ALL_ITERS = False


class AlphaFold_small(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(AlphaFold_small, self).__init__()

        self.globals = config.globals
        self.config = config.model

        # Structure module
        self.structure_module = StructureModule(
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeadsSmall(
            self.config["heads"],
        )

    def iteration(self, feats, _recycle=True):
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if(feats[k].dtype == torch.float32):
                feats[k] = feats[k].to(dtype=dtype)

        """
        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        device = feats["target_feat"].device
        """
        # device = feats["aatype"].device

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Save representations to outputs, to match Evoformer output dict
        # single: [*, N_res, C_s]
        outputs["single"] = feats['single']
        # pair: [*, N, N, C_z]
        outputs["pair"] = feats['pair']
        del feats['pair']

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            outputs,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=outputs["single"].dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        return outputs

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        is_grad_enabled = torch.is_grad_enabled()

        # Initialize saved representations
        if SAVE_ALL_ITERS:
            reps_to_save = {}

        # Main recycling loop
        # no recycling
        num_iters = 1
        for cycle_no in range(num_iters): # num_iters = 1, ie single iteration
            print(cycle_no, num_iters)
            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):

                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs = self.iteration(
                    batch,
                    _recycle=(num_iters > 1)
                )

        # Save reps at last recycle
        if SAVE_ALL_ITERS:
            reps_to_save[f"states_{cycle_no}"] = outputs["sm"]["states"]
            reps_to_save[f"single_{cycle_no}"] = outputs["single"]
            reps_to_save[f"pair_{cycle_no}"] = outputs["pair"]

            # Return representations from all recycles
            return reps_to_save

        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))

        return outputs
