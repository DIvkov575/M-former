import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional, Union
from collections import deque
import math
import random
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, from_numpy
from torch.nn.functional import one_hot
from scipy.spatial.distance import cdist
import numba
from numba import types

# Replicated from boltz/src/boltz/data/const.py
####################################################################################################
# CHAINS
####################################################################################################

chain_types = [
    "PROTEIN",
    "DNA",
    "RNA",
    "NONPOLYMER",
]
chain_type_ids = {chain: i for i, chain in enumerate(chain_types)}


out_types = [
    "dna_protein",
    "rna_protein",
    "ligand_protein",
    "dna_ligand",
    "rna_ligand",
    "intra_ligand",
    "intra_dna",
    "intra_rna",
    "intra_protein",
    "protein_protein",
    "modified",
]


out_types_weights_af3 = {
    "dna_protein": 10.0,
    "rna_protein": 10.0,
    "ligand_protein": 10.0,
    "dna_ligand": 5.0,
    "rna_ligand": 5.0,
    "intra_ligand": 20.0,
    "intra_dna": 4.0,
    "intra_rna": 16.0,
    "intra_protein": 20.0,
    "protein_protein": 20.0,
    "modified": 0.0,
}


out_types_weights = {
    "dna_protein": 5.0,
    "rna_protein": 5.0,
    "ligand_protein": 20.0,
    "dna_ligand": 2.0,
    "rna_ligand": 2.0,
    "intra_ligand": 20.0,
    "intra_dna": 2.0,
    "intra_rna": 8.0,
    "intra_protein": 20.0,
    "protein_protein": 20.0,
    "modified": 0.0,
}


out_single_types = ["protein", "ligand", "dna", "rna"]

clash_types = [
    "dna_protein",
    "rna_protein",
    "ligand_protein",
    "protein_protein",
    "dna_ligand",
    "rna_ligand",
    "ligand_ligand",
    "rna_dna",
    "dna_dna",
    "rna_rna",
]

chain_types_to_clash_type = {
    frozenset(("PROTEIN", "DNA")): "dna_protein",
    frozenset(("PROTEIN", "RNA")): "rna_protein",
    frozenset(("PROTEIN", "NONPOLYMER")): "ligand_protein",
    frozenset(("PROTEIN",)): "protein_protein",
    frozenset(("NONPOLYMER", "DNA")): "dna_ligand",
    frozenset(("NONPOLYMER", "RNA")): "rna_ligand",
    frozenset(("NONPOLYMER",)): "ligand_ligand",
    frozenset(("DNA", "RNA")): "rna_dna",
    frozenset(("DNA",)): "dna_dna",
    frozenset(("RNA",)): "rna_rna",
}

chain_type_to_out_single_type = {
    "PROTEIN": "protein",
    "DNA": "dna",
    "RNA": "rna",
    "NONPOLYMER": "ligand",
}
####################################################################################################
# RESIDUES & TOKENS
####################################################################################################


canonical_tokens = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",  # unknown protein token
]

tokens = [
    "<pad>",
    "-",
    *canonical_tokens,
    "A",
    "G",
    "C",
    "U",
    "N",  # unknown rna token
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # unknown dna token
]

token_ids = {token: i for i, token in enumerate(tokens)}
num_tokens = len(tokens)
unk_token = {"PROTEIN": "UNK", "DNA": "DN", "RNA": "N"}
unk_token_ids = {m: token_ids[t] for m, t in unk_token.items()}

prot_letter_to_token = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
    "J": "UNK",
    "B": "UNK",
    "Z": "UNK",
    "O": "UNK",
    "U": "UNK",
    "-": "-",
}

prot_token_to_letter = {v: k for k, v in prot_letter_to_token.items()}
prot_token_to_letter["UNK"] = "X"

rna_letter_to_token = {
    "A": "A",
    "G": "G",
    "C": "C",
    "U": "U",
    "N": "N",
}
rna_token_to_letter = {v: k for k, v in rna_letter_to_token.items()}

dna_letter_to_token = {
    "A": "DA",
    "G": "DG",
    "C": "DC",
    "T": "DT",
    "N": "DN",
}
dna_token_to_letter = {v: k for k, v in dna_letter_to_token.items()}

####################################################################################################
# ATOMS
####################################################################################################

num_elements = 128

chirality_types = [
    "CHI_UNSPECIFIED",
    "CHI_TETRAHEDRAL_CW",
    "CHI_TETRAHEDRAL_CCW",
    "CHI_SQUAREPLANAR",
    "CHI_OCTAHEDRAL",
    "CHI_TRIGONALBIPYRAMIDAL",
    "CHI_OTHER",
]
chirality_type_ids = {chirality: i for i, chirality in enumerate(chirality_types)}
unk_chirality_type = "CHI_OTHER"

hybridization_map = [
    "S",
    "SP",
    "SP2",
    "SP2D",
    "SP3",
    "SP3D",
    "SP3D2",
    "OTHER",
    "UNSPECIFIED",
]
hybridization_type_ids = {hybrid: i for i, hybrid in enumerate(hybridization_map)}
unk_hybridization_type = "UNSPECIFIED"

# fmt: off
ref_atoms = {
    "PAD": [],
    "UNK": ["N", "CA", "C", "O", "CB"],
    "-": [],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],  # noqa: E501
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "A": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],  # noqa: E501
    "G": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],  # noqa: E501
    "C": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],  # noqa: E501
    "U": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],  # noqa: E501
    "N": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"],  # noqa: E501
    "DA": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],  # noqa: E501
    "DG": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],  # noqa: E501
    "DC": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],  # noqa: E501
    "DT": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],  # noqa: E501
    "DN": ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
}

protein_backbone_atom_names = ["N", "CA", "C", "O"]
nucleic_backbone_atom_names = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]

protein_backbone_atom_index = {name: i for i, name in enumerate(protein_backbone_atom_names)}
nucleic_backbone_atom_index = {name: i for i, name in enumerate(nucleic_backbone_atom_names)}

ref_symmetries = {
    "PAD": [],
    "ALA": [],
    "ARG": [],
    "ASN": [],
    "ASP": [[(6, 7), (7, 6)]],
    "CYS": [],
    "GLN": [],
    "GLU": [[(7, 8), (8, 7)]],
    "GLY": [],
    "HIS": [],
    "ILE": [],
    "LEU": [],
    "LYS": [],
    "MET": [],
    "PHE": [[(6, 7), (7, 6), (8, 9), (9, 8)]],
    "PRO": [],
    "SER": [],
    "THR": [],
    "TRP": [],
    "TYR": [[(6, 7), (7, 6), (8, 9), (9, 8)]],
    "VAL": [],
    "A": [[(1, 2), (2, 1)]],
    "G": [[(1, 2), (2, 1)]],
    "C": [[(1, 2), (2, 1)]],
    "U": [[(1, 2), (2, 1)]],
    # "N": [[(1, 2), (2, 1)]],
    "DA": [[(1, 2), (2, 1)]],
    "DG": [[(1, 2), (2, 1)]],
    "DC": [[(1, 2), (2, 1)]],
    "DT": [[(1, 2), (2, 1)]],
    # "DN": [[(1, 2), (2, 1)]]
}


res_to_center_atom = {
    "UNK": "CA",
    "ALA": "CA",
    "ARG": "CA",
    "ASN": "CA",
    "ASP": "CA",
    "CYS": "CA",
    "GLN": "CA",
    "GLU": "CA",
    "GLY": "CA",
    "HIS": "CA",
    "ILE": "CA",
    "LEU": "CA",
    "LYS": "CA",
    "MET": "CA",
    "PHE": "CA",
    "PRO": "CA",
    "SER": "CA",
    "THR": "CA",
    "TRP": "CA",
    "TYR": "CA",
    "VAL": "CA",
    "A": "C1'",
    "G": "C1'",
    "C": "C1'",
    "U": "C1'",
    "N": "C1'",
    "DA": "C1'",
    "DG": "C1'",
    "DC": "C1'",
    "DT": "C1'",
    "DN": "C1'"
}

res_to_disto_atom = {
    "UNK": "CB",
    "ALA": "CB",
    "ARG": "CB",
    "ASN": "CB",
    "ASP": "CB",
    "CYS": "CB",
    "GLN": "CB",
    "GLU": "CB",
    "GLY": "CA",
    "HIS": "CB",
    "ILE": "CB",
    "LEU": "CB",
    "LYS": "CB",
    "MET": "CB",
    "PHE": "CB",
    "PRO": "CB",
    "SER": "CB",
    "THR": "CB",
    "TRP": "CB",
    "TYR": "CB",
    "VAL": "CB",
    "A": "C4",
    "G": "C4",
    "C": "C2",
    "U": "C2",
    "N": "C1'",
    "DA": "C4",
    "DG": "C4",
    "DC": "C2",
    "DT": "C2",
    "DN": "C1'"
}

res_to_center_atom_id = {
    res: ref_atoms[res].index(atom)
    for res, atom in res_to_center_atom.items()
}

res_to_disto_atom_id = {
    res: ref_atoms[res].index(atom)
    for res, atom in res_to_disto_atom.items()
}

# fmt: on

####################################################################################################
# BONDS
####################################################################################################

atom_interface_cutoff = 5.0
interface_cutoff = 15.0

bond_types = [
    "OTHER",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
    "COVALENT",
]
bond_type_ids = {bond: i for i, bond in enumerate(bond_types)}
unk_bond_type = "OTHER"


####################################################################################################
# Contacts
####################################################################################################


pocket_contact_info = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET": 2,
    "BINDER": 3,
}

contact_conditioning_info = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET>BINDER": 2,
    "BINDER>POCKET": 3,
    "CONTACT": 4,
}


####################################################################################################
# MSA
####################################################################################################

max_msa_seqs = 16384
max_paired_seqs = 8192


####################################################################################################
# CHUNKING
####################################################################################################

chunk_size_threshold = 384

####################################################################################################
# Method conditioning
####################################################################################################

# Methods
method_types_ids = {
    "MD": 0,
    "X-RAY DIFFRACTION": 1,
    "ELECTRON MICROSCOPY": 2,
    "SOLUTION NMR": 3,
    "SOLID-STATE NMR": 4,
    "NEUTRON DIFFRACTION": 4,
    "ELECTRON CRYSTALLOGRAPHY": 4,
    "FIBER DIFFRACTION": 4,
    "POWDER DIFFRACTION": 4,
    "INFRARED SPECTROSCOPY": 4,
    "FLUORESCENCE TRANSFER": 4,
    "EPR": 4,
    "THEORETICAL MODEL": 4,
    "SOLUTION SCATTERING": 4,
    "OTHER": 4,
    "AFDB": 5,
    "BOLTZ-1": 6,
    "FUTURE1": 7,  # Placeholder for future supervision sources
    "FUTURE2": 8,
    "FUTURE3": 9,
    "FUTURE4": 10,
    "FUTURE5": 11,
}
method_types_ids = {k.lower(): v for k, v in method_types_ids.items()}
num_method_types = len(set(method_types_ids.values()))

# Temperature
temperature_bins = [(265, 280), (280, 295), (295, 310)]
temperature_bins_ids = {temp: i for i, temp in enumerate(temperature_bins)}
temperature_bins_ids["other"] = len(temperature_bins)
num_temp_bins = len(temperature_bins_ids)


# pH
ph_bins = [(0, 6), (6, 8), (8, 14)]
ph_bins_ids = {ph: i for i, ph in enumerate(ph_bins)}
ph_bins_ids["other"] = len(ph_bins)
num_ph_bins = len(ph_bins_ids)

####################################################################################################
# VDW_RADII
####################################################################################################

# fmt: off
vdw_radii = [
    1.2, 1.4, 2.2, 1.9, 1.8, 1.7, 1.6, 1.55, 1.5, 1.54,
    2.4, 2.2, 2.1, 2.1, 1.95, 1.8, 1.8, 1.88, 2.8, 2.4,
    2.3, 2.15, 2.05, 2.05, 2.05, 2.05, 2.0, 2.0, 2.0, 2.1,
    2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9, 2.55, 2.4, 2.3,
    2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2, 2.25,
    2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.43,
    2.43, 2.42, 2.4, 2.38, 2.37, 2.35, 2.33, 2.32, 2.3, 2.28,
    2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05,
    2.2, 2.3, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4,
    2.0, 2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0
]
# fmt: on

####################################################################################################
# Excluded ligands
####################################################################################################

ligand_exclusion = {
    "144",
    "15P",
    "1PE",
    "2F2",
    "2JC",
    "3HR",
    "3SY",
    "7N5",
    "7PE",
    "9JE",
    "AAE",
    "ABA",
    "ACE",
    "ACN",
    "ACT",
    "ACY",
    "AZI",
    "BAM",
    "BCN",
    "BCT",
    "BDN",
    "BEN",
    "BME",
    "BO3",
    "BTB",
    "BTC",
    "BU1",
    "C8E",
    "CAD",
    "CAQ",
    "CBM",
    "CCN",
    "CIT",
    "CL",
    "CLR",
    "CM",
    "CMO",
    "CO3",
    "CPT",
    "CXS",
    "D10",
    "DEP",
    "DIO",
    "DMS",
    "DN",
    "DOD",
    "DOX",
    "EDO",
    "EEE",
    "EGL",
    "EOH",
    "EOX",
    "EPE",
    "ETF",
    "FCY",
    "FJO",
    "FLC",
    "FMT",
    "FW5",
    "GOL",
    "GSH",
    "GTT",
    "GYF",
    "HED",
    "IHP",
    "IHS",
    "IMD",
    "IOD",
    "IPA",
    "IPH",
    "LDA",
    "MB3",
    "MEG",
    "MES",
    "MLA",
    "MLI",
    "MOH",
    "MPD",
    "MRD",
    "MSE",
    "MYR",
    "N",
    "NA",
    "NH2",
    "NH4",
    "NHE",
    "NO3",
    "O4B",
    "OHE",
    "OLA",
    "OLC",
    "OMB",
    "OME",
    "OXA",
    "P6G",
    "PE3",
    "PE4",
    "PEG",
    "PEO",
    "PEP",
    "PG0",
    "PG4",
    "PGE",
    "PGR",
    "PLM",
    "PO4",
    "POL",
    "POP",
    "PVO",
    "SAR",
    "SCN",
    "SEO",
    "SEP",
    "SIN",
    "SO4",
    "SPD",
    "SPM",
    "SR",
    "STE",
    "STO",
    "STU",
    "TAR",
    "TBU",
    "TME",
    "TPO",
    "TRS",
    "UNK",
    "UNL",
    "UNX",
    "UPL",
    "URE",
}


####################################################################################################
# TEMPLATES
####################################################################################################

min_coverage_residues = 10
min_coverage_fraction = 0.1


####################################################################################################
# Ambiguous atoms
####################################################################################################

ambiguous_atoms = {
    "CA": {
        "*": "C",
        "OEX": "CA",
        "OEC": "CA",
        "543": "CA",
        "OC6": "CA",
        "OC1": "CA",
        "OC7": "CA",
        "OEY": "CA",
        "OC4": "CA",
        "OC3": "CA",
        "ICA": "CA",
        "CA": "CA",
        "OC2": "CA",
        "OC5": "CA",
    },
    "CD": {"*": "C", "CD": "CD", "CD3": "CD", "CD5": "CD", "CD1": "CD"},
    "BR": "BR",
    "CL": {
        "*": "CL",
        "C8P": "C",
        "L3T": "C",
        "TLC": "C",
        "TZ0": "C",
        "471": "C",
        "NLK": "C",
        "PGM": "C",
        "PNE": "C",
        "RCY": "C",
        "11F": "C",
        "PII": "C",
        "C1Q": "C",
        "4MD": "C",
        "R5A": "C",
        "KW2": "C",
        "I7M": "C",
        "R48": "C",
        "FC3": "C",
        "55V": "C",
        "KPF": "C",
        "SPZ": "C",
        "0TT": "C",
        "R9A": "C",
        "5NA": "C",
        "C55": "C",
        "NIX": "C",
        "5PM": "C",
        "PP8": "C",
        "544": "C",
        "812": "C",
        "NPM": "C",
        "KU8": "C",
        "A1AMM": "C",
        "4S0": "C",
        "AQC": "C",
        "2JK": "C",
        "WJR": "C",
        "A1AAW": "C",
        "85E": "C",
        "MB0": "C",
        "ZAB": "C",
        "85K": "C",
        "GBP": "C",
        "A1H80": "C",
        "A1AFR": "C",
        "L9M": "C",
        "MYK": "C",
        "MB9": "C",
        "38R": "C",
        "EKB": "C",
        "NKF": "C",
        "UMQ": "C",
        "T4K": "C",
        "3PT": "C",
        "A1A7S": "C",
        "1Q9": "C",
        "11R": "C",
        "D2V": "C",
        "SM8": "C",
        "IFC": "C",
        "DB5": "C",
        "L2T": "C",
        "GNB": "C",
        "PP7": "C",
        "072": "C",
        "P88": "C",
        "DRL": "C",
        "C9W": "C",
        "NTP": "C",
        "4HJ": "C",
        "7NA": "C",
        "LPC": "C",
        "T8W": "C",
        "63R": "C",
        "570": "C",
        "R4A": "C",
        "3BG": "C",
        "4RB": "C",
        "GSO": "C",
        "BQ6": "C",
        "R4P": "C",
        "5CP": "C",
        "TTR": "C",
        "6UZ": "C",
        "SPJ": "C",
        "0SA": "C",
        "ZL1": "C",
        "BYG": "C",
        "F0E": "C",
        "PC0": "C",
        "B2Q": "C",
        "KV6": "C",
        "NTO": "C",
        "CLG": "C",
        "R7U": "C",
        "SMQ": "C",
        "GM2": "C",
        "Z7P": "C",
        "NXF": "C",
        "C6Q": "C",
        "A1G": "C",
        "433": "C",
        "L9N": "C",
        "7OX": "C",
        "A1H84": "C",
        "97L": "C",
        "HDV": "C",
        "LUO": "C",
        "R6A": "C",
        "1PC": "C",
        "4PT": "C",
        "SBZ": "C",
        "EAB": "C",
        "FL4": "C",
        "OPS": "C",
        "C2X": "C",
        "SLL": "C",
        "BFC": "C",
        "GIP": "C",
        "7CP": "C",
        "CLH": "C",
        "34E": "C",
        "5NE": "C",
        "PBF": "C",
        "ABD": "C",
        "ABC": "C",
        "LPF": "C",
        "TIZ": "C",
        "4HH": "C",
        "AFC": "C",
        "WQH": "C",
        "9JL": "C",
        "CS3": "C",
        "NL0": "C",
        "KPY": "C",
        "DNA": "C",
        "B3C": "C",
        "TKL": "C",
        "KVS": "C",
        "HO6": "C",
        "NLH": "C",
        "1PB": "C",
        "CYF": "C",
        "G4M": "C",
        "R5B": "C",
        "N4S": "C",
        "N11": "C",
        "C8F": "C",
        "PIJ": "C",
        "WIN": "C",
        "NT1": "C",
        "WJW": "C",
        "HF7": "C",
        "TY1": "C",
        "VM1": "C",
    },
    "OS": {"*": "O", "DWC": "OS", "OHX": "OS", "OS": "OS", "8WV": "OS", "OS4": "OS"},
    "PB": {"*": "P", "ZN9": "PB", "ZN7": "PB", "PBM": "PB", "PB": "PB", "CSB": "PB"},
    "CE": {"*": "C", "CE": "CE"},
    "FE": {"*": "FE", "TFR": "F", "PF5": "F", "IFC": "F", "F5C": "F"},
    "NA": {"*": "N", "CGO": "NA", "R2K": "NA", "LVQ": "NA", "NA": "NA"},
    "ND": {"*": "N", "ND": "ND"},
    "CF": {"*": "C", "CF": "CF"},
    "RU": "RU",
    "BRAF": "BR",
    "EU": "EU",
    "CLAA": "CL",
    "CLBQ": "CL",
    "CM": {"*": "C", "ZCM": "CM"},
    "SN": {"*": "SN", "TAP": "S", "SND": "S", "TAD": "S", "XPT": "S"},
    "AG": "AG",
    "CLN": "CL",
    "CLM": "CL",
    "CLA": {"*": "CL", "PII": "C", "TDL": "C", "D0J": "C", "GM2": "C", "PIJ": "C"},
    "CLB": {
        "*": "CL",
        "TD5": "C",
        "PII": "C",
        "TDL": "C",
        "GM2": "C",
        "TD7": "C",
        "TD6": "C",
        "PIJ": "C",
    },
    "CR": {
        "*": "C",
        "BW9": "CR",
        "CQ4": "CR",
        "AC9": "CR",
        "TIL": "CR",
        "J7U": "CR",
        "CR": "CR",
    },
    "CLAY": "CL",
    "CLBC": "CL",
    "PD": {
        "*": "P",
        "F6Q": "PD",
        "SVP": "PD",
        "SXC": "PD",
        "U5U": "PD",
        "PD": "PD",
        "PLL": "PD",
    },
    "CO": {
        "*": "C",
        "J1S": "CO",
        "OCN": "CO",
        "OL3": "CO",
        "OL4": "CO",
        "B12": "CO",
        "XCO": "CO",
        "UFU": "CO",
        "CON": "CO",
        "OL5": "CO",
        "B13": "CO",
        "7KI": "CO",
        "PL1": "CO",
        "OCO": "CO",
        "J1R": "CO",
        "COH": "CO",
        "SIR": "CO",
        "6KI": "CO",
        "NCO": "CO",
        "9CO": "CO",
        "PC3": "CO",
        "BWU": "CO",
        "B1Z": "CO",
        "J83": "CO",
        "CO": "CO",
        "COY": "CO",
        "CNC": "CO",
        "3CO": "CO",
        "OCL": "CO",
        "R5Q": "CO",
        "X5Z": "CO",
        "CBY": "CO",
        "OLS": "CO",
        "F0X": "CO",
        "I2A": "CO",
        "OCM": "CO",
    },
    "CU": {
        "*": "C",
        "8ZR": "CU",
        "K7E": "CU",
        "CU3": "CU",
        "SI9": "CU",
        "35N": "CU",
        "C2O": "CU",
        "SI7": "CU",
        "B15": "CU",
        "SI0": "CU",
        "CUP": "CU",
        "SQ1": "CU",
        "CUK": "CU",
        "CUL": "CU",
        "SI8": "CU",
        "IC4": "CU",
        "CUM": "CU",
        "MM2": "CU",
        "B30": "CU",
        "S32": "CU",
        "V79": "CU",
        "IMF": "CU",
        "CUN": "CU",
        "MM1": "CU",
        "MP1": "CU",
        "IME": "CU",
        "B17": "CU",
        "C2C": "CU",
        "1CU": "CU",
        "CU6": "CU",
        "C1O": "CU",
        "CU1": "CU",
        "B22": "CU",
        "CUS": "CU",
        "RUQ": "CU",
        "CUF": "CU",
        "CUA": "CU",
        "CU": "CU",
        "CUO": "CU",
        "0TE": "CU",
        "SI4": "CU",
    },
    "CS": {"*": "C", "CS": "CS"},
    "CLQ": "CL",
    "CLR": "CL",
    "CLU": "CL",
    "TE": "TE",
    "NI": {
        "*": "N",
        "USN": "NI",
        "NFO": "NI",
        "NI2": "NI",
        "NFS": "NI",
        "NFR": "NI",
        "82N": "NI",
        "R5N": "NI",
        "NFU": "NI",
        "A1ICD": "NI",
        "NI3": "NI",
        "M43": "NI",
        "MM5": "NI",
        "BF8": "NI",
        "TCN": "NI",
        "NIK": "NI",
        "CUV": "NI",
        "MM6": "NI",
        "J52": "NI",
        "NI": "NI",
        "SNF": "NI",
        "XCC": "NI",
        "F0L": "NI",
        "UWE": "NI",
        "NFC": "NI",
        "3NI": "NI",
        "HNI": "NI",
        "F43": "NI",
        "RQM": "NI",
        "NFE": "NI",
        "NFB": "NI",
        "B51": "NI",
        "NI1": "NI",
        "WCC": "NI",
        "NUF": "NI",
    },
    "SB": {"*": "S", "UJI": "SB", "SB": "SB", "118": "SB", "SBO": "SB", "3CG": "SB"},
    "MO": "MO",
    "SEG": "SE",
    "CLL": "CL",
    "CLAH": "CL",
    "CLC": {
        "*": "CL",
        "TD5": "C",
        "PII": "C",
        "TDL": "C",
        "GM2": "C",
        "TD7": "C",
        "TD6": "C",
        "PIJ": "C",
    },
    "CLD": {"*": "CL", "PII": "C", "GM2": "C", "PIJ": "C"},
    "CLAD": "CL",
    "CLAE": "CL",
    "LA": "LA",
    "RH": "RH",
    "BRAC": "BR",
    "BRAD": "BR",
    "CLBN": "CL",
    "CLAC": "CL",
    "BRAB": "BR",
    "BRAE": "BR",
    "MG": "MG",
    "IR": "IR",
    "SE": {
        "*": "SE",
        "HII": "S",
        "NT2": "S",
        "R2P": "S",
        "S2P": "S",
        "0IU": "S",
        "QMB": "S",
        "81S": "S",
        "0QB": "S",
        "UB4": "S",
        "OHS": "S",
        "Q78": "S",
        "0Y2": "S",
        "B3M": "S",
        "NT1": "S",
        "81R": "S",
    },
    "BRAG": "BR",
    "CLF": {"*": "CL", "PII": "C", "GM2": "C", "PIJ": "C"},
    "CLE": {"*": "CL", "PII": "C", "GM2": "C", "PIJ": "C"},
    "BRAX": "BR",
    "CLK": "CL",
    "ZN": "ZN",
    "AS": "AS",
    "AU": "AU",
    "PT": "PT",
    "CLAS": "CL",
    "MN": "MN",
    "CLBE": "CL",
    "CLBF": "CL",
    "CLAF": "CL",
    "NA'": {"*": "N", "CGO": "NA"},
    "BRAH": "BR",
    "BRAI": "BR",
    "BRA": "BR",
    "BRB": "BR",
    "BRAV": "BR",
    "HG": {
        "*": "HG",
        "BBA": "H",
        "MID": "H",
        "APM": "H",
        "4QQ": "H",
        "0ZG": "H",
        "APH": "H",
    },
    "AR": "AR",
    "D": "H",
    "CLAN": "CL",
    "SI": "SI",
    "CLS": "CL",
    "ZR": "ZR",
    "CLAR": {"*": "CL", "ZM4": "C"},
    "HO": "HO",
    "CLI": {"*": "CL", "GM2": "C"},
    "CLH": {"*": "CL", "GM2": "C"},
    "CLAP": "CL",
    "CLBL": "CL",
    "CLBM": "CL",
    "PR": {"*": "PR", "UF0": "P", "252": "P"},
    "IN": "IN",
    "CLJ": "CL",
    "BRU": "BR",
    "SC": {"*": "S", "SFL": "SC"},
    "CLG": {"*": "CL", "GM2": "C"},
    "BRAT": "BR",
    "BRAR": "BR",
    "CLAG": "CL",
    "CLAB": "CL",
    "CLV": "CL",
    "TI": "TI",
    "CLAX": "CL",
    "CLAJ": "CL",
    "CL'": {"*": "CL", "BNR": "C", "25A": "C", "BDA": "C"},
    "CLAW": "CL",
    "BRF": "BR",
    "BRE": "BR",
    "RE": "RE",
    "GD": "GD",
    "SM": {"*": "S", "SM": "SM"},
    "CLBH": "CL",
    "CLBI": "CL",
    "CLAI": "CL",
    "CLY": "CL",
    "CLZ": "CL",
    "AC": "AC",
    "BR'": "BR",
    "CLT": "CL",
    "CLO": "CL",
    "CLP": "CL",
    "LU": "LU",
    "BA": {"*": "B", "BA": "BA"},
    "CLAU": "CL",
    "RB": "RB",
    "LI": "LI",
    "MOM": "MO",
    "BRAQ": "BR",
    "SR": {"*": "S", "SR": "SR", "OER": "SR"},
    "CLAT": "CL",
    "BRAL": "BR",
    "SEB": "SE",
    "CLW": "CL",
    "CLX": "CL",
    "BE": "BE",
    "BRG": "BR",
    "SEA": "SE",
    "BRAW": "BR",
    "BRBB": "BR",
    "ER": "ER",
    "TH": "TH",
    "BRR": "BR",
    "CLBV": "CL",
    "AL": "AL",
    "CLAV": "CL",
    "BRH": "BR",
    "CLAQ": "CL",
    "GA": "GA",
    "X": "*",
    "TL": "TL",
    "CLBB": "CL",
    "TB": "TB",
    "CLAK": "CL",
    "XE": {"*": "*", "XE": "XE"},
    "SEL": "SE",
    "PU": {"*": "P", "4PU": "PU"},
    "CLAZ": "CL",
    "SE'": "SE",
    "CLBA": "CL",
    "SEN": "SE",
    "SNN": "SN",
    "MOB": "MO",
    "YB": "YB",
    "BRC": "BR",
    "BRD": "BR",
    "CLAM": "CL",
    "DA": "H",
    "DB": "H",
    "DC": "H",
    "DXT": "H",
    "DXU": "H",
    "DXX": "H",
    "DXY": "H",
    "DXZ": "H",
    "DY": "DY",
    "TA": "TA",
    "XD": "*",
    "SED": "SE",
    "CLAL": "CL",
    "BRAJ": "BR",
    "AM": "AM",
    "CLAO": "CL",
    "BI": "BI",
    "KR": "KR",
    "BRBJ": "BR",
    "UNK": "*",
}

# Replicated from boltz/src/boltz/data/types.py
class NumpySerializable:
    """Serializable datatype."""

    @classmethod
    def load(cls: "NumpySerializable", path: Path) -> "NumpySerializable":
        """Load the object from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        return cls(**np.load(path, allow_pickle=True))

    def dump(self, path: Path) -> None:
        """Dump the object to an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        np.savez_compressed(str(path), **asdict(self))


class JSONSerializable: # Removed DataClassDictMixin dependency
    """Serializable datatype."""

    @classmethod
    def load(cls: "JSONSerializable", path: Path) -> "JSONSerializable":
        """Load the object from a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        """
        with path.open("r") as f:
            return cls(**json.load(f)) # Adapted to use direct dict unpacking

    def dump(self, path: Path) -> None:
        """Dump the object to a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        """
        with path.open("w") as f:
            json.dump(asdict(self), f) # Adapted to use asdict


####################################################################################################
# STRUCTURE
####################################################################################################

Atom = [
    ("name", np.dtype("4i1")),
    ("element", np.dtype("i1")),
    ("charge", np.dtype("i1")),
    ("coords", np.dtype("3f4")),
    ("conformer", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("chirality", np.dtype("i1")),
]

AtomV2 = [
    ("name", np.dtype("<U4")),
    ("coords", np.dtype("3f4")),
    ("is_present", np.dtype("?")),
    ("bfactor", np.dtype("f4")),
    ("plddt", np.dtype("f4")),
]

Bond = [
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]

BondV2 = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]

Residue = [
    ("name", np.dtype("<U5")),
    ("res_type", np.dtype("i1")),
    ("res_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("atom_center", np.dtype("i4")),
    ("atom_disto", np.dtype("i4")),
    ("is_standard", np.dtype("?")),
    ("is_present", np.dtype("?")),
]

Chain = [
    ("name", np.dtype("<U5")),
    ("mol_type", np.dtype("i1")),
    ("entity_id", np.dtype("i4")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_num", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
]

Connection = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
    ("res_1", np.dtype("i4")),
    ("res_2", np.dtype("i4")),
    ("atom_1", np.dtype("i4")),
    ("atom_2", np.dtype("i4")),
]

Interface = [
    ("chain_1", np.dtype("i4")),
    ("chain_2", np.dtype("i4")),
]

Coords = [
    ("coords", np.dtype("3f4")),
]

Ensemble = [
    ("atom_coord_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
]


@dataclass(frozen=True)
class Structure(NumpySerializable):
    """Structure datatype."""

    atoms: np.ndarray
    bonds: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    connections: np.ndarray
    interfaces: np.ndarray
    mask: np.ndarray

    @classmethod
    def load(cls: "Structure", path: Path) -> "Structure":
        """Load a structure from an NPZ file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Structure
            The loaded structure.

        """
        structure = np.load(path)
        return cls(
            atoms=structure["atoms"],
            bonds=structure["bonds"],
            residues=structure["residues"],
            chains=structure["chains"],
            connections=structure["connections"].astype(Connection),
            interfaces=structure["interfaces"],
            mask=structure["mask"],
        )

    def remove_invalid_chains(self) -> "Structure":  # noqa: PLR0915
        """Remove invalid chains.

        Parameters
        ----------
        structure : Structure
            The structure to process.

        Returns
        -------
        Structure
            The structure with masked chains removed.

        """
        entity_counter = {}
        atom_idx, res_idx, chain_idx = 0, 0, 0
        atoms, residues, chains = [], [], []
        atom_map, res_map, chain_map = {}, {}, {}
        for i, chain in enumerate(self.chains):
            # Skip masked chains
            if not self.mask[i]:
                continue

            # Update entity counter
            entity_id = chain["entity_id"]
            if entity_id not in entity_counter:
                entity_counter[entity_id] = 0
            else:
                entity_counter[entity_id] += 1

            # Update the chain
            new_chain = chain.copy()
            new_chain["atom_idx"] = atom_idx
            new_chain["res_idx"] = res_idx
            new_chain["asym_id"] = chain_idx
            new_chain["sym_id"] = entity_counter[entity_id]
            chains.append(new_chain)
            chain_map[i] = chain_idx
            chain_idx += 1

            # Add the chain residues
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            for j, res in enumerate(self.residues[res_start:res_end]):
                # Update the residue
                new_res = res.copy()
                new_res["atom_idx"] = atom_idx
                new_res["atom_center"] = (
                    atom_idx + new_res["atom_center"] - res["atom_idx"]
                )
                new_res["atom_disto"] = (
                    atom_idx + new_res["atom_disto"] - res["atom_idx"]
                )
                residues.append(new_res)
                res_map[res_start + j] = res_idx
                res_idx += 1

                # Update the atoms
                start = res["atom_idx"]
                end = res["atom_idx"] + res["atom_num"]
                atoms.append(self.atoms[start:end])
                atom_map.update({k: atom_idx + k - start for k in range(start, end)})
                atom_idx += res["atom_num"]

        # Concatenate the tables
        atoms = np.concatenate(atoms, dtype=Atom)
        residues = np.array(residues, dtype=Residue)
        chains = np.array(chains, dtype=Chain)

        # Update bonds
        bonds = []
        for bond in self.bonds:
            atom_1 = bond["atom_1"]
            atom_2 = bond["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_bond = bond.copy()
                new_bond["atom_1"] = atom_map[atom_1]
                new_bond["atom_2"] = atom_map[atom_2]
                bonds.append(new_bond)

        # Update connections
        connections = []
        for connection in self.connections:
            chain_1 = connection["chain_1"]
            chain_2 = connection["chain_2"]
            res_1 = connection["res_1"]
            res_2 = connection["res_2"]
            atom_1 = connection["atom_1"]
            atom_2 = connection["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_connection = connection.copy()
                new_connection["chain_1"] = chain_map[chain_1]
                new_connection["chain_2"] = chain_map[chain_2]
                new_connection["res_1"] = res_map[res_1]
                new_connection["res_2"] = res_map[res_2]
                new_connection["atom_1"] = atom_map[atom_1]
                new_connection["atom_2"] = atom_map[atom_2]
                connections.append(new_connection)

        # Create arrays
        bonds = np.array(bonds, dtype=Bond)
        connections = np.array(connections, dtype=Connection)
        interfaces = np.array([], dtype=Interface)
        mask = np.ones(len(chains), dtype=bool)

        return Structure(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            connections=connections,
            interfaces=interfaces,
            mask=mask,
        )


@dataclass(frozen=True)
class StructureV2(NumpySerializable):
    """Structure datatype."""

    atoms: np.ndarray
    bonds: np.ndarray
    residues: np.ndarray
    chains: np.ndarray
    interfaces: np.ndarray
    mask: np.ndarray
    coords: np.ndarray
    ensemble: np.ndarray
    pocket: Optional[np.ndarray] = None

    def remove_invalid_chains(self) -> "StructureV2":  # noqa: PLR0915
        """Remove invalid chains.

        Parameters
        ----------
        structure : Structure
            The structure to process.

        Returns
        -------
        Structure
            The structure with masked chains removed.

        """
        entity_counter = {}
        atom_idx, res_idx, chain_idx = 0, 0, 0
        atoms, residues, chains = [], [], []
        atom_map, res_map, chain_map = {}, {}, {}
        for i, chain in enumerate(self.chains):
            # Skip masked chains
            if not self.mask[i]:
                continue

            # Update entity counter
            entity_id = chain["entity_id"]
            if entity_id not in entity_counter:
                entity_counter[entity_id] = 0
            else:
                entity_counter[entity_id] += 1

            # Update the chain
            new_chain = chain.copy()
            new_chain["atom_idx"] = atom_idx
            new_chain["res_idx"] = res_idx
            new_chain["asym_id"] = chain_idx
            new_chain["sym_id"] = entity_counter[entity_id]
            chains.append(new_chain)
            chain_map[i] = chain_idx
            chain_idx += 1

            # Add the chain residues
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]
            for j, res in enumerate(self.residues[res_start:res_end]):
                # Update the residue
                new_res = res.copy()
                new_res["atom_idx"] = atom_idx
                new_res["atom_center"] = (
                    atom_idx + new_res["atom_center"] - res["atom_idx"]
                )
                new_res["atom_disto"] = (
                    atom_idx + new_res["atom_disto"] - res["atom_idx"]
                )
                residues.append(new_res)
                res_map[res_start + j] = res_idx
                res_idx += 1

                # Update the atoms
                start = res["atom_idx"]
                end = res["atom_idx"] + res["atom_num"]
                atoms.append(self.atoms[start:end])
                atom_map.update({k: atom_idx + k - start for k in range(start, end)})
                atom_idx += res["atom_num"]

        # Concatenate the tables
        atoms = np.concatenate(atoms, dtype=AtomV2)
        residues = np.array(residues, dtype=Residue)
        chains = np.array(chains, dtype=Chain)

        # Update bonds
        bonds = []
        for bond in self.bonds:
            chain_1 = bond["chain_1"]
            chain_2 = bond["chain_2"]
            res_1 = bond["res_1"]
            res_2 = bond["res_2"]
            atom_1 = bond["atom_1"]
            atom_2 = bond["atom_2"]
            if (atom_1 in atom_map) and (atom_2 in atom_map):
                new_bond = bond.copy()
                new_bond["chain_1"] = chain_map[chain_1]
                new_bond["chain_2"] = chain_map[chain_2]
                new_bond["res_1"] = res_map[res_1]
                new_bond["res_2"] = res_map[res_2]
                new_bond["atom_1"] = atom_map[atom_1]
                new_bond["atom_2"] = atom_map[atom_2]
                bonds.append(new_bond)

        # Create arrays
        bonds = np.array(bonds, dtype=BondV2)
        interfaces = np.array([], dtype=Interface)
        mask = np.ones(len(chains), dtype=bool)
        coords = [(x,) for x in atoms["coords"]]
        coords = np.array(coords, dtype=Coords)
        ensemble = np.array([(0, len(coords))], dtype=Ensemble)

        return StructureV2(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            interfaces=interfaces,
            mask=mask,
            coords=coords,
            ensemble=ensemble,
        )


####################################################################################################
# MSA
####################################################################################################


MSAResidue = [
    ("res_type", np.dtype("i1")),
]

MSADeletion = [
    ("res_idx", np.dtype("i2")),
    ("deletion", np.dtype("i2")),
]

MSASequence = [
    ("seq_idx", np.dtype("i2")),
    ("taxonomy", np.dtype("i4")),
    ("res_start", np.dtype("i4")),
    ("res_end", np.dtype("i4")),
    ("del_start", np.dtype("i4")),
    ("del_end", np.dtype("i4")),
]


@dataclass(frozen=True)
class MSA(NumpySerializable):
    """MSA datatype."""

    sequences: np.ndarray
    deletions: np.ndarray
    residues: np.ndarray


####################################################################################################
# RECORD
####################################################################################################


@dataclass(frozen=True)
class StructureInfo:
    """StructureInfo datatype."""

    resolution: Optional[float] = None
    method: Optional[str] = None
    deposited: Optional[str] = None
    released: Optional[str] = None
    revised: Optional[str] = None
    num_chains: Optional[int] = None
    num_interfaces: Optional[int] = None
    pH: Optional[float] = None
    temperature: Optional[float] = None


@dataclass(frozen=False)
class ChainInfo:
    """ChainInfo datatype."""

    chain_id: int
    chain_name: str
    mol_type: int
    cluster_id: Union[str, int]
    msa_id: Union[str, int]
    num_residues: int
    valid: bool = True
    entity_id: Optional[Union[str, int]] = None


@dataclass(frozen=True)
class InterfaceInfo:
    """InterfaceInfo datatype."""

    chain_1: int
    chain_2: int
    valid: bool = True


@dataclass(frozen=True)
class InferenceOptions:
    """InferenceOptions datatype."""

    pocket_constraints: Optional[
        list[tuple[int, list[tuple[int, int]], float, bool]]
    ] = None
    contact_constraints: Optional[
        list[tuple[tuple[int, int], tuple[int, int], float, bool]]
    ] = None


@dataclass(frozen=True)
class MDInfo:
    """MDInfo datatype."""

    forcefield: Optional[list[str]]
    temperature: Optional[float]  # Kelvin
    pH: Optional[float]
    pressure: Optional[float]  # bar
    prod_ensemble: Optional[str]
    solvent: Optional[str]
    ion_concentration: Optional[float]  # mM
    time_step: Optional[float]  # fs
    sample_frame_time: Optional[float]  # ps
    sim_time: Optional[float]  # ns
    coarse_grained: Optional[bool] = False


@dataclass(frozen=True)
class TemplateInfo:
    """InterfaceInfo datatype."""

    name: str
    query_chain: str
    query_st: int
    query_en: int
    template_chain: str
    template_st: int
    template_en: int
    force: bool = False
    threshold: Optional[float] = float("inf")


@dataclass(frozen=True)
class AffinityInfo:
    """AffinityInfo datatype."""

    chain_id: int
    mw: float


@dataclass(frozen=True)
class Record(JSONSerializable):
    """Record datatype."""

    id: str
    structure: StructureInfo
    chains: list[ChainInfo]
    interfaces: list[InterfaceInfo]
    inference_options: Optional[InferenceOptions] = None
    templates: Optional[list[TemplateInfo]] = None
    md: Optional[MDInfo] = None
    affinity: Optional[AffinityInfo] = None


####################################################################################################
# RESIDUE CONSTRAINTS
####################################################################################################


RDKitBoundsConstraint = [
    ("atom_idxs", np.dtype("2i4")),
    ("is_bond", np.dtype("?")),
    ("is_angle", np.dtype("?")),
    ("upper_bound", np.dtype("f4")),
    ("lower_bound", np.dtype("f4")),
]

ChiralAtomConstraint = [
    ("atom_idxs", np.dtype("4i4")),
    ("is_reference", np.dtype("?")),
    ("is_r", np.dtype("?")),
]

StereoBondConstraint = [
    ("atom_idxs", np.dtype("4i4")),
    ("is_reference", np.dtype("?")),
    ("is_e", np.dtype("?")),
]

PlanarBondConstraint = [
    ("atom_idxs", np.dtype("6i4")),
]

PlanarRing5Constraint = [
    ("atom_idxs", np.dtype("5i4")),
]

PlanarRing6Constraint = [
    ("atom_idxs", np.dtype("6i4")),
]


@dataclass(frozen=True)
class ResidueConstraints(NumpySerializable):
    """ResidueConstraints datatype."""

    rdkit_bounds_constraints: np.ndarray
    chiral_atom_constraints: np.ndarray
    stereo_bond_constraints: np.ndarray
    planar_bond_constraints: np.ndarray
    planar_ring_5_constraints: np.ndarray
    planar_ring_6_constraints: np.ndarray


####################################################################################################
# TARGET
####################################################################################################


@dataclass(frozen=True)
class Target:
    """Target datatype."""

    record: Record
    structure: Structure
    sequences: Optional[dict[str, str]] = None
    residue_constraints: Optional[ResidueConstraints] = None
    templates: Optional[dict[str, StructureV2]] = None
    extra_mols: Optional[dict[str, str]] = None # Removed RDKit Mol dependency


@dataclass(frozen=True)
class Manifest(JSONSerializable):
    """Manifest datatype."""

    records: list[Record]

    @classmethod
    def load(cls: "JSONSerializable", path: Path) -> "JSONSerializable":
        """Load the object from a JSON file.

        Parameters
        ----------
        path : Path
            The path to the file.

        Returns
        -------
        Serializable
            The loaded object.

        Raises
        ------
        TypeError
            If the file is not a valid manifest file.

        """
        with path.open("r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                manifest = cls(**data) # Adapted to use direct dict unpacking
            elif isinstance(data, list):
                records = []
                for r in data:
                    # Manually convert chains to ChainInfo objects
                    r_chains = [ChainInfo(**chain_dict) for chain_dict in r['chains']]
                    records.append(Record(chains=r_chains, **{k: v for k, v in r.items() if k != 'chains'}))
                manifest = cls(records=records)
            else:
                msg = "Invalid manifest file."
                raise TypeError(msg)

        return manifest


####################################################################################################
# INPUT
####################################################################################################


@dataclass(frozen=True, slots=True)
class Input:
    """Input datatype."""

    structure: Structure
    msa: dict[str, MSA]
    record: Optional[Record] = None
    residue_constraints: Optional[ResidueConstraints] = None
    templates: Optional[dict[str, StructureV2]] = None
    extra_mols: Optional[dict[str, str]] = None # Removed RDKit Mol dependency


####################################################################################################
# TOKENS
####################################################################################################

Token = [
    ("token_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i1")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("entity_id", np.dtype("i4")),
    ("mol_type", np.dtype("i1")),
    ("center_idx", np.dtype("i4")),
    ("disto_idx", np.dtype("i4")),
    ("center_coords", np.dtype("3f4")),
    ("disto_coords", np.dtype("3f4")),
    ("resolved_mask", np.dtype("?")),
    ("disto_mask", np.dtype("?")),
    ("cyclic_period", np.dtype("i4")),
]

TokenBond = [
    ("token_1", np.dtype("i4")),
    ("token_2", np.dtype("i4")),
]


TokenV2 = [
    ("token_idx", np.dtype("i4")),
    ("atom_idx", np.dtype("i4")),
    ("atom_num", np.dtype("i4")),
    ("res_idx", np.dtype("i4")),
    ("res_type", np.dtype("i4")),
    ("res_name", np.dtype("<U8")),
    ("sym_id", np.dtype("i4")),
    ("asym_id", np.dtype("i4")),
    ("entity_id", np.dtype("i4")),
    ("mol_type", np.dtype("i4")),  # the total bytes need to be divisible by 4
    ("center_idx", np.dtype("i4")),
    ("disto_idx", np.dtype("i4")),
    ("center_coords", np.dtype("3f4")),
    ("disto_coords", np.dtype("3f4")),
    ("resolved_mask", np.dtype("?")),
    ("disto_mask", np.dtype("?")),
    ("modified", np.dtype("?")),
    ("frame_rot", np.dtype("9f4")),
    ("frame_t", np.dtype("3f4")),
    ("frame_mask", np.dtype("i4")),
    ("cyclic_period", np.dtype("i4")),
    ("affinity_mask", np.dtype("?")),
]

TokenBondV2 = [
    ("token_1", np.dtype("i4")),
    ("token_2", np.dtype("i4")),
    ("type", np.dtype("i1")),
]


@dataclass(frozen=True)
class Tokenized:
    """Tokenized datatype."""

    tokens: np.ndarray
    bonds: np.ndarray
    structure: Structure
    msa: dict[str, MSA]
    record: Optional[Record] = None
    residue_constraints: Optional[ResidueConstraints] = None
    templates: Optional[dict[str, StructureV2]] = None
    template_tokens: Optional[dict[str, np.ndarray]] = None
    template_bonds: Optional[dict[str, np.ndarray]] = None
    extra_mols: Optional[dict[str, str]] = None # Removed RDKit Mol dependency

# Replicated from boltz/src/boltz/data/tokenize/tokenizer.py and boltz/src/boltz/data/tokenize/boltz.py
class Tokenizer(ABC):
    """Tokenize an input structure for training."""

    @abstractmethod
    def tokenize(self, data: Input) -> Tokenized:
        """Tokenize the input data.

        Parameters
        ----------
        data : Input
            The input data.

        Returns
        -------
        Tokenized
            The tokenized data.

        """
        raise NotImplementedError


@dataclass
class TokenData:
    """TokenData datatype."""

    token_idx: int
    atom_idx: int
    atom_num: int
    res_idx: int
    res_type: int
    sym_id: int
    asym_id: int
    entity_id: int
    mol_type: int
    center_idx: int
    disto_idx: int
    center_coords: np.ndarray
    disto_coords: np.ndarray
    resolved_mask: bool
    disto_mask: bool
    cyclic_period: int


def token_astuple(token: TokenData) -> tuple:
    """Convert a TokenData object to a tuple."""
    return (
        token.token_idx,
        token.atom_idx,
        token.atom_num,
        token.res_idx,
        token.res_type,
        token.sym_id,
        token.asym_id,
        token.entity_id,
        token.mol_type,
        token.center_idx,
        token.disto_idx,
        token.center_coords,
        token.disto_coords,
        token.resolved_mask,
        token.disto_mask,
        token.cyclic_period,
    )


class BoltzTokenizer(Tokenizer):
    """Tokenize an input structure for training."""

    def tokenize(self, data: Input) -> Tokenized:
        """Tokenize the input data.

        Parameters
        ----------
        data : Input
            The input data.

        Returns
        -------
        Tokenized
            The tokenized data.

        """
        # Get structure data
        struct = data.structure

        # Create token data
        token_data = []

        # Keep track of atom_idx to token_idx
        token_idx = 0
        atom_to_token = {}

        # Filter to valid chains only
        chains = struct.chains[struct.mask]

        for chain in chains:
            # Get residue indices
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]

            for res in struct.residues[res_start:res_end]:
                # Get atom indices
                atom_start = res["atom_idx"]
                atom_end = res["atom_idx"] + res["atom_num"]

                # Standard residues are tokens
                if res["is_standard"]:
                    # Get center and disto atoms
                    center = struct.atoms[res["atom_center"]]
                    disto = struct.atoms[res["atom_disto"]]

                    # Token is present if centers are
                    is_present = res["is_present"] & center["is_present"]
                    is_disto_present = res["is_present"] & disto["is_present"]

                    # Apply chain transformation
                    c_coords = center["coords"]
                    d_coords = disto["coords"]

                    # Create token
                    token = TokenData(
                        token_idx=token_idx,
                        atom_idx=res["atom_idx"],
                        atom_num=res["atom_num"],
                        res_idx=res["res_idx"],
                        res_type=res["res_type"],
                        sym_id=chain["sym_id"],
                        asym_id=chain["asym_id"],
                        entity_id=chain["entity_id"],
                        mol_type=chain["mol_type"],
                        center_idx=res["atom_center"],
                        disto_idx=res["atom_disto"],
                        center_coords=c_coords,
                        disto_coords=d_coords,
                        resolved_mask=is_present,
                        disto_mask=is_disto_present,
                        cyclic_period=chain["cyclic_period"],
                    )
                    token_data.append(token_astuple(token))

                    # Update atom_idx to token_idx
                    atom_to_token.update(
                        dict.fromkeys(range(atom_start, atom_end), token_idx))

                    token_idx += 1

                # Non-standard are tokenized per atom
                else:
                    # We use the unk protein token as res_type
                    unk_token_val = unk_token["PROTEIN"]
                    unk_id = token_ids[unk_token_val]

                    # Get atom coordinates
                    atom_data = struct.atoms[atom_start:atom_end]
                    atom_coords = atom_data["coords"]

                    # Tokenize each atom
                    for i, atom in enumerate(atom_data):
                        # Token is present if atom is
                        is_present = res["is_present"] & atom["is_present"]
                        index = atom_start + i

                        # Create token
                        token = TokenData(
                            token_idx=token_idx,
                            atom_idx=index,
                            atom_num=1,
                            res_idx=res["res_idx"],
                            res_type=unk_id,
                            sym_id=chain["sym_id"],
                            asym_id=chain["asym_id"],
                            entity_id=chain["entity_id"],
                            mol_type=chain["mol_type"],
                            center_idx=index,
                            disto_idx=index,
                            center_coords=atom_coords[i],
                            disto_coords=atom_coords[i],
                            resolved_mask=is_present,
                            disto_mask=is_present,
                            cyclic_period=chain[
                                "cyclic_period"
                            ],  # Enforced to be False in chain parser
                        )
                        token_data.append(token_astuple(token))

                        # Update atom_idx to token_idx
                        atom_to_token[index] = token_idx
                        token_idx += 1

        # Create token bonds
        token_bonds = []

        # Add atom-atom bonds from ligands
        for bond in struct.bonds:
            if (
                bond["atom_1"] not in atom_to_token
                or bond["atom_2"] not in atom_to_token
            ):
                continue
            token_bond = (
                atom_to_token[bond["atom_1"]],
                atom_to_token[bond["atom_2"]],
            )
            token_bonds.append(token_bond)

        # Add connection bonds (covalent)
        for conn in struct.connections:
            if (
                conn["atom_1"] not in atom_to_token
                or conn["atom_2"] not in atom_to_token
            ):
                continue
            token_bond = (
                atom_to_token[conn["atom_1"]],
                atom_to_token[conn["atom_2"]],
            )
            token_bonds.append(token_bond)

        token_data = np.array(token_data, dtype=Token)
        token_bonds = np.array(token_bonds, dtype=TokenBond)
        tokenized = Tokenized(
            tokens=token_data,
            bonds=token_bonds,
            structure=data.structure,
            msa=data.msa,
            record=data.record,
            residue_constraints=data.residue_constraints,
        )
        return tokenized

# Replicated from boltz/src/boltz/data/crop/cropper.py and boltz/src/boltz/data/crop/boltz.py
class Cropper(ABC):
    """Abstract base class for cropper."""

    @abstractmethod
    def crop(
        self,
        data: Tokenized,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : Optional[int]
            The maximum number of atoms to consider.
        chain_id : Optional[int]
            The chain ID to crop.
        interface_id : Optional[int]
            The interface ID to crop.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        raise NotImplementedError


def pick_random_token(
    tokens: np.ndarray,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from the data.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    return tokens[random.randint(len(tokens))]


def pick_chain_token(
    tokens: np.ndarray,
    chain_id: int,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from a chain.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    chain_id : int
        The chain ID.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Filter to chain
    chain_tokens = tokens[tokens["asym_id"] == chain_id]

    # Pick from chain, fallback to all tokens
    if chain_tokens.size:
        query = pick_random_token(chain_tokens, random)
    else:
        query = pick_random_token(tokens, random)

    return query


def pick_interface_token(
    tokens: np.ndarray,
    interface: np.ndarray,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from an interface.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    interface : int
        The interface ID.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Sample random interface
    chain_1 = int(interface["chain_1"])
    chain_2 = int(interface["chain_2"])

    tokens_1 = tokens[tokens["asym_id"] == chain_1]
    tokens_2 = tokens[tokens["asym_id"] == chain_2]

    # If no interface, pick from the chains
    if tokens_1.size and (not tokens_2.size):
        query = pick_random_token(tokens_1, random)
    elif tokens_2.size and (not tokens_1.size):
        query = pick_random_token(tokens_2, random)
    elif (not tokens_1.size) and (not tokens_2.size):
        query = pick_random_token(tokens, random)
    else:
        # If we have tokens, compute distances
        tokens_1_coords = tokens_1["center_coords"]
        tokens_2_coords = tokens_2["center_coords"]

        dists = cdist(tokens_1_coords, tokens_2_coords)
        cuttoff = dists < interface_cutoff

        # In rare cases, the interface cuttoff is slightly
        # too small, then we slightly expand it if it happens
        if not np.any(cuttoff):
            cuttoff = dists < (interface_cutoff + 5.0)

        tokens_1 = tokens_1[np.any(cuttoff, axis=1)]
        tokens_2 = tokens_2[np.any(cuttoff, axis=0)]

        # Select random token
        candidates = np.concatenate([tokens_1, tokens_2])
        query = pick_random_token(candidates, random)

    return query


class BoltzCropper(Cropper):
    """Interpolate between contiguous and spatial crops."""

    def __init__(self, min_neighborhood: int = 0, max_neighborhood: int = 40) -> None:
        """Initialize the cropper.

        Modulates the type of cropping to be performed.
        Smaller neighborhoods result in more spatial
        cropping. Larger neighborhoods result in more
        continuous cropping. A mix can be achieved by
        providing a range over which to sample.

        Parameters
        ----------
        min_neighborhood : int
            The minimum neighborhood size, by default 0.
        max_neighborhood : int
            The maximum neighborhood size, by default 40.

        """
        sizes = list(range(min_neighborhood, max_neighborhood + 1, 2))
        self.neighborhood_sizes = sizes

    def crop(  # noqa: PLR0915
        self,
        data: Tokenized,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : int, optional
            The maximum number of atoms to consider.
        chain_id : int, optional
            The chain ID to crop.
        interface_id : int, optional
            The interface ID to crop.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        # Check inputs
        if chain_id is not None and interface_id is not None:
            msg = "Only one of chain_id or interface_id can be provided."
            raise ValueError(msg)

        # Randomly select a neighborhood size
        neighborhood_size = random.choice(self.neighborhood_sizes)

        # Get token data
        token_data = data.tokens
        token_bonds = data.bonds
        mask = data.structure.mask
        chains = data.structure.chains
        interfaces = data.structure.interfaces

        # Filter to valid chains
        valid_chains = chains[mask]

        # Filter to valid interfaces
        valid_interfaces = interfaces
        valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_1"]]]
        valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_2"]]]

        # Filter to resolved tokens
        valid_tokens = token_data[token_data["resolved_mask"]]

        # Check if we have any valid tokens
        if not valid_tokens.size:
            msg = "No valid tokens in structure"
            raise ValueError(msg)

        # Pick a random token, chain, or interface
        if chain_id is not None:
            query = pick_chain_token(valid_tokens, chain_id, random)
        elif interface_id is not None:
            interface = interfaces[interface_id]
            query = pick_interface_token(valid_tokens, interface, random)
        elif valid_interfaces.size:
            idx = random.randint(len(valid_interfaces))
            interface = valid_interfaces[idx]
            query = pick_interface_token(valid_tokens, interface, random)
        else:
            idx = random.randint(len(valid_chains))
            chain_id = valid_chains[idx]["asym_id"]
            query = pick_chain_token(valid_tokens, chain_id, random)

        # Sort all tokens by distance to query_coords
        dists = valid_tokens["center_coords"] - query["center_coords"]
        indices = np.argsort(np.linalg.norm(dists, axis=1))

        # Select cropped indices
        cropped: set[int] = set()
        total_atoms = 0
        for idx in indices:
            # Get the token
            token = valid_tokens[idx]

            # Get all tokens from this chain
            chain_tokens = token_data[token_data["asym_id"] == token["asym_id"]]

            # Pick the whole chain if possible, otherwise select
            # a contiguous subset centered at the query token
            if len(chain_tokens) <= neighborhood_size:
                new_tokens = chain_tokens
            else:
                # First limit to the maximum set of tokens, with the
                # neighborhood on both sides to handle edges. This
                # is mostly for efficiency with the while loop below.
                min_idx = token["res_idx"] - neighborhood_size
                max_idx = token["res_idx"] + neighborhood_size

                max_token_set = chain_tokens
                max_token_set = max_token_set[max_token_set["res_idx"] >= min_idx]
                max_token_set = max_token_set[max_token_set["res_idx"] <= max_idx]

                # Start by adding just the query token
                new_tokens = max_token_set[max_token_set["res_idx"] == token["res_idx"]]

                # Expand the neighborhood until we have enough tokens, one
                # by one to handle some edge cases with non-standard chains.
                # We switch to the res_idx instead of the token_idx to always
                # include all tokens from modified residues or from ligands.
                min_idx = max_idx = token["res_idx"]
                while new_tokens.size < neighborhood_size:
                    min_idx = min_idx - 1
                    max_idx = max_idx + 1
                    new_tokens = max_token_set
                    new_tokens = new_tokens[new_tokens["res_idx"] >= min_idx]
                    new_tokens = new_tokens[new_tokens["res_idx"] <= max_idx]

            # Compute new tokens and new atoms
            new_indices = set(new_tokens["token_idx"]) - cropped
            new_tokens = token_data[list(new_indices)]
            new_atoms = np.sum(new_tokens["atom_num"])

            # Stop if we exceed the max number of tokens or atoms
            if (len(new_indices) > (max_tokens - len(cropped))) or (
                (max_atoms is not None) and ((total_atoms + new_atoms) > max_atoms)
            ):
                break

            # Add new indices
            cropped.update(new_indices)
            total_atoms += new_atoms

        # Get the cropped tokens sorted by index
        token_data = token_data[sorted(list(cropped))] # Convert set to list for sorting

        # Only keep bonds within the cropped tokens
        indices = token_data["token_idx"]
        token_bonds = token_bonds[np.isin(token_bonds["token_1"], indices)]
        token_bonds = token_bonds[np.isin(token_bonds["token_2"], indices)]

        # Return the cropped tokens
        return replace(data, tokens=token_data, bonds=token_bonds)

# Replicated from boltz/src/boltz/data/feature/featurizer.py
# Dummy function for get_symmetries, as we don't have the full boltz.data.feature.symmetry module
def get_symmetries(symmetries_cfg: str) -> dict:
    return {}

# Dummy function for center_random_augmentation, as we don't have the full boltz.model.modules.utils module
def center_random_augmentation(coords, mask, centering):
    return coords

def pad_dim(tensor: Tensor, dim: int, pad_len: int, pad_value: Union[int, float] = 0) -> Tensor:
    """Pads a tensor along a given dimension.

    Args:
        tensor (Tensor): The input tensor.
        dim (int): The dimension to pad.
        pad_len (int): The amount of padding to apply.
        pad_value (Union[int, float], optional): The value to pad with. Defaults to 0.

    Returns:
        Tensor: The padded tensor.
    """
    if pad_len <= 0:
        return tensor
    
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_len
    
    padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    
    if dim == 0:
        return torch.cat([tensor, padding], dim=dim)
    elif dim == 1:
        return torch.cat([tensor, padding], dim=dim)
    elif dim == 2:
        return torch.cat([tensor, padding], dim=dim)
    else:
        # For higher dimensions, we need to construct the padding tuple
        # This is a simplified version, might need adjustment for very complex cases
        paddings = [0] * (2 * len(tensor.shape))
        paddings[-(2 * dim + 1)] = pad_len
        return torch.nn.functional.pad(tensor, paddings, "constant", pad_value)


def select_subset_from_mask(mask, p):
    num_true = np.sum(mask)
    v = np.random.geometric(p) + 1
    k = min(v, num_true)

    true_indices = np.where(mask)[0]

    # Randomly select k indices from the true_indices
    selected_indices = np.random.choice(true_indices, size=k, replace=False)

    new_mask = np.zeros_like(mask)
    new_mask[selected_indices] = 1

    return new_mask


def compute_frames_nonpolymer(
    data: Tokenized,
    coords,
    resolved_mask,
    atom_to_token,
    frame_data: list,
    resolved_frame_data: list,
) -> tuple[list, list]:
    """Get the frames for non-polymer tokens.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    frame_data : list
        The frame data.
    resolved_frame_data : list
        The resolved frame data.

    Returns
    -------
    tuple[list, list]
        The frame data and resolved frame data.

    """
    frame_data = np.array(frame_data)
    resolved_frame_data = np.array(resolved_frame_data)
    asym_id_token = data.tokens["asym_id"]
    asym_id_atom = data.tokens["asym_id"][atom_to_token]
    token_idx = 0
    atom_idx = 0
    for id in np.unique(data.tokens["asym_id"]):
        mask_chain_token = asym_id_token == id
        mask_chain_atom = asym_id_atom == id
        num_tokens = mask_chain_token.sum()
        num_atoms = mask_chain_atom.sum()
        if (
            data.tokens[token_idx]["mol_type"] != chain_type_ids["NONPOLYMER"]
            or num_atoms < 3
        ):
            token_idx += num_tokens
            atom_idx += num_atoms
            continue
        dist_mat = (
            (
                coords.reshape(-1, 3)[mask_chain_atom][:, None, :]
                - coords.reshape(-1, 3)[mask_chain_atom][None, :, :]
            )
            ** 2
        ).sum(-1) ** 0.5
        resolved_pair = 1 - (
            resolved_mask[mask_chain_atom][None, :]
            * resolved_mask[mask_chain_atom][:, None]
        ).astype(np.float32)
        resolved_pair[resolved_pair == 1] = math.inf
        indices = np.argsort(dist_mat + resolved_pair, axis=1)
        frames = (
            np.concatenate(
                [
                    indices[:, 1:2],
                    indices[:, 0:1],
                    indices[:, 2:3],
                ],
                axis=1,
            )
            + atom_idx
        )
        frame_data[token_idx : token_idx + num_atoms, :] = frames
        resolved_frame_data[token_idx : token_idx + num_atoms] = resolved_mask[
            frames
        ].all(axis=1)
        token_idx += num_tokens
        atom_idx += num_atoms
    frames_expanded = coords.reshape(-1, 3)[frame_data]

    mask_collinear = compute_collinear_mask(
        frames_expanded[:, 1] - frames_expanded[:, 0],
        frames_expanded[:, 1] - frames_expanded[:, 2],
    )
    return frame_data, resolved_frame_data & mask_collinear


def compute_collinear_mask(v1, v2):
    norm1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2, axis=1, keepdims=True)
    v1 = v1 / (norm1 + 1e-6)
    v2 = v2 / (norm2 + 1e-6)
    mask_angle = np.abs(np.sum(v1 * v2, axis=1)) < 0.9063
    mask_overlap1 = norm1.reshape(-1) > 1e-2
    mask_overlap2 = norm2.reshape(-1) > 1e-2
    return mask_angle & mask_overlap1 & mask_overlap2


def dummy_msa(residues: np.ndarray) -> MSA:
    """Create a dummy MSA for a chain.

    Parameters
    ----------
    residues : np.ndarray
        The residues for the chain.

    Returns
    -------
    MSA
        The dummy MSA.

    """
    residues = [res["res_type"] for res in residues]
    deletions = []
    sequences = [(0, -1, 0, len(residues), 0, 0)]
    return MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )


def construct_paired_msa(  # noqa: C901, PLR0915, PLR0912
    data: Tokenized,
    max_seqs: int,
    max_pairs: int = 8192,
    max_total: int = 16384,
    random_subset: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pair the MSA data.

    Parameters
    ----------
    data : Input
        The input data.

    Returns
    -------
    Tensor
        The MSA data.
    Tensor
        The deletion data.
    Tensor
        Mask indicating paired sequences.

    """
    # Get unique chains (ensuring monotonicity in the order)
    assert np.all(np.diff(data.tokens["asym_id"], n=1) >= 0)
    chain_ids = np.unique(data.tokens["asym_id"])

    # Get relevant MSA, and create a dummy for chains without
    msa = {k: data.msa[k] for k in chain_ids if k in data.msa}
    for chain_id in chain_ids:
        if chain_id not in msa:
            chain = data.structure.chains[chain_id]
            res_start = chain["res_idx"]
            res_end = res_start + chain["res_num"]
            residues = data.structure.residues[res_start:res_end]
            msa[chain_id] = dummy_msa(residues)

    # Map taxonomies to (chain_id, seq_idx)
    taxonomy_map: dict[str, list] = {}
    for chain_id, chain_msa in msa.items():
        sequences = chain_msa.sequences
        sequences = sequences[sequences["taxonomy"] != -1]
        for sequence in sequences:
            seq_idx = sequence["seq_idx"]
            taxon = sequence["taxonomy"]
            taxonomy_map.setdefault(taxon, []).append((chain_id, seq_idx))

    # Remove taxonomies with only one sequence and sort by the
    # number of chain_id present in each of the taxonomies
    taxonomy_map = {k: v for k, v in taxonomy_map.items() if len(v) > 1}
    taxonomy_map = sorted(
        taxonomy_map.items(),
        key=lambda x: len({c for c, _ in x[1]}),
        reverse=True,
    )

    # Keep track of the sequences available per chain, keeping the original
    # order of the sequences in the MSA to favor the best matching sequences
    visited = {(c, s) for c, items in taxonomy_map for s in items}
    available = {}
    for c in chain_ids:
        available[c] = deque(
            i for i in range(1, len(msa[c].sequences)) if (c, i) not in visited
        )

    # Create sequence pairs
    is_paired = []
    pairing = []

    # Start with the first sequence for each chain
    is_paired.append({c: 1 for c in chain_ids})
    pairing.append({c: 0 for c in chain_ids})

    # Then add up to 8191 paired rows
    for _, pairs in taxonomy_map:
        # Group occurences by chain_id in case we have multiple
        # sequences from the same chain and same taxonomy
        chain_occurences = {}
        for chain_id, seq_idx in pairs:
            chain_occurences.setdefault(chain_id, []).append(seq_idx)

        # We create as many pairings as the maximum number of occurences
        max_occurences = max(len(v) for v in chain_occurences.values())
        for i in range(max_occurences):
            row_pairing = {}
            row_is_paired = {}

            # Add the chains present in the taxonomy
            for chain_id, seq_idxs in chain_occurences.items():
                # Roll over the sequence index to maximize diversity
                idx = i % len(seq_idxs)
                seq_idx = seq_idxs[idx]

                # Add the sequence to the pairing
                row_pairing[chain_id] = seq_idx
                row_is_paired[chain_id] = 1

            # Add any missing chains
            for chain_id in chain_ids:
                if chain_id not in row_pairing:
                    row_is_paired[chain_id] = 0
                    if available[chain_id]:
                        # Add the next available sequence
                        row_pairing[chain_id] = available[chain_id].popleft()
                    else:
                        # No more sequences available, we place a gap
                        row_pairing[chain_id] = -1

            pairing.append(row_pairing)
            is_paired.append(row_is_paired)

            # Break if we have enough pairs
            if len(pairing) >= max_paired_seqs:
                break

        # Break if we have enough pairs
        if len(pairing) >= max_paired_seqs:
            break

    # Now add up to 16384 unpaired rows total
    max_left = max(len(v) for v in available.values())
    for _ in range(min(max_total - len(pairing), max_left)):
        row_pairing = {}
        row_is_paired = {}
        for chain_id in chain_ids:
            row_is_paired[chain_id] = 0
            if available[chain_id]:
                # Add the next available sequence
                row_pairing[chain_id] = available[chain_id].popleft()
            else:
                # No more sequences available, we place a gap
                row_pairing[chain_id] = -1

        pairing.append(row_pairing)
        is_paired.append(row_is_paired)

        # Break if we have enough sequences
        if len(pairing) >= max_total:
            break

    # Randomly sample a subset of the pairs
    # ensuring the first row is always present
    if random_subset:
        num_seqs = len(pairing)
        if num_seqs > max_seqs:
            indices = np.random.choice(
                list(range(1, num_seqs)), size=max_seqs - 1, replace=False
            )  # noqa: NPY002
            pairing = [pairing[0]] + [pairing[i] for i in indices]
            is_paired = [is_paired[0]] + [is_paired[i] for i in indices]
    else:
        # Deterministic downsample to max_seqs
        pairing = pairing[:max_seqs]
        is_paired = is_paired[:max_seqs]

    # Map (chain_id, seq_idx, res_idx) to deletion
    deletions = numba.typed.Dict.empty(
        key_type=numba.types.Tuple(
            [numba.types.int64, numba.types.int64, numba.types.int64]),
        value_type=numba.types.int64
    )
    for chain_id, chain_msa in msa.items():
        chain_deletions = chain_msa.deletions
        for sequence in chain_msa.sequences:
            seq_idx = sequence["seq_idx"]
            del_start = sequence["del_start"]
            del_end = sequence["del_end"]
            chain_deletions = chain_deletions[del_start:del_end]
            for deletion_data in chain_deletions:
                res_idx = deletion_data["res_idx"]
                deletion_values = deletion_data["deletion"]
                deletions[(chain_id, seq_idx, res_idx)] = deletion_values

    # Add all the token MSA data
    msa_data, del_data, paired_data = prepare_msa_arrays(
        data.tokens, pairing, is_paired, deletions, msa
    )

    msa_data = torch.tensor(msa_data, dtype=torch.long)
    del_data = torch.tensor(del_data, dtype=torch.float)
    paired_data = torch.tensor(paired_data, dtype=torch.float)

    return msa_data, del_data, paired_data


def prepare_msa_arrays(
    tokens,
    pairing: list[dict[int, int]],
    is_paired: list[dict[int, int]],
    deletions: dict[tuple[int, int, int], int],
    msa: dict[int, MSA],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Reshape data to play nicely with numba jit."""
    token_asym_ids_arr = np.array([t["asym_id"] for t in tokens], dtype=np.int64)
    token_res_idxs_arr = np.array([t["res_idx"] for t in tokens], dtype=np.int64)

    chain_ids = sorted(msa.keys())

    # chain_ids are not necessarily contiguous (e.g. they might be 0, 24, 25).
    # This allows us to look up a chain_id by it's index in the chain_ids list.
    chain_id_to_idx = {chain_id: i for i, chain_id in enumerate(chain_ids)}
    token_asym_ids_idx_arr = np.array(
        [chain_id_to_idx[asym_id] for asym_id in token_asym_ids_arr], dtype=np.int64
    )

    pairing_arr = np.zeros((len(pairing), len(chain_ids)), dtype=np.int64)
    is_paired_arr = np.zeros((len(is_paired), len(chain_ids)), dtype=np.int64)

    for i, row_pairing in enumerate(pairing):
        for chain_id in chain_ids:
            pairing_arr[i, chain_id_to_idx[chain_id]] = row_pairing[chain_id]

    for i, row_is_paired in enumerate(is_paired):
        for chain_id in chain_ids:
            is_paired_arr[i, chain_id_to_idx[chain_id]] = row_is_paired[chain_id]

    max_seq_len = max(len(msa[chain_id].sequences) for chain_id in chain_ids)

    # we want res_start from sequences
    msa_sequences = np.full((len(chain_ids), max_seq_len), -1, dtype=np.int64)
    for chain_id in chain_ids:
        for i, seq in enumerate(msa[chain_id].sequences):
            msa_sequences[chain_id_to_idx[chain_id], i] = seq["res_start"]

    max_residues_len = max(len(msa[chain_id].residues) for chain_id in chain_ids)
    msa_residues = np.full((len(chain_ids), max_residues_len), -1, dtype=np.int64)
    for chain_id in chain_ids:
        residues = msa[chain_id].residues.astype(np.int64)
        idxs = np.arange(len(residues))
        chain_idx = chain_id_to_idx[chain_id]
        msa_residues[chain_idx, idxs] = residues

    return _prepare_msa_arrays_inner(
        token_asym_ids_arr,
        token_res_idxs_arr,
        token_asym_ids_idx_arr,
        pairing_arr,
        is_paired_arr,
        deletions,
        msa_sequences,
        msa_residues,
        token_ids["-"],
    )


deletions_dict_type = types.DictType(types.UniTuple(types.int64, 3), types.int64)


@numba.njit(
    [
        types.Tuple(
            (
                types.int64[:, ::1],  # msa_data
                types.int64[:, ::1],  # del_data
                types.int64[:, ::1],  # paired_data
            )
        )(
            types.int64[::1],  # token_asym_ids
            types.int64[::1],  # token_res_idxs
            types.int64[::1],  # token_asym_ids_idx
            types.int64[:, ::1],  # pairing
            types.int64[:, ::1],  # is_paired
            deletions_dict_type,  # deletions
            types.int64[:, ::1],  # msa_sequences
            types.int64[:, ::1],  # msa_residues
            types.int64,  # gap_token
        )
    ],
    cache=True,
)
def _prepare_msa_arrays_inner(
    token_asym_ids: npt.NDArray[np.int64],
    token_res_idxs: npt.NDArray[np.int64],
    token_asym_ids_idx: npt.NDArray[np.int64],
    pairing: npt.NDArray[np.int64],
    is_paired: npt.NDArray[np.int64],
    deletions: dict[tuple[int, int, int], int],
    msa_sequences: npt.NDArray[np.int64],
    msa_residues: npt.NDArray[np.int64],
    gap_token: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    n_tokens = len(token_asym_ids)
    n_pairs = len(pairing)
    msa_data = np.full((n_tokens, n_pairs), gap_token, dtype=np.int64)
    paired_data = np.zeros((n_tokens, n_pairs), dtype=np.int64)
    del_data = np.zeros((n_tokens, n_pairs), dtype=np.int64)

    # Add all the token MSA data
    for token_idx in range(n_tokens):
        chain_id_idx = token_asym_ids_idx[token_idx]
        chain_id = token_asym_ids[token_idx]
        res_idx = token_res_idxs[token_idx]

        for pair_idx in range(n_pairs):
            seq_idx = pairing[pair_idx, chain_id_idx]
            paired_data[token_idx, pair_idx] = is_paired[pair_idx, chain_id_idx]

            # Add residue type
            if seq_idx != -1:
                res_start = msa_sequences[chain_id_idx, seq_idx]
                res_type = msa_residues[chain_id_idx, res_start + res_idx]
                k = (chain_id, seq_idx, res_idx)
                if k in deletions:
                    del_data[token_idx, pair_idx] = deletions[k]
                msa_data[token_idx, pair_idx] = res_type

    return msa_data, del_data, paired_data


####################################################################################################
# FEATURES
####################################################################################################


def process_token_features(
    data: Tokenized,
    max_tokens: Optional[int] = None,
    binder_pocket_conditioned_prop: Optional[float] = 0.0,
    binder_pocket_cutoff: Optional[float] = 6.0,
    binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
    only_ligand_binder_pocket: Optional[bool] = False,
    inference_binder: Optional[list[int]] = None,
    inference_pocket: Optional[list[tuple[int, int]]] = None,
) -> dict[str, Tensor]:
    """Get the token features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_tokens : int
        The maximum number of tokens.

    Returns
    -------
    dict[str, Tensor]
        The token features.

    """
    # Token data
    token_data = data.tokens
    token_bonds = data.bonds

    # Token core features
    token_index = torch.arange(len(token_data), dtype=torch.long)
    residue_index = from_numpy(token_data["res_idx"].copy()).long()
    asym_id = from_numpy(token_data["asym_id"].copy()).long()
    entity_id = from_numpy(token_data["entity_id"].copy()).long()
    sym_id = from_numpy(token_data["sym_id"].copy()).long()
    mol_type = from_numpy(token_data["mol_type"].copy()).long()
    res_type = from_numpy(token_data["res_type"].copy()).long()
    res_type = one_hot(res_type, num_classes=num_tokens)
    disto_center = from_numpy(token_data["disto_coords"].copy())

    # Token mask features
    pad_mask = torch.ones(len(token_data), dtype=torch.float)
    resolved_mask = from_numpy(token_data["resolved_mask"].copy()).float()
    disto_mask = from_numpy(token_data["disto_mask"].copy()).float()
    cyclic_period = from_numpy(token_data["cyclic_period"].copy())

    # Token bond features
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        num_tokens_actual = max_tokens if pad_len > 0 else len(token_data)
    else:
        num_tokens_actual = len(token_data)

    tok_to_idx = {tok["token_idx"]: idx for idx, tok in enumerate(token_data)}
    bonds = torch.zeros(num_tokens_actual, num_tokens_actual, dtype=torch.float)
    for token_bond in token_bonds:
        token_1 = tok_to_idx[token_bond["token_1"]]
        token_2 = tok_to_idx[token_bond["token_2"]]
        bonds[token_1, token_2] = 1
        bonds[token_2, token_1] = 1

    bonds = bonds.unsqueeze(-1)

    # Pocket conditioned feature
    pocket_feature = (
        np.zeros(len(token_data)) + pocket_contact_info["UNSPECIFIED"]
    )
    if inference_binder is not None:
        assert inference_pocket is not None
        pocket_residues = set(inference_pocket)
        for idx, token in enumerate(token_data):
            if token["asym_id"] == inference_binder:
                pocket_feature[idx] = pocket_contact_info["BINDER"]
            elif (token["asym_id"], token["res_idx"]) in pocket_residues:
                pocket_feature[idx] = pocket_contact_info["POCKET"]
            else:
                pocket_feature[idx] = pocket_contact_info["UNSELECTED"]
    elif (
        binder_pocket_conditioned_prop > 0.0
        and random.random() < binder_pocket_conditioned_prop
    ):
        # choose as binder a random ligand in the crop, if there are no ligands select a protein chain
        binder_asym_ids = np.unique(
            token_data["asym_id"][
                token_data["mol_type"] == chain_type_ids["NONPOLYMER"]
            ]
        )

        if len(binder_asym_ids) == 0:
            if not only_ligand_binder_pocket:
                binder_asym_ids = np.unique(token_data["asym_id"])

        if len(binder_asym_ids) > 0:
            pocket_asym_id = random.choice(binder_asym_ids)
            binder_mask = token_data["asym_id"] == pocket_asym_id

            binder_coords = []
            for token in token_data:
                if token["asym_id"] == pocket_asym_id:
                    binder_coords.append(
                        data.structure.atoms["coords"][
                            token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                        ]
                    )
            binder_coords = np.concatenate(binder_coords, axis=0)

            # find the tokens in the pocket
            token_dist = np.zeros(len(token_data)) + 1000
            for i, token in enumerate(token_data):
                if (
                    token["mol_type"] != chain_type_ids["NONPOLYMER"]
                    and token["asym_id"] != pocket_asym_id
                    and token["resolved_mask"] == 1
                ):
                    token_coords = data.structure.atoms["coords"][
                        token["atom_idx"] : token["atom_idx"] + token["atom_num"]
                    ]

                    # find chain and apply chain transformation
                    for chain in data.structure.chains:
                        if chain["asym_id"] == token["asym_id"]:
                            break

                    token_dist[i] = np.min(
                        np.linalg.norm(
                            token_coords[:, None, :] - binder_coords[None, :, :],
                            axis=-1,
                        )
                    )

            pocket_mask = token_dist < binder_pocket_cutoff

            if np.sum(pocket_mask) > 0:
                pocket_feature = (
                    np.zeros(len(token_data)) + pocket_contact_info["UNSELECTED"]
                )
                pocket_feature[binder_mask] = pocket_contact_info["BINDER"]

                if binder_pocket_sampling_geometric_p > 0.0:
                    # select a subset of the pocket, according
                    # to a geometric distribution with one as minimum
                    pocket_mask = select_subset_from_mask(
                        pocket_mask, binder_pocket_sampling_geometric_p
                    )

                pocket_feature[pocket_mask] = pocket_contact_info["POCKET"]
    pocket_feature = from_numpy(pocket_feature).long()
    pocket_feature = one_hot(pocket_feature, num_classes=len(pocket_contact_info))

    # Pad to max tokens if given
    if max_tokens is not None:
        pad_len = max_tokens - len(token_data)
        if pad_len > 0:
            token_index = pad_dim(token_index, 0, pad_len)
            residue_index = pad_dim(residue_index, 0, pad_len)
            asym_id = pad_dim(asym_id, 0, pad_len)
            entity_id = pad_dim(entity_id, 0, pad_len)
            sym_id = pad_dim(sym_id, 0, pad_len)
            mol_type = pad_dim(mol_type, 0, pad_len)
            res_type = pad_dim(res_type, 0, pad_len)
            disto_center = pad_dim(disto_center, 0, pad_len)
            pad_mask = pad_dim(pad_mask, 0, pad_len)
            resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            disto_mask = pad_dim(disto_mask, 0, pad_len)
            pocket_feature = pad_dim(pocket_feature, 0, pad_len)
            cyclic_period = pad_dim(cyclic_period, 0, pad_len)

    token_features = {
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "res_type": res_type,
        "disto_center": disto_center,
        "token_bonds": bonds,
        "token_pad_mask": pad_mask,
        "token_resolved_mask": resolved_mask,
        "token_disto_mask": disto_mask,
        "pocket_feature": pocket_feature,
        "cyclic_period": cyclic_period,
    }
    return token_features


def process_atom_features(
    data: Tokenized,
    atoms_per_window_queries: int = 32,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
    max_atoms: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Tensor]:
    """Get the atom features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_atoms : int, optional
        The maximum number of atoms.

    Returns
    -------
    dict[str, Tensor]
        The atom features.

    """
    # Filter to tokens' atoms
    atom_data = []
    ref_space_uid = []
    coord_data = []
    frame_data = []
    resolved_frame_data = []
    atom_to_token = []
    token_to_rep_atom = []  # index on cropped atom table
    r_set_to_rep_atom = []
    disto_coords = []
    atom_idx = 0

    chain_res_ids = {}
    for token_id, token in enumerate(data.tokens):
        # Get the chain residue ids
        chain_idx, res_id = token["asym_id"], token["res_idx"]
        chain = data.structure.chains[chain_idx]

        if (chain_idx, res_id) not in chain_res_ids:
            new_idx = len(chain_res_ids)
            chain_res_ids[(chain_idx, res_id)] = new_idx
        else:
            new_idx = chain_res_ids[(chain_idx, res_id)]

        # Map atoms to token indices
        ref_space_uid.extend([new_idx] * token["atom_num"])
        atom_to_token.extend([token_id] * token["atom_num"])

        # Add atom data
        start = token["atom_idx"]
        end = token["atom_idx"] + token["atom_num"]
        token_atoms = data.structure.atoms[start:end]

        # Map token to representative atom
        token_to_rep_atom.append(atom_idx + token["disto_idx"] - start)
        if (chain["mol_type"] != chain_type_ids["NONPOLYMER"]) and token[
            "resolved_mask"
        ]:
            r_set_to_rep_atom.append(atom_idx + token["center_idx"] - start)

        # Get token coordinates
        token_coords = np.array([token_atoms["coords"]])
        coord_data.append(token_coords)

        # Get frame data
        res_type = tokens[token["res_type"]]

        if token["atom_num"] < 3 or res_type in ["PAD", "UNK", "-"]:
            idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
            mask_frame = False
        elif (token["mol_type"] == chain_type_ids["PROTEIN"]) and (
            res_type in ref_atoms
        ):
            idx_frame_a, idx_frame_b, idx_frame_c = (
                ref_atoms[res_type].index("N"),
                ref_atoms[res_type].index("CA"),
                ref_atoms[res_type].index("C"),
            )
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        elif (
            token["mol_type"] == chain_type_ids["DNA"]
            or token["mol_type"] == chain_type_ids["RNA"]
        ) and (res_type in ref_atoms):
            idx_frame_a, idx_frame_b, idx_frame_c = (
                ref_atoms[res_type].index("C1'"),
                ref_atoms[res_type].index("C3'"),
                ref_atoms[res_type].index("C4'"),
            )
            mask_frame = (
                token_atoms["is_present"][idx_frame_a]
                and token_atoms["is_present"][idx_frame_b]
                and token_atoms["is_present"][idx_frame_c]
            )
        else:
            idx_frame_a, idx_frame_b, idx_frame_c = 0, 0, 0
            mask_frame = False
        frame_data.append(
            [idx_frame_a + atom_idx, idx_frame_b + atom_idx, idx_frame_c + atom_idx]
        )
        resolved_frame_data.append(mask_frame)

        # Get distogram coordinates
        disto_coords_tok = data.structure.atoms[token["disto_idx"]]["coords"]
        disto_coords.append(disto_coords_tok)

        # Update atom data. This is technically never used again (we rely on coord_data),
        # but we update for consistency and to make sure the Atom object has valid, transformed coordinates.
        token_atoms = token_atoms.copy()
        token_atoms["coords"] = token_coords[0]  # atom has a copy of first coords
        atom_data.append(token_atoms)
        atom_idx += len(token_atoms)

    disto_coords = np.array(disto_coords)

    # Compute distogram
    t_center = torch.Tensor(disto_coords)
    t_dists = torch.cdist(t_center, t_center)
    boundaries = torch.linspace(min_dist, max_dist, num_bins - 1)
    distogram = (t_dists.unsqueeze(-1) > boundaries).sum(dim=-1).long()
    disto_target = one_hot(distogram, num_classes=num_bins)

    atom_data = np.concatenate(atom_data)
    coord_data = np.concatenate(coord_data, axis=1)
    ref_space_uid = np.array(ref_space_uid)

    # Compute features
    ref_atom_name_chars = from_numpy(atom_data["name"]).long()
    ref_element = from_numpy(atom_data["element"]).long()
    ref_charge = from_numpy(atom_data["charge"])
    ref_pos = from_numpy(
        atom_data["conformer"].copy()
    )  # not sure why I need to copy here..
    ref_space_uid = from_numpy(ref_space_uid)
    coords = from_numpy(coord_data.copy())
    resolved_mask = from_numpy(atom_data["is_present"])
    pad_mask = torch.ones(len(atom_data), dtype=torch.float)
    atom_to_token = torch.tensor(atom_to_token, dtype=torch.long)
    token_to_rep_atom = torch.tensor(token_to_rep_atom, dtype=torch.long)
    r_set_to_rep_atom = torch.tensor(r_set_to_rep_atom, dtype=torch.long)
    frame_data, resolved_frame_data = compute_frames_nonpolymer(
        data,
        coord_data,
        atom_data["is_present"],
        atom_to_token,
        frame_data,
        resolved_frame_data,
    )  # Compute frames for NONPOLYMER tokens
    frames = from_numpy(frame_data.copy())
    frame_resolved_mask = from_numpy(resolved_frame_data.copy())
    # Convert to one-hot
    ref_atom_name_chars = one_hot(
        ref_atom_name_chars % num_bins, num_classes=num_bins
    )  # added for lower case letters
    ref_element = one_hot(ref_element, num_classes=num_elements)
    atom_to_token = one_hot(atom_to_token, num_classes=token_id + 1)
    token_to_rep_atom = one_hot(token_to_rep_atom, num_classes=len(atom_data))
    r_set_to_rep_atom = one_hot(r_set_to_rep_atom, num_classes=len(atom_data))

    # Center the ground truth coordinates
    center = (coords * resolved_mask[None, :, None]).sum(dim=1)
    center = center / resolved_mask.sum().clamp(min=1)
    coords = coords - center[:, None]

    # Apply random roto-translation to the input atoms
    ref_pos = center_random_augmentation(
        ref_pos[None], resolved_mask[None], centering=False
    )[0]

    # Compute padding and apply
    if max_atoms is not None:
        assert max_atoms % atoms_per_window_queries == 0
        pad_len = max_atoms - len(atom_data)
    else:
        pad_len = (
            (len(atom_data) - 1) // atoms_per_window_queries + 1
        ) * atoms_per_window_queries - len(atom_data)

    if pad_len > 0:
        pad_mask = pad_dim(pad_mask, 0, pad_len)
        ref_pos = pad_dim(ref_pos, 0, pad_len)
        resolved_mask = pad_dim(resolved_mask, 0, pad_len)
        ref_element = pad_dim(ref_element, 0, pad_len)
        ref_charge = pad_dim(ref_charge, 0, pad_len)
        ref_atom_name_chars = pad_dim(ref_atom_name_chars, 0, pad_len)
        ref_space_uid = pad_dim(ref_space_uid, 0, pad_len)
        coords = pad_dim(coords, 1, pad_len)
        atom_to_token = pad_dim(atom_to_token, 0, pad_len)
        token_to_rep_atom = pad_dim(token_to_rep_atom, 1, pad_len)
        r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 1, pad_len)

    if max_tokens is not None:
        pad_len = max_tokens - token_to_rep_atom.shape[0]
        if pad_len > 0:
            atom_to_token = pad_dim(atom_to_token, 1, pad_len)
            token_to_rep_atom = pad_dim(token_to_rep_atom, 0, pad_len)
            r_set_to_rep_atom = pad_dim(r_set_to_rep_atom, 0, pad_len)
            disto_target = pad_dim(pad_dim(disto_target, 0, pad_len), 1, pad_len)
            frames = pad_dim(frames, 0, pad_len)
            frame_resolved_mask = pad_dim(frame_resolved_mask, 0, pad_len)

    return {
        "ref_pos": ref_pos,
        "atom_resolved_mask": resolved_mask,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "coords": coords,
        "atom_pad_mask": pad_mask,
        "atom_to_token": atom_to_token,
        "token_to_rep_atom": token_to_rep_atom,
        "r_set_to_rep_atom": r_set_to_rep_atom,
        "disto_target": disto_target,
        "frames_idx": frames,
        "frame_resolved_mask": frame_resolved_mask,
    }


def process_msa_features(
    data: Tokenized,
    max_seqs_batch: int,
    max_seqs: int,
    max_tokens: Optional[int] = None,
    pad_to_max_seqs: bool = False,
) -> dict[str, Tensor]:
    """Get the MSA features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.
    max_seqs : int
        The maximum number of MSA sequences.
    max_tokens : int
        The maximum number of tokens.
    pad_to_max_seqs : bool
        Whether to pad to the maximum number of sequences.

    Returns
    -------
    dict[str, Tensor]
        The MSA features.

    """
    # Created paired MSA
    msa, deletion, paired = construct_paired_msa(data, max_seqs_batch)
    msa, deletion, paired = (
        msa.transpose(1, 0),
        deletion.transpose(1, 0),
        paired.transpose(1, 0),
    )  # (N_MSA, N_RES, N_AA)

    # Prepare features
    msa = torch.nn.functional.one_hot(msa, num_classes=num_tokens)
    msa_mask = torch.ones_like(msa[:, :, 0])
    profile = msa.float().mean(dim=0)
    has_deletion = deletion > 0
    deletion = np.pi / 2 * np.arctan(deletion / 3)
    deletion_mean = deletion.mean(axis=0)

    # Pad in the MSA dimension (dim=0)
    if pad_to_max_seqs:
        pad_len = max_seqs - msa.shape[0]
        if pad_len > 0:
            msa = pad_dim(msa, 0, pad_len, token_ids["-"])
            paired = pad_dim(paired, 0, pad_len)
            msa_mask = pad_dim(msa_mask, 0, pad_len)
            has_deletion = pad_dim(has_deletion, 0, pad_len)
            deletion = pad_dim(deletion, 0, pad_len)

    # Pad in the token dimension (dim=1)
    if max_tokens is not None:
        pad_len = max_tokens - msa.shape[1]
        if pad_len > 0:
            msa = pad_dim(msa, 1, pad_len, token_ids["-"])
            paired = pad_dim(paired, 1, pad_len)
            msa_mask = pad_dim(msa_mask, 1, pad_len)
            has_deletion = pad_dim(has_deletion, 1, pad_len)
            deletion = pad_dim(deletion, 1, pad_len)
            profile = pad_dim(profile, 0, pad_len)
            deletion_mean = pad_dim(deletion_mean, 0, pad_len)

    return {
        "msa": msa,
        "msa_paired": paired,
        "deletion_value": deletion,
        "has_deletion": has_deletion,
        "deletion_mean": deletion_mean,
        "profile": profile,
        "msa_mask": msa_mask,
    }


# Dummy functions for symmetry features, as we don't have the full boltz.data.feature.symmetry module
def get_amino_acids_symmetries(cropped):
    return {}

def get_chain_symmetries(cropped):
    return {}

def get_ligand_symmetries(cropped, symmetries):
    return {}


def process_symmetry_features(
    cropped: Tokenized, symmetries: dict
) -> dict[str, Tensor]:
    """Get the symmetry features.

    Parameters
    ----------
    data : Tokenized
        The tokenized data.

    Returns
    -------
    dict[str, Tensor]
        The symmetry features.

    """
    features = get_chain_symmetries(cropped)
    features.update(get_amino_acids_symmetries(cropped))
    features.update(get_ligand_symmetries(cropped, symmetries))

    return features


def process_residue_constraint_features(
    data: Tokenized,
) -> dict[str, Tensor]:
    residue_constraints = data.residue_constraints
    if residue_constraints is not None:
        rdkit_bounds_constraints = residue_constraints.rdkit_bounds_constraints
        chiral_atom_constraints = residue_constraints.chiral_atom_constraints
        stereo_bond_constraints = residue_constraints.stereo_bond_constraints
        planar_bond_constraints = residue_constraints.planar_bond_constraints
        planar_ring_5_constraints = residue_constraints.planar_ring_5_constraints
        planar_ring_6_constraints = residue_constraints.planar_ring_6_constraints

        rdkit_bounds_index = torch.tensor(
            rdkit_bounds_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
        rdkit_bounds_bond_mask = torch.tensor(
            rdkit_bounds_constraints["is_bond"].copy(), dtype=torch.bool
        )
        rdkit_bounds_angle_mask = torch.tensor(
            rdkit_bounds_constraints["is_angle"].copy(), dtype=torch.bool
        )
        rdkit_upper_bounds = torch.tensor(
            rdkit_bounds_constraints["upper_bound"].copy(), dtype=torch.float
        )
        rdkit_lower_bounds = torch.tensor(
            rdkit_bounds_constraints["lower_bound"].copy(), dtype=torch.float
        )

        chiral_atom_index = torch.tensor(
            chiral_atom_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
        chiral_reference_mask = torch.tensor(
            chiral_atom_constraints["is_reference"].copy(), dtype=torch.bool
        )
        chiral_atom_orientations = torch.tensor(
            chiral_atom_constraints["is_r"].copy(), dtype=torch.bool
        )

        stereo_bond_index = torch.tensor(
            stereo_bond_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
        stereo_reference_mask = torch.tensor(
            stereo_bond_constraints["is_reference"].copy(), dtype=torch.bool
        )
        stereo_bond_orientations = torch.tensor(
            stereo_bond_constraints["is_e"].copy(), dtype=torch.bool
        )

        planar_bond_index = torch.tensor(
            planar_bond_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
        planar_ring_5_index = torch.tensor(
            planar_ring_5_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
        planar_ring_6_index = torch.tensor(
            planar_ring_6_constraints["atom_idxs"].copy(), dtype=torch.long
        ).T
    else:
        rdkit_bounds_index = torch.empty((2, 0), dtype=torch.long)
        rdkit_bounds_bond_mask = torch.empty((0,), dtype=torch.bool)
        rdkit_bounds_angle_mask = torch.empty((0,), dtype=torch.bool)
        rdkit_upper_bounds = torch.empty((0,), dtype=torch.float)
        rdkit_lower_bounds = torch.empty((0,), dtype=torch.float)
        chiral_atom_index = torch.empty(
            (
                4,
                0,
            ),
            dtype=torch.long,
        )
        chiral_reference_mask = torch.empty((0,), dtype=torch.bool)
        chiral_atom_orientations = torch.empty((0,), dtype=torch.bool)
        stereo_bond_index = torch.empty((4, 0), dtype=torch.long)
        stereo_reference_mask = torch.empty((0,), dtype=torch.bool)
        stereo_bond_orientations = torch.empty((0,), dtype=torch.bool)
        planar_bond_index = torch.empty((6, 0), dtype=torch.long)
        planar_ring_5_index = torch.empty((5, 0), dtype=torch.long)
        planar_ring_6_index = torch.empty((6, 0), dtype=torch.long)

    return {
        "rdkit_bounds_index": rdkit_bounds_index,
        "rdkit_bounds_bond_mask": rdkit_bounds_bond_mask,
        "rdkit_bounds_angle_mask": rdkit_bounds_angle_mask,
        "rdkit_upper_bounds": rdkit_upper_bounds,
        "rdkit_lower_bounds": rdkit_lower_bounds,
        "chiral_atom_index": chiral_atom_index,
        "chiral_reference_mask": chiral_reference_mask,
        "chiral_atom_orientations": chiral_atom_orientations,
        "stereo_bond_index": stereo_bond_index,
        "stereo_reference_mask": stereo_reference_mask,
        "stereo_bond_orientations": stereo_bond_orientations,
        "planar_bond_index": planar_bond_index,
        "planar_ring_5_index": planar_ring_5_index,
        "planar_ring_6_index": planar_ring_6_index,
    }


def process_chain_feature_constraints(
    data: Tokenized,
) -> dict[str, Tensor]:
    structure = data.structure
    if structure.connections.shape[0] > 0:
        connected_chain_index, connected_atom_index = [], []
        for connection in structure.connections:
            connected_chain_index.append([connection["chain_1"], connection["chain_2"]])
            connected_atom_index.append([connection["atom_1"], connection["atom_2"]])
        connected_chain_index = torch.tensor(connected_chain_index, dtype=torch.long).T
        connected_atom_index = torch.tensor(connected_atom_index, dtype=torch.long).T
    else:
        connected_chain_index = torch.empty((2, 0), dtype=torch.long)
        connected_atom_index = torch.empty((2, 0), dtype=torch.long)

    symmetric_chain_index = []
    for i, chain_i in enumerate(structure.chains):
        for j, chain_j in enumerate(structure.chains):
            if j <= i:
                continue
            if chain_i["entity_id"] == chain_j["entity_id"]:
                symmetric_chain_index.append([i, j])
    if len(symmetric_chain_index) > 0:
        symmetric_chain_index = torch.tensor(symmetric_chain_index, dtype=torch.long).T
    else:
        symmetric_chain_index = torch.empty((2, 0), dtype=torch.long)
    return {
        "connected_chain_index": connected_chain_index,
        "connected_atom_index": connected_atom_index,
        "symmetric_chain_index": symmetric_chain_index,
    }


class BoltzFeaturizer:
    """Boltz featurizer."""

    def process(
        self,
        data: Tokenized,
        training: bool,
        max_seqs: int = 4096,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        max_tokens: Optional[int] = None,
        max_atoms: Optional[int] = None,
        pad_to_max_seqs: bool = False,
        compute_symmetries: bool = False,
        symmetries: Optional[dict] = None,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        only_ligand_binder_pocket: Optional[bool] = False,
        inference_binder: Optional[int] = None,
        inference_pocket: Optional[list[tuple[int, int]]] = None,
        compute_constraint_features: bool = False,
    ) -> dict[str, Tensor]:
        """Compute features.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        training : bool
            Whether the model is in training mode.
        max_tokens : int, optional
            The maximum number of tokens.
        max_atoms : int, optional
            The maximum number of atoms
        max_seqs : int, optional
            The maximum number of sequences.

        Returns
        -------
        dict[str, Tensor]
            The features for model training.

        """
        # Compute random number of sequences
        if training and max_seqs is not None:
            max_seqs_batch = np.random.randint(1, max_seqs + 1)  # noqa: NPY002
        else:
            max_seqs_batch = max_seqs

        # Compute token features
        token_features = process_token_features(
            data,
            max_tokens,
            binder_pocket_conditioned_prop,
            binder_pocket_cutoff,
            binder_pocket_sampling_geometric_p,
            only_ligand_binder_pocket,
            inference_binder=inference_binder,
            inference_pocket=inference_pocket,
        )

        # Compute atom features
        atom_features = process_atom_features(
            data,
            atoms_per_window_queries,
            min_dist,
            max_dist,
            num_bins,
            max_atoms,
            max_tokens,
        )

        # Compute MSA features
        msa_features = process_msa_features(
            data,
            max_seqs_batch,
            max_seqs,
            max_tokens,
            pad_to_max_seqs,
        )

        # Compute symmetry features
        symmetry_features = {}
        if compute_symmetries:
            symmetry_features = process_symmetry_features(data, symmetries)

        # Compute constraint features
        residue_constraint_features = {}
        chain_constraint_features = {}
        if compute_constraint_features:
            residue_constraint_features = process_residue_constraint_features(data)
            chain_constraint_features = process_chain_feature_constraints(data)

        return {
            **token_features,
            **atom_features,
            **msa_features,
            **symmetry_features,
            **residue_constraint_features,
            **chain_constraint_features,
        }


# Replicated from boltz/src/boltz/data/module/training.py

@dataclass
class Sample:
    """Sample datatype."""

    record: Record
    chain_id: Optional[int] = None
    interface_id: Optional[int] = None


class Sampler(ABC):
    """Abstract base class for sampler."""

    @abstractmethod
    def sample(self, records: list[Record], random: np.random.RandomState):
        """Sample records.

        Parameters
        ----------
        records : list[Record]
            The records to sample from.
        random : np.random.RandomState
            The random state for reproducibility.

        Returns
        -------
        Iterator[Sample]
            The sampled records.

        """
        raise NotImplementedError


class RandomSampler(Sampler):
    """Randomly sample records."""

    def sample(self, records: list[Record], random: np.random.RandomState):
        """Sample records.

        Parameters
        ----------
        records : list[Record]
            The records to sample from.
        random : np.random.RandomState
            The random state for reproducibility.

        Returns
        -------
        Iterator[Sample]
            The sampled records.

        """
        while True:
            record = random.choice(records)
            yield Sample(record=record)


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    target_dir: str
    msa_dir: str
    prob: float
    sampler: Sampler
    cropper: Cropper
    filters: Optional[list] = None
    split: Optional[str] = None
    manifest_path: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration."""

    datasets: list[DatasetConfig]
    featurizer: BoltzFeaturizer
    tokenizer: BoltzTokenizer # Changed from Tokenizer to BoltzTokenizer
    max_atoms: int
    max_tokens: int
    max_seqs: int
    samples_per_epoch: int
    batch_size: int
    num_workers: int
    random_seed: int
    pin_memory: bool
    symmetries: str
    atoms_per_window_queries: int
    min_dist: float
    max_dist: float
    num_bins: int
    filters: Optional[list] = None # Changed from DynamicFilter to list
    overfit: Optional[int] = None
    pad_to_max_tokens: bool = False
    pad_to_max_atoms: bool = False
    pad_to_max_seqs: bool = False
    crop_validation: bool = False
    return_train_symmetries: bool = False
    return_val_symmetries: bool = True
    train_binder_pocket_conditioned_prop: float = 0.0
    val_binder_pocket_conditioned_prop: float = 0.0
    binder_pocket_cutoff: float = 6.0
    binder_pocket_sampling_geometric_p: float = 0.0
    val_batch_size: int = 1
    compute_constraint_features: bool = False


@dataclass
class Dataset:
    """Data holder."""

    target_dir: Path
    msa_dir: Path
    manifest: Manifest
    prob: float
    sampler: Sampler
    cropper: Cropper
    tokenizer: BoltzTokenizer # Changed from Tokenizer to BoltzTokenizer
    featurizer: BoltzFeaturizer


def load_input(record: Record, target_dir: Path, msa_dir: Path) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    # Assuming structure files are also in msa_dir for simplicity, or not used for now
    # For now, we'll assume structure is not needed for MSA-only processing
    # If structure is needed, it would be record.id.npz
    # For now, we'll create a dummy structure
    # For now, we'll create a minimal dummy structure that allows tokenization
    # This is a placeholder and might need to be more sophisticated if structure-based features are critical.

    # Minimal Atom
    dummy_atoms_data = np.array([
        ([ord('C'), ord('A'), ord(' '), ord(' ')], 6, 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], True, 0)
    ], dtype=Atom)

    # Minimal Residue
    dummy_residues_data = np.array([
        (b'ALA', 0, 0, 0, 1, 0, 0, True, True) # Added 0 for atom_disto
    ], dtype=Residue)

    # Minimal Chain
    dummy_chains_data = np.array([
        (b'A', 0, 0, 0, 0, 0, 1, 0, 1, 0) # mol_type=0 (PROTEIN), atom_idx=0, atom_num=1, res_idx=0, res_num=1
    ], dtype=Chain)

    dummy_connections = np.array([], dtype=Connection)
    dummy_interfaces = np.array([], dtype=Interface)
    dummy_mask = np.array([True], dtype=bool) # Mask for the single chain

    structure = Structure(
        atoms=dummy_atoms_data,
        bonds=np.array([], dtype=Bond),
        residues=dummy_residues_data,
        chains=dummy_chains_data,
        connections=dummy_connections,
        interfaces=dummy_interfaces,
        mask=dummy_mask,
    )

    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1 and msa_id != "":
            msa = np.load(msa_dir / f"{msa_id}.npz", allow_pickle=True)
            msas[chain.chain_id] = MSA(**msa)

    return Input(structure, msas, record=record)


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : list[dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, Tensor]
        The collated data.

    """
    # Filter out None values (failed samples)
    data = [d for d in data if d is not None]
    if not data: # If all samples failed, return an empty dict or raise an error
        return {}

    # Get the keys from the first valid sample
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                # Pad to max shape in the batch
                max_shape = np.array([v.shape for v in values]).max(axis=0)
                padded_values = []
                for v in values:
                    padding_needed = max_shape - np.array(v.shape)
                    # Construct padding for each dimension
                    paddings = []
                    for dim_pad in reversed(padding_needed):
                        paddings.extend([0, dim_pad])
                    padded_v = torch.nn.functional.pad(v, paddings, "constant", 0)
                    padded_values.append(padded_v)
                values = torch.stack(padded_values, dim=0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class TrainingDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        samples_per_epoch: int,
        symmetries: dict,
        max_atoms: int,
        max_tokens: int,
        max_seqs: int,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
        binder_pocket_sampling_geometric_p: Optional[float] = 0.0,
        return_symmetries: Optional[bool] = False,
        compute_constraint_features: bool = False,
    ) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.datasets = datasets
        self.probs = [d.prob for d in datasets]
        self.samples_per_epoch = samples_per_epoch
        self.symmetries = symmetries
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.max_atoms = max_atoms
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff
        self.binder_pocket_sampling_geometric_p = binder_pocket_sampling_geometric_p
        self.return_symmetries = return_symmetries
        self.compute_constraint_features = compute_constraint_features
        self.samples = []
        for dataset in datasets:
            records = dataset.manifest.records
            if overfit is not None:
                records = records[:overfit]
            iterator = dataset.sampler.sample(records, np.random)
            self.samples.append(iterator)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        # Pick a random dataset
        dataset_idx = np.random.choice(
            len(self.datasets),
            p=self.probs,
        )
        dataset = self.datasets[dataset_idx]

        # Get a sample from the dataset
        sample: Sample = next(self.samples[dataset_idx])

        # Get the structure
        try:
            input_data = load_input(sample.record, dataset.target_dir, dataset.msa_dir)
        except Exception as e:
            print(
                f"Failed to load input for {sample.record.id} with error {e}. Skipping."
            )
            return None

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {sample.record.id} with error {e}. Skipping.")
            return None

        # Compute crop
        try:
            if self.max_tokens is not None:
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_tokens=self.max_tokens,
                    random=np.random,
                    max_atoms=self.max_atoms,
                    chain_id=sample.chain_id,
                    interface_id=sample.interface_id,
                )
        except Exception as e:
            print(f"Cropper failed on {sample.record.id} with error {e}. Skipping.")
            return None

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        # Compute features
        try:
            features = dataset.featurizer.process(
                tokenized,
                training=True,
                max_atoms=self.max_atoms if self.pad_to_max_atoms else None,
                max_tokens=self.max_tokens if self.pad_to_max_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                binder_pocket_cutoff=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=self.binder_pocket_sampling_geometric_p,
                compute_constraint_features=self.compute_constraint_features,
            )
        except Exception as e:
            print(f"Featurizer failed on {sample.record.id} with error {e}. Skipping.")
            return None

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return self.samples_per_epoch


class ValidationDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        seed: int,
        symmetries: dict,
        max_atoms: Optional[int] = None,
        max_tokens: Optional[int] = None,
        max_seqs: Optional[int] = None,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        pad_to_max_seqs: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        overfit: Optional[int] = None,
        crop_validation: bool = False,
        return_symmetries: Optional[bool] = False,
        binder_pocket_conditioned_prop: Optional[float] = 0.0,
        binder_pocket_cutoff: Optional[float] = 6.0,
        compute_constraint_features: bool = False,
    ) -> None:
        """Initialize the validation dataset."""
        super().__init__()
        self.datasets = datasets
        self.max_atoms = max_atoms
        self.max_tokens = max_tokens
        self.max_seqs = max_seqs
        self.seed = seed
        self.symmetries = symmetries
        self.random = np.random if overfit else np.random.RandomState(self.seed)
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.pad_to_max_seqs = pad_to_max_seqs
        self.overfit = overfit
        self.crop_validation = crop_validation
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.return_symmetries = return_symmetries
        self.binder_pocket_conditioned_prop = binder_pocket_conditioned_prop
        self.binder_pocket_cutoff = binder_pocket_cutoff
        self.compute_constraint_features = compute_constraint_features

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        # Pick dataset based on idx
        for dataset in self.datasets:
            size = len(dataset.manifest.records)
            if self.overfit is not None:
                size = min(size, self.overfit)
            if idx < size:
                break
            idx -= size

        # Get a sample from the dataset
        record = dataset.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(record, dataset.target_dir, dataset.msa_dir)
        except Exception as e:
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = dataset.tokenizer.tokenize(input_data)
        except Exception as e:
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Compute crop
        try:
            if self.crop_validation and (self.max_tokens is not None):
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_tokens=self.max_tokens,
                    random=self.random,
                    max_atoms=self.max_atoms,
                )
        except Exception as e:
            print(f"Cropper failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        # Compute features
        try:
            pad_atoms = self.crop_validation and self.pad_to_max_atoms
            pad_tokens = self.crop_validation and self.pad_to_max_tokens

            features = dataset.featurizer.process(
                tokenized,
                training=False,
                max_atoms=self.max_atoms if pad_atoms else None,
                max_tokens=self.max_tokens if pad_tokens else None,
                max_seqs=self.max_seqs,
                pad_to_max_seqs=self.pad_to_max_seqs,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                binder_pocket_conditioned_prop=self.binder_pocket_conditioned_prop,
                binder_pocket_cutoff=self.binder_pocket_cutoff,
                binder_pocket_sampling_geometric_p=1.0,  # this will only sample a single pocket token
                only_ligand_binder_pocket=True,
                compute_constraint_features=self.compute_constraint_features,
            )
        except Exception as e:
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(0)

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        if self.overfit is not None:
            length = sum(len(d.manifest.records[: self.overfit]) for d in self.datasets)
        else:
            length = sum(len(d.manifest.records) for d in self.datasets)

        return length


# Removed pl.LightningDataModule dependency
# class BoltzTrainingDataModule(pl.LightningDataModule):
#     """DataModule for boltz."""

#     def __init__(self, cfg: DataConfig) -> None:
#         """Initialize the DataModule.

#         Parameters
#         ----------
#         config : DataConfig
#             The data configuration.

#         """
#         super().__init__()
#         self.cfg = cfg

#         assert self.cfg.val_batch_size == 1, "Validation only works with batch size=1."

#         # Load symmetries
#         symmetries = get_symmetries(cfg.symmetries)

#         # Load datasets
#         train: list[Dataset] = []
#         val: list[Dataset] = []

#         for data_config in cfg.datasets:
#             # Set target_dir
#             target_dir = Path(data_config.target_dir)
#             msa_dir = Path(data_config.msa_dir)

#             # Load manifest
#             if data_config.manifest_path is not None:
#                 path = Path(data_config.manifest_path)
#             else:
#                 path = target_dir / "manifest.json"
#             manifest: Manifest = Manifest.load(path)

#             # Split records if given
#             if data_config.split is not None:
#                 with Path(data_config.split).open("r") as f:
#                     split = {x.lower() for x in f.read().splitlines()}

#                 train_records = []
#                 val_records = []
#                 for record in manifest.records:
#                     if record.id.lower() in split:
#                         val_records.append(record)
#                     else:
#                         train_records.append(record)
#             else:
#                 train_records = manifest.records
#                 val_records = []

#             # Filter training records
#             train_records = [
#                 record
#                 for record in train_records
#                 if all(f.filter(record) for f in cfg.filters)
#             ]
#             # Filter training records
#             if data_config.filters is not None:
#                 train_records = [
#                     record
#                     for record in train_records
#                     if all(f.filter(record) for f in data_config.filters)
#                 ]

#             # Create train dataset
#             train_manifest = Manifest(train_records)
#             train.append(
#                 Dataset(
#                     target_dir,
#                     msa_dir,
#                     train_manifest,
#                     data_config.prob,
#                     data_config.sampler,
#                     data_config.cropper,
#                     cfg.tokenizer,
#                     cfg.featurizer,
#                 )
#             )

#             # Create validation dataset
#             if val_records:
#                 val_manifest = Manifest(val_records)
#                 val.append(
#                     Dataset(
#                         target_dir,
#                         msa_dir,
#                         val_manifest,
#                         data_config.prob,
#                         data_config.sampler,
#                         data_config.cropper,
#                         cfg.tokenizer,
#                         cfg.featurizer,
#                     )
#                 )

#         # Print dataset sizes
#         for dataset in train:
#             dataset: Dataset
#             print(f"Training dataset size: {len(dataset.manifest.records)}")

#         for dataset in val:
#             dataset: Dataset
#             print(f"Validation dataset size: {len(dataset.manifest.records)}")

#         # Create wrapper datasets
#         self._train_set = TrainingDataset(
#             datasets=train,
#             samples_per_epoch=cfg.samples_per_epoch,
#             max_atoms=cfg.max_atoms,
#             max_tokens=cfg.max_tokens,
#             max_seqs=cfg.max_seqs,
#             pad_to_max_atoms=cfg.pad_to_max_atoms,
#             pad_to_max_tokens=cfg.pad_to_max_tokens,
#             pad_to_max_seqs=cfg.pad_to_max_seqs,
#             symmetries=symmetries,
#             atoms_per_window_queries=cfg.atoms_per_window_queries,
#             min_dist=cfg.min_dist,
#             max_dist=cfg.max_dist,
#             num_bins=cfg.num_bins,
#             overfit=cfg.overfit,
#             binder_pocket_conditioned_prop=cfg.train_binder_pocket_conditioned_prop,
#             binder_pocket_cutoff=cfg.binder_pocket_cutoff,
#             binder_pocket_sampling_geometric_p=cfg.binder_pocket_sampling_geometric_p,
#             return_symmetries=cfg.return_train_symmetries,
#             compute_constraint_features=cfg.compute_constraint_features,
#         )
#         self._val_set = ValidationDataset(
#             datasets=train if cfg.overfit is not None else val,
#             seed=cfg.random_seed,
#             max_atoms=cfg.max_atoms,
#             max_tokens=cfg.max_tokens,
#             max_seqs=cfg.max_seqs,
#             pad_to_max_atoms=cfg.pad_to_max_atoms,
#             pad_to_max_tokens=cfg.pad_to_max_tokens,
#             pad_to_max_seqs=cfg.pad_to_max_seqs,
#             symmetries=symmetries,
#             atoms_per_window_queries=cfg.atoms_per_window_queries,
#             min_dist=cfg.min_dist,
#             max_dist=cfg.max_dist,
#             num_bins=cfg.num_bins,
#             overfit=cfg.overfit,
#             crop_validation=cfg.crop_validation,
#             return_symmetries=cfg.return_val_symmetries,
#             binder_pocket_conditioned_prop=cfg.val_binder_pocket_conditioned_prop,
#             binder_pocket_cutoff=cfg.binder_pocket_cutoff,
#             compute_constraint_features=cfg.compute_constraint_features,
#         )

#     def setup(self, stage: Optional[str] = None) -> None:
#         """Run the setup for the DataModule.

#         Parameters
#         ----------
#         stage : str, optional
#             The stage, one of 'fit', 'validate', 'test'.

#         """
#         return

#     def train_dataloader(self) -> DataLoader:
#         """Get the training dataloader.

#         Returns
#         -------
#         DataLoader
#             The training dataloader.

#         """
#         return DataLoader(
#             self._train_set,
#             batch_size=self.cfg.batch_size,
#             num_workers=self.cfg.num_workers,
#             pin_memory=self.cfg.pin_memory,
#             shuffle=False,
#             collate_fn=collate,
#         )

#     def val_dataloader(self) -> DataLoader:
#         """Get the validation dataloader.

#         Returns
#         -------
#         DataLoader
#             The validation dataloader.

#         """
#         return DataLoader(
#             self._val_set,
#             batch_size=self.cfg.val_batch_size,
#             num_workers=self.cfg.num_workers,
#             pin_memory=self.cfg.pin_memory,
#             shuffle=False,
#             collate_fn=collate,
#         )