"""Molecular graph featurization: SMILES -> torch_geometric Data.

Feature spec (fixed, shared with the model):
  ATOM_F  = 35  (element 11 | degree 6 | charge 5 | numH 5 | hybridization 6 | aromatic 1 | in_ring 1)
  BOND_F  = 6   (bond type 4 | conjugated 1 | in_ring 1)
"""

from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

ELEMENTS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

ATOM_F = len(ELEMENTS) + len(DEGREES) + len(CHARGES) + len(NUM_HS) + len(HYBRIDS) + 1 + 2
BOND_F = len(BOND_TYPES) + 1 + 2


def _onehot(value, choices) -> list[float]:
    return [float(value == c) for c in choices]


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def strip_salts(smiles: str) -> str | None:
    """Strip salts/solvents keeping the largest organic fragment; None if invalid."""
    from rdkit.Chem.SaltRemover import SaltRemover

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:
        pass
    try:
        remover = SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)
    except Exception:
        pass
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumAtoms())
    if mol.GetNumAtoms() == 0 or not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        return None
    return Chem.MolToSmiles(mol)


def atom_features(atom: Chem.Atom) -> np.ndarray:
    feats = []
    feats += _onehot(atom.GetSymbol(), ELEMENTS)
    feats += _onehot(atom.GetTotalDegree(), DEGREES)
    feats += _onehot(atom.GetFormalCharge(), CHARGES)
    feats += _onehot(atom.GetTotalNumHs(), NUM_HS)
    hyb = atom.GetHybridization()
    feats += _onehot(hyb, HYBRIDS)
    feats.append(float(hyb not in HYBRIDS))  # "other" slot
    feats.append(float(atom.GetIsAromatic()))
    feats.append(float(atom.IsInRing()))
    assert len(feats) == ATOM_F
    return np.asarray(feats, dtype=np.float32)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    bt = bond.GetBondType()
    feats = _onehot(bt, BOND_TYPES)
    feats.append(float(bt not in BOND_TYPES))
    feats.append(float(bond.GetIsConjugated()))
    feats.append(float(bond.IsInRing()))
    assert len(feats) == BOND_F
    return np.asarray(feats, dtype=np.float32)


def smiles_to_graph(smiles: str) -> Data | None:
    """Canonical SMILES -> PyG graph. Returns None for invalid/empty molecules."""
    canon = strip_salts(smiles)
    if canon is None:
        return None
    mol = Chem.MolFromSmiles(canon)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = np.stack([atom_features(a) for a in mol.GetAtoms()])
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        if edge_index
        else torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(np.stack(edge_attr), dtype=torch.float32)
        if edge_attr
        else torch.zeros((0, BOND_F), dtype=torch.float32),
    )
    data.smiles = canon
    data.num_nodes = mol.GetNumAtoms()
    return data


def graphs_to_batch(graphs: list[Data]):
    from torch_geometric.loader import DataLoader

    return next(iter(DataLoader(graphs, batch_size=len(graphs), shuffle=False)))
