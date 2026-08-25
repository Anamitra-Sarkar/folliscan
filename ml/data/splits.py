"""Leakage-safe dataset splitting grouped by Bemis-Murcko scaffold.

Key invariant: one scaffold string <-> exactly one fold. Acyclic molecules
(empty Murcko scaffold) become singleton groups keyed by canonical SMILES,
so they can never bridge splits.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def murcko_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return None


def scaffold_key(smiles: str) -> str:
    """Stable grouping key: Murcko scaffold, or singleton key for acyclic/invalid."""
    sc = murcko_scaffold(smiles)
    if not sc:
        return f"__singleton__{smiles}"
    return sc


def _key_to_fold(key: str, n_folds: int, seed: int) -> int:
    h = hashlib.md5(f"{seed}::{key}".encode()).hexdigest()
    return int(h, 16) % n_folds


def scaffold_split(df: pd.DataFrame, smiles_col: str = "smiles", seed: int = 42):
    """80/10/10 train/val/test; every molecule sharing a scaffold lands in one fold."""
    keys = [scaffold_key(s) for s in df[smiles_col]]
    folds = np.array([_key_to_fold(k, 10, seed) for k in keys], dtype=int)

    train = df[folds < 8].reset_index(drop=True)
    val = df[folds == 8].reset_index(drop=True)
    test = df[folds == 9].reset_index(drop=True)

    # Sanity: no key may appear in more than one split.
    kt = {keys[i] for i in train.index}
    kv = {keys[i] for i in val.index}
    ke = {keys[i] for i in test.index}
    assert not (kt & kv or kt & ke or kv & ke), "scaffold leakage"

    out_dfs = []
    for d in (train, val, test):
        d = d.copy()
        d["scaffold"] = [scaffold_key(s) for s in d[smiles_col]]
        out_dfs.append(d)
    return tuple(out_dfs)


def add_scaffold_column(df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    df = df.copy()
    df["scaffold"] = [scaffold_key(s) for s in df[smiles_col]]
    return df
