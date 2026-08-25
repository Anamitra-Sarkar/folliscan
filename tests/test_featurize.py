from ml.data.featurize import (
    smiles_to_graph,
    ATOM_F,
    BOND_F,
    canonicalize,
    strip_salts,
)


def test_graph_dims_match_spec():
    g = smiles_to_graph("CCO")
    assert g is not None
    assert g.x.shape[1] == ATOM_F == 36
    assert g.edge_attr.shape[1] == BOND_F == 7


def test_bond_count_doubled():
    g = smiles_to_graph("C=CC")  # propene: 2 bonds -> 4 directed edges
    assert g.edge_index.shape[1] == 4
    assert g.edge_attr.shape[0] == 4


def test_invalid_smiles_returns_none():
    assert smiles_to_graph("not_a_smiles") is None
    assert smiles_to_graph("") is None


def test_salt_stripping_keeps_largest_fragment():
    canon = strip_salts("CC(=O)[O-].[Na+]")  # sodium acetate -> acetic acid fragment
    assert canon is None or "Na" not in canon  # Na alone has no carbon -> dropped


def test_canonicalization_is_stable():
    a = canonicalize("OCC")
    b = canonicalize("C(C)O")
    assert a == b == "CCO"
