import pandas as pd

from ml.data.splits import scaffold_split, add_scaffold_column


def _toy_df():
    return pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1O", "CCC", "CCCC", "c1ccc(O)cc1", "CCCCCC",
                   "CCO", "c1ccccc1O", "CCC", "CCCC", "c1ccc(O)cc1", "CCCCCC"],
    })


def test_no_scaffold_leakage_with_duplicates():
    tr, va, te = scaffold_split(_toy_df())
    kt = set(tr["scaffold"])
    kv = set(va["scaffold"])
    ke = set(te["scaffold"])
    assert not (kt & kv or kt & ke or kv & ke)


def test_split_preserves_all_rows():
    tr, va, te = scaffold_split(_toy_df())
    assert len(tr) + len(va) + len(te) == len(_toy_df())


def test_same_molecule_stays_in_one_split():
    df = add_scaffold_column(pd.DataFrame({"smiles": ["CCO"] * 10}))
    tr, va, te = scaffold_split(df)
    locations = [("t" if i < len(tr) else "v" if i < len(tr) + len(va) else "e")
                 for i in range(len(df))]
    cco_locs = set(loc for loc, s in zip(locations, list(tr["smiles"]) + list(va["smiles"]) + list(te["smiles"])) if s == "CCO")
    assert len(cco_locs) <= 1  # all CCO rows land in exactly one split
