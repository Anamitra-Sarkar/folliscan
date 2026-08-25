import pandas as pd

from ml.data.merge import build_master_dataset
from ml.data.task_registry import TASK_INDEX


def _master():
    hair = pd.DataFrame({"smiles": ["CCO"], "label": [1]})
    tox = pd.DataFrame({"smiles": ["CCO"], "label": [0]})
    return build_master_dataset({
        "SULT1A1_active": hair,
        "NR-AR": tox,
        "SRD5A1_inhibitor": pd.DataFrame(
            {"smiles": ["CCC", "CCC", "CCC"], "label": [1, 0, 0]}),
    })


def test_tasks_do_not_bleed_votes():
    r = _master().iloc[0]
    assert r["labels"][TASK_INDEX["SULT1A1_active"]] == 1.0   # positive vote kept here
    assert r["labels"][TASK_INDEX["NR-AR"]] == 0.0            # negative here
    assert all(r["mask"][i] == 0 for i in range(21)
               if i not in (TASK_INDEX["SULT1A1_active"], TASK_INDEX["NR-AR"]))


def test_majority_vote_and_mask():
    df = _master()
    ccc = df[df["smiles"] == "CCC"].iloc[0]
    # SRD5A1: 2 negatives vs 1 positive -> negative label with mask=1
    assert ccc["mask"][TASK_INDEX["SRD5A1_inhibitor"]] == 1
    assert ccc["labels"][TASK_INDEX["SRD5A1_inhibitor"]] == 0.0


def test_unlabelled_sentinel():
    r = _master().iloc[0]
    assert r["labels"][TASK_INDEX["skin_sensitizer"]] == -1.0
    assert r["mask"][TASK_INDEX["skin_sensitizer"]] == 0
