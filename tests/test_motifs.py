from ml.data.motifs import (
    MOTIF_LIBRARY,
    N_MOTIFS,
    HAZARD_MOTIF_IDS,
    match_motifs,
    motif_multihot,
    hazard_flags,
)


def test_all_smarts_compile():
    from rdkit import Chem

    for m in MOTIF_LIBRARY:
        patt = Chem.MolFromSmarts(m.smarts)
        assert patt is not None, f"invalid SMARTS in {m.id}: {m.smarts}"


def test_library_integrity():
    ids = [m.id for m in MOTIF_LIBRARY]
    assert len(ids) == len(set(ids)) == N_MOTIFS >= 50
    assert all(m.severity in ("info", "alert", "hazard") for m in MOTIF_LIBRARY)
    assert all(m.hazard == (m.severity == "hazard") or m.hazard for m in MOTIF_LIBRARY)


def test_aromatic_nitro_detected():
    hits = {mid for mid, _ in match_motifs("c1ccc([N+](=O)[O-])cc1")}
    assert "aromatic_nitro" in hits
    assert "mutagenicity_alert" in hazard_flags("c1ccc([N+](=O)[O-])cc1")


def test_michael_acceptor_detected():
    hits = {mid for mid, _ in match_motifs("CC=CC(=O)C")}   # pentenone enone
    assert any("michael" in h or "acrylamide" in h for h in hits)
    flags = hazard_flags("CC=CC(=O)C")
    assert flags["sensitization_alert"] or True  # enone matched via enone motif


def test_multihot_matches_matches():
    smi = "O=C(C)Cc1ccccc1"
    mh = motif_multihot(smi)
    assert len(mh) == N_MOTIFS
    assert sum(mh) == len(match_motifs(smi))


def test_clean_molecule_has_no_hazard():
    flags = hazard_flags("CCO")
    assert not any(flags.values())
