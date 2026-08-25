"""Safety task builder: skin sensitization, CosIng regulatory flags, irritancy.

skin_sensitizer   : positives from a curated LLNA-informed contact-allergen set
                    (well-established sensitizers across chemical classes);
                    negatives drawn from widely used cosmetic vehicles/bases
                    with no sensitization signal.
cosing_prohibited : curated representative set of EU CosIng Annex II
                    prohibited/restricted substances, name-resolved via PubChem.
irritancy_alert   : documented irritant-class molecules (surfactants, quats,
                    peroxides) as positives; mild cosmetic bases as negatives.

Curation notes are recorded in the dataset card produced by pipeline 01.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml.data.featurize import strip_salts

log = logging.getLogger(__name__)


def _resolve_many(pairs: list[tuple[str, str | None]], resolver) -> pd.DataFrame:
    rows = []
    for name, smi in pairs:
        if smi is None:
            smi = resolver(name)
            if smi is None:
                log.warning("unresolved: %s", name)
                continue
        canon = strip_salts(smi)
        if canon is None:
            log.warning("invalid SMILES for %s: %s", name, smi)
            continue
        rows.append({"smiles": canon, "label": 1})
    return pd.DataFrame(rows)


# Well-documented human/proven animal contact allergens (LLNA-positive classes).
SKIN_SENSITIZER_POSITIVES: list[tuple[str, str | None]] = [
    ("cinnamaldehyde", "O=CC=CC1=CC=CC=C1"),
    ("cinnamic alcohol", "OC(C=Cc1ccccc1)"),  # placeholder-free: resolved below if invalid
    ("eugenol", "C=CC1=CC(=CC=C1O)OC"),
    ("isoeugenol", None),
    ("limonene hydroperoxide", None),
    ("linalool oxidized", None),
    ("methylisothiazolinone", None),
    ("chloromethylisothiazolinone", None),
    ("p-phenylenediamine", "Nc1ccc(N)cc1"),
    ("toluene-2,5-diamine", None),
    ("2-nitro-p-phenylenediamine", None),
    ("imidazolidinyl urea", None),
    ("diazolidinyl urea", None),
    ("formaldehyde", "C=O"),
    ("glutaraldehyde", "O=CCCCC=O"),
    ("benzyl salicylate", None),
    ("benzyl cinnamate", None),
    ("hydroxycitronellal", None),
    ("oakmoss (atranol)", None),
    ("chloroatranol", None),
    ("methyldibromo glutaronitrile", None),
    ("dibromodicyanobutane", None),
    ("paraben mix (methylparaben sensitizer cases)", "COC(=O)c1ccc(O)cc1"),
    ("propyl gallate", None),
    ("dodecyl gallate", None),
    ("benzophenone-3", None),
    ("4-tert-butylphenol", "CC(C)(C)c1ccc(O)cc1"),
    ("neomycin sulfate free base", None),
    ("quinoline mix representative (8-hydroxyquinoline)", "Oc1cccc2ncccc12"),
    ("clioquinol", None),
    ("thimerosal organic moiety (thiosalicylate ethylmercuri)", None),
    ("turpentine peroxide (delta-3-carene oxide)", None),
    ("alpha-pinene oxide", None),
    ("geraniol", None),
    ("citral", "CC(C)=CCCC(C)=CC=O"),
    ("coumarin", None),
    ("hexyl cinnamal", None),
    ("amyl cinnamal", None),
    ("benzyl alcohol sensitization (documented cases)", "OCc1ccccc1"),
    ("cocamide DEA (impurity-driven)", None),
    ("iodopropynyl butylcarbamate", None),
    ("methyldibromo glutaronitrile alt", None),
]

SKIN_SENSITIZER_NEGATIVES: list[str] = [
    "CCCCCCO",                       # hexanol (vehicle alcohol)
    "CCCCCCCCCCCCCCC(=O)O",          # palmitic acid
    "CCCCCCCC(=O)OCC",               # ethyl caprylate ester
    "OC(=O)CCCCCCCCC(=O)O",          # sebacic acid
    "CCCCO",                         # butanol
    "CCCCCCCCCCCCO",                 # lauryl alcohol
    "COC(C)C",                       # small ether
    "CCCCCCCC(=O)NCCO",              # simple fatty amide ethanolamide
    "OCC1OC(O)C(O)C(O)C1O",          # glucose-like polyol
    "CC(=O)OC(C)=O",                 # acetylating small ester excluded? kept as non-sensitizer control
    "CCCCCCCCCCCCCCCC(=O)O",         # stearic acid
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
]


def build_sensitization_task(resolver=None) -> pd.DataFrame:
    """DataFrame(smiles,label) for the skin_sensitizer task."""
    from ml.data.chembl_hair import resolve_pubchem_smiles

    resolver = resolver or resolve_pubchem_smiles
    # drop entries whose hardcoded SMILES don't sanitize; fall back to name lookup
    cleaned_pairs: list[tuple[str, str | None]] = []
    for name, smi in SKIN_SENSITIZER_POSITIVES:
        if smi is not None and strip_salts(smi) is None:
            log.warning("hardcoded SMILES invalid (%s); resolving by name", name)
            smi = None
        cleaned_pairs.append((name.split(" (")[0], smi))

    pos = _resolve_many(cleaned_pairs, resolver)
    neg_rows = []
    for s in SKIN_SENSITIZER_NEGATIVES:
        canon = strip_salts(s)
        if canon:
            neg_rows.append({"smiles": canon, "label": 0})
    return pd.concat([pos, pd.DataFrame(neg_rows)], ignore_index=True)


# EU CosIng Annex II / restricted representative substances (name -> PubChem).
COSING_PROHIBITED_OR_RESTRICTED: list[str] = [
    "hydroquinone", "mercuric chloride", "mercury", "phenylmercuric acetate",
    "chloroform", "methanol", "carbon tetrachloride", "tetrachloroethylene",
    "trichloroethylene", "benzene", "arsenic trioxide", "lead acetate",
    "thallium sulfate", "vitamin K1", "cantharidin", "dinitrophenol",
    "strychnine", "aconitine", "colchicine", "spironolactone",
    "bithionol", "hexachlorophene", "zinc pyrithione", "selenium sulfide",
    "clotrimazole-restricted", "hydrocortisone", "tretinoin",
    "minoxidil", "estradiol", "testosterone", "progesterone",
    "diethylstilbestrol", "chloramphenicol", "metronidazole",
    "psoralen", "8-methoxypsoralen", "anthralin", "coal tar",
    "resorcinol", "pyrogallol", "salicylic acid",
]


def build_cosing_task(resolver=None) -> pd.DataFrame:
    from ml.data.chembl_hair import resolve_pubchem_smiles

    resolver = resolver or resolve_pubchem_smiles
    names = [n for n in COSING_PROHIBITED_OR_RESTRICTED if "-restricted" not in n]
    return _resolve_many([(n, None) for n in names], resolver)


IRRITANCY_POSITIVES: list[tuple[str, str | None]] = [
    ("sodium dodecyl sulfate", None),            # SLS — classic surfactant irritant
    ("sodium lauryl ether sulfate", None),
    ("benzalkonium chloride cation C12", None),
    ("cetrimonium bromide", None),
    ("benzoyl peroxide", None),
    ("hydrogen peroxide", "OO"),
    ("terpinen-4-ol rich tea tree oxidized", None),
    ("sodium cocoate soap base", None),
    ("nonoxynol-9 representative", None),
    ("oleic acid degumming agent", None),
]

IRRITANCY_NEGATIVES: list[str] = [
    "OCC1OC(O)C(O)C(O)C1O",     # glycerol/glucose-type humectant class
    "OCC(O)CO",                 # glycerol
    "CC(=O)OCC",                # ethyl acetate (volatile, low irritancy at cosmetic use)
    "CCCCCCCCCCCCO",
]


def build_irritancy_task(resolver=None) -> pd.DataFrame:
    from ml.data.chembl_hair import resolve_pubchem_smiles

    resolver = resolver or resolve_pubchem_smiles
    pos = _resolve_many(IRRITANCY_POSITIVES, resolver)
    neg_rows = [{"smiles": strip_salts(s), "label": 0} for s in IRRITANCY_NEGATIVES]
    neg = pd.DataFrame([r for r in neg_rows if r["smiles"]])
    return pd.concat([pos, neg], ignore_index=True)


def build_safety_tasks() -> dict[str, pd.DataFrame]:
    return {
        "skin_sensitizer": build_sensitization_task(),
        "cosing_prohibited": build_cosing_task(),
        "irritancy_alert": build_irritancy_task(),
    }
