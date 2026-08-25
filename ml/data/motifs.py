"""Substructure Motif Encoder support: curated SMARTS library + subgraph matching.

Motifs cover three roles:
  - pharmacophore/benefit motifs (hair-health relevant chemistry)
  - toxicophores / regulatory structural alerts (hazard=True)
  - common functional groups (context features)

`hazard` flags feed the PINN-style regulatory constraint layer and the
explanation payload. Library is fixed at definition time; N_MOTIFS derives
from it so model input dim always matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")


@dataclass(frozen=True)
class Motif:
    id: str
    name: str
    smarts: str
    severity: str  # "info" | "alert" | "hazard"
    hazard: bool = False
    message: str = ""


MOTIF_LIBRARY: list[Motif] = [
    # --- reactive toxicophores / genotoxicity alerts ---
    Motif("aromatic_nitro", "Aromatic nitro", "[c][N+](=O)[O-]", "hazard", True,
          "Aromatic nitro group; potential mutagenicity alert (ICH M7 class)"),
    Motif("nitroso", "Nitroso", "[#6][NX2]=[OX1]", "hazard", True,
          "Nitroso group; strong mutagenicity concern"),
    Motif("n_nitrosamine", "N-nitrosamine", "[NX3]([#6])[NX2]=[OX1]", "hazard", True,
          "N-nitrosamine; potent mutagenic impurity class"),
    Motif("epoxide", "Epoxide", "[CX4r3]1[CX4r3]O1", "hazard", True, "Epoxide ring; direct alkylating electrophile"),
    Motif("aziridine", "Aziridine", "[NX3r3]1[CX4r3][CX4r3]1", "hazard", True, "Aziridine ring; DNA alkylation alert"),
    Motif("michael_acceptor_enone", "Michael acceptor (enone)", "[CX3]=[CX3][CX3](=[OX1])", "hazard", True,
          "alpha,beta-unsaturated carbonyl; protein-reactive Michael acceptor (skin sensitization)"),
    Motif("acrylamide", "Acrylamide motif", "[NX3,CX3][CX3](=O)[CX3]=[CX3]", "hazard", True, "Acrylamide-type Michael acceptor"),
    Motif("aldehyde", "Aldehyde", "[CX3H1](=O)[#6]", "alert", True, "Aldehyde; reactive and sensitizing potential"),
    Motif("acid_halide", "Acid halide", "[CX3](=[OX1])[F,Cl,Br,I]", "hazard", True, "Acid halide; highly reactive acylating agent"),
    Motif("anhydride", "Anhydride", "[CX3](=[OX1])[OX2][CX3](=[OX1])", "alert", True, "Carboxylic anhydride; sensitizer"),
    Motif("isocyanate", "Isocyanate", "[NX2]=[CX2]=[OX1]", "hazard", True, "Isocyanate; respiratory/skin sensitizer"),
    Motif("isothiocyanate", "Isothiocyanate", "[NX2]=[CX2]=[SX1]", "alert", True, "Isothiocyanate; sensitizer"),
    Motif("sulfonate_ester", "Sulfonate ester", "[SX4](=[OX1])(=[OX1])[OX2][#6]", "hazard", True,
          "Sulfonate ester; ICH M7 alkylating alert"),
    Motif("alkyl_halide", "Alkyl halide", "[Cl,Br,I][CX4;H2,H3][!$([N,O,S,F])]", "alert", True, "Alkyl halide; SN2 alkylating potential"),
    Motif("benzyl_halide", "Benzyl halide", "[Cl,Br,I]C[c]", "hazard", True, "Benzylic halide; resonance-stabilized alkylator"),
    Motif("allyl_halide", "Allyl halide", "[Cl,Br,I]C=C", "alert", True, "Allylic halide; reactive allylation"),
    Motif("beta_propiolactone", "beta-propiolactone", "[CX3](=O)1CCO1", "hazard", True, "Strained lactone; genotoxic alert"),
    Motif("hydrazine", "Hydrazine", "[NX3][NX3]", "hazard", True, "Hydrazine moiety; mutagenicity/metabolic alert"),
    Motif("hydroxylamine", "Hydroxylamine", "[NX3][OX2H1]", "alert", True, "Hydroxylamine; methemoglobinemia/mutagenicity concern"),
    Motif("aromatic_amine_primary", "Primary aromatic amine", "[NX3;H2][c]", "alert", True,
          "Primary aromatic amine; potential mutagenicity alert (ICH M7)"),
    Motif("azo", "Azo group", "[#6][NX2]=[NX2][#6]", "alert", True, "Azo bond; reductive cleavage to aromatic amines"),
    Motif("triazene", "Triazene", "[NX3][NX2]=[NX2]", "hazard", True, "Triazene; alkyl-diazonium generator"),
    Motif("diazonium", "Diazonium", "[NX2+]#[NX1]", "hazard", True, "Diazonium salt; potent alkylating species"),
    Motif("peroxide", "Organic peroxide", "[OX2][OX2]", "hazard", True, "Peroxide bond; oxidative stress generator"),
    Motif("quinone", "Quinone", "[CX3]=[CX3][CX3](=[OX1])[CX3](=[OX1])[CX3]=[CX3]", "hazard", True,
          "Quinone system; redox cycling and protein adduct formation"),
    Motif("catechol", "Catechol", "c(O)c(O)", "alert", False, "Catechol; oxidizes to quinone, melanin/oxidative chemistry"),
    Motif("hydroquinone", "Hydroquinone", "c(O)cc(O)", "alert", True, "Hydroquinone; restricted in cosmetics, depigmenting/irritant"),
    Motif("thiol", "Thiol", "[SX2H1]", "info", False, "Free thiol; nucleophilic, disulfide exchange"),
    Motif("boronic_acid", "Boronic acid", "[BX3](O)(O)", "alert", False, "Boronic acid; possible mutagenicity concern in Ames"),
    Motif("organophosphate", "Organophosphate", "[PX4](=O)([#8])([#8])", "alert", True, "Phosphate ester; cholinesterase-inhibition family"),
    Motif("carbamate", "Carbamate", "[NX3][CX3](=[OX1])[OX2]", "info", False, "Carbamate linkage"),
    Motif("furan", "Furan ring", "c1ccoc1", "alert", True, "Furan; CYP-mediated bioactivation alert"),

    # --- skin sensitization / irritancy relevant ---
    Motif("para_phenylenediamine", "p-phenylenediamine core", "[NX3]c1ccc(N)cc1", "hazard", True,
          "p-phenylenediamine scaffold; potent contact allergen"),
    Motif("alpha_methylene_gamma_butyrolactone", "Tulipalin-type lactone", "[CX3](=O)1OC(=C)C1", "alert", True,
          "Exocyclic methylene lactone; strong sensitizer (tulipalin-like)"),
    Motif("cinnamyl", "Cinnamyl aldehyde/alcohol", "[CX3]=[CX3][CX3H1](=O)c1ccccc1", "alert", True,
          "Cinnamaldehyde-type alpha,beta-unsaturated aldehyde; known fragrance allergen"),
    Motif("isothiazolinone", "Isothiazolinone", "c1csnn1=O", "hazard", True, "Isothiazolinone biocide; MIT/CMIT allergen family"),
    Motif("terpene_allylic_oxide", "Terpene allylic oxidation site", "[CX4]([CH3])C=C[CX4][OX2]", "info", False,
          "Allylic alcohol on terpene skeleton; autoxidation sensitization route"),
    Motif("quaternary_ammonium", "Quaternary ammonium", "[NX4+]", "alert", False,
          "Quaternary ammonium; surfactant irritancy class"),
    Motif("sulfate_ester", "Sulfate ester", "OS(=O)(=O)OC", "alert", False, "Sulfate ester; surfactant/irritancy class (e.g., SLS)"),
    Motif("long_chain_surfactant", "Long-chain amphiphile", "[CX4H3][CX4H2][CX4H2][CX4H2][CX4H2][CX4H2][CX4H2][CX4H2][SX4,NX4+,OX2]", "alert",
          False, "Long hydrophobic chain with polar head; membrane-disrupting irritancy profile"),
    Motif("strong_acid", "Strong acid group", "[SX4](=O)(=O)([OX2H1])", "info", False, "Sulfonic/carboxylic acidity"),
    Motif("phenol_group", "Phenol", "[OX2H1]c", "info", False, "Phenolic OH"),
    Motif("polyphenol_flavonoid_core", "Flavonoid-like polyphenol", "O=c1c(-c2ccccc2)oc2cc(O)cc(O)c12", "info", False,
          "Flavonoid core; antioxidant, Wnt-modulating literature signal"),

    # --- hair-biology benefit motifs ---
    Motif("pyrimidine_diamine_minoxidil_core", "Minoxidil-type pyrimidinediamine", "Nc1nc(N)c(N)n1", "info", False,
          "2,4-diaminopyrimidine core of minoxidil-family K_ATP openers"),
    Motif("steroid_nucleus", "Steroid nucleus", "C1CCC2C3CC(C4CCC(O)C4)CC3CCC2C1", "info", False,
          "Steroidal framework; 5a-reductase/AR pharmacology context"),
    Motif("amide_peptide_bond", "Amide bond", "[CX3](=O)[NX3]", "info", False, "Peptide/amide linkage; GHK-style peptide actives"),
    Motif("guanidine", "Guanidine", "[NX3][CX3](=[NX3])[NX3]", "info", False, "Guanidine; arginine/peptide chemistry"),
    Motif("imidazole_caffeine_like", "Xanthine/imidazole core", "n1cnc2c1C(=O)N(C)C(=O)N2C", "info", False,
          "Caffeine-family xanthine; follicle-stimulating literature signal"),
    Motif("retinoid_conjugated_acid", "Retinoid conjugated chain", "[CX3](=O)OC/C=C/C(C)=C/C(C)=C/C=C/c1ccccc1", "info", False,
          "Retinoic-acid-like polyene acid; epidermal turnover modulation"),
    Motif("prostaglandin_analog_chain", "Prostaglandin-like chain", "C[C@H](O)[CX4H2][CX4H2][CX4H2][CX3](=O)O", "info", False,
          "Prostaglandin-analog side chain (latanoprost family; hypertrichosis signal)"),
    Motif("azelaic_diacid", "Dicarboxylic acid (azelaic-type)", "OC(=O)CCCCCCCC(=O)O", "info", False,
          "Azelaic-acid-type diacid; anti-androgenic topical evidence"),
    Motif("ketoconazole_imidazole_ring", "Imidazole antifungal ring", "c1ncc[nH]1", "info", False,
          "Imidazole; ketoconazole-type anti-androgenic antifungal context"),

    # --- general functional-group context ---
    Motif("carboxylic_acid", "Carboxylic acid", "[CX3](=O)[OX2H1]", "info", False, ""),
    Motif("ester", "Ester", "[CX3](=O)[OX2][#6]", "info", False, ""),
    Motif("ether", "Ether", "[OD2]([#6])[#6]", "info", False, ""),
    Motif("amine_tertiary", "Tertiary amine", "[NX3;H0]([#6])([#6])[#6]", "info", False, ""),
    Motif("alcohol_aliphatic", "Aliphatic alcohol", "[OX2H1][CX4]", "info", False, ""),
    Motif("ketone", "Ketone", "[#6][CX3](=O)[#6]", "info", False, ""),
    Motif("aromatic_ring", "Aromatic ring", "c1ccccc1", "info", False, ""),
    Motif("halogen_aromatic", "Aryl halide", "[#9,#17,#35,#53][c]", "info", False, ""),
    Motif("nitrile", "Nitrile", "[NX1]#[CX2]", "info", False, ""),
    Motif("amide_secondary", "Secondary amide", "[CX3](=O)[NX3H1]", "info", False, ""),
]

assert len(MOTIF_LIBRARY) == len({m.id for m in MOTIF_LIBRARY}), "motif ids must be unique"
MOTIFS_BY_ID: dict[str, Motif] = {m.id: m for m in MOTIF_LIBRARY}
N_MOTIFS = len(MOTIF_LIBRARY)
HAZARD_MOTIF_IDS = [m.id for m in MOTIF_LIBRARY if m.hazard]
MOTIF_INDEX: dict[str, int] = {m.id: i for i, m in enumerate(MOTIF_LIBRARY)}

_compiled = None


def _compiled_smarts() -> list[tuple[Motif, Chem.Mol]]:
    global _compiled
    if _compiled is None:
        _compiled = []
        for m in MOTIF_LIBRARY:
            patt = Chem.MolFromSmarts(m.smarts)
            if patt is None:
                raise ValueError(f"Invalid SMARTS for motif {m.id}: {m.smarts}")
            _compiled.append((m, patt))
    return _compiled


def match_motifs(mol_or_smiles) -> list[tuple[str, tuple[int, ...]]]:
    """Return [(motif_id, matched_atom_indices)] for every library hit."""
    mol = mol_or_smiles if isinstance(mol_or_smiles, Chem.Mol) else Chem.MolFromSmiles(str(mol_or_smiles))
    if mol is None:
        return []
    out = []
    for motif, patt in _compiled_smarts():
        try:
            hits = mol.GetSubstructMatches(patt, uniquify=True)
        except Exception:
            continue
        if hits:
            atoms: list[int] = []
            for h in hits:
                atoms.extend(h)
            out.append((motif.id, tuple(sorted(set(atoms)))))
    return out


def motif_multihot(mol_or_smiles) -> list[int]:
    matches = match_motifs(mol_or_smiles)
    vec = [0] * N_MOTIFS
    for mid, _ in matches:
        vec[MOTIF_INDEX[mid]] = 1
    return vec


def hazard_flags(mol_or_smiles) -> dict[str, bool]:
    """Per-task hard-rule flags consumed by the PINN constraint layer."""
    ids = {mid for mid, _ in match_motifs(mol_or_smiles)}
    return {
        "mutagenicity_alert": any(
            i in ids for i in ("aromatic_nitro", "nitroso", "n_nitrosamine", "sulfonate_ester",
                               "benzyl_halide", "hydrazine", "triazene", "diazonium", "epoxide")
        ),
        "sensitization_alert": any(
            i in ids for i in ("michael_acceptor_enone", "acrylamide", "alpha_methylene_gamma_butyrolactone",
                               "cinnamyl", "isothiazolinone", "para_phenylenediamine", "aldehyde")
        ),
        "irritancy_alert": any(
            i in ids for i in ("quaternary_ammonium", "sulfate_ester", "long_chain_surfactant", "peroxide")
        ),
        "cosing_alert": any(i in ids for i in ("hydroquinone", "quinone")),
    }
