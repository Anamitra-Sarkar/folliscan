"""Single source of truth for the 21 prediction tasks (order = model output index)."""

TASK_REGISTRY = [
    {"id": "SULT1A1_active", "group": "hair", "desc": "SULT1A1 bioactivation competence (minoxidil-type response)"},
    {"id": "AR_antagonist", "group": "hair", "desc": "Androgen receptor antagonism"},
    {"id": "SRD5A1_inhibitor", "group": "hair", "desc": "5-alpha reductase type 1 inhibition"},
    {"id": "Wnt_bcatenin_activator", "group": "hair", "desc": "Wnt/beta-catenin pathway activation"},
    {"id": "FGF7_KGF_active", "group": "hair", "desc": "KGF/FGF7 fibroblast pathway activity"},
    {"id": "hair_growth_lit_positive", "group": "hair", "desc": "Literature-curated hair-growth positive"},
    {"id": "NR-AR", "group": "tox", "desc": "Tox21 androgen receptor agonist/antagonist"},
    {"id": "NR-AR-LBD", "group": "tox", "desc": "Tox21 androgen receptor LBD"},
    {"id": "NR-AhR", "group": "tox", "desc": "Tox21 aryl hydrocarbon receptor"},
    {"id": "NR-Aromatase", "group": "tox", "desc": "Tox21 aromatase inhibition"},
    {"id": "NR-ER", "group": "tox", "desc": "Tox21 estrogen receptor agonist/antagonist"},
    {"id": "NR-ER-LBD", "group": "tox", "desc": "Tox21 estrogen receptor LBD"},
    {"id": "NR-PPAR-gamma", "group": "tox", "desc": "Tox21 PPAR-gamma"},
    {"id": "SR-ARE", "group": "tox", "desc": "Tox21 antioxidant response element"},
    {"id": "SR-ATAD5", "group": "tox", "desc": "Tox21 ATAD5 DNA damage response"},
    {"id": "SR-HSE", "group": "tox", "desc": "Tox21 heat shock response"},
    {"id": "SR-MMP", "group": "tox", "desc": "Tox21 mitochondrial membrane potential disruption"},
    {"id": "SR-p53", "group": "tox", "desc": "Tox21 p53 stress response"},
    {"id": "skin_sensitizer", "group": "safety", "desc": "Skin sensitization positive (LLNA/KeratinoSens-informed)"},
    {"id": "cosing_prohibited", "group": "safety", "desc": "Matches EU CosIng Annex II prohibited/restricted substance"},
    {"id": "irritancy_alert", "group": "safety", "desc": "Structural/class-based skin irritancy alert"},
]

assert len(TASK_REGISTRY) == 21
for i, t in enumerate(TASK_REGISTRY):
    t["index"] = i
    assert set(t) == {"id", "group", "desc", "index"}

TASK_IDS = [t["id"] for t in TASK_REGISTRY]
GROUPS = ["hair", "tox", "safety"]
GROUP_INDICES = {
    g: [t["index"] for t in TASK_REGISTRY if t["group"] == g] for g in GROUPS
}
TASK_INDEX = {t["id"]: t["index"] for t in TASK_REGISTRY}


def get_registry() -> list[dict]:
    return [dict(t) for t in TASK_REGISTRY]
