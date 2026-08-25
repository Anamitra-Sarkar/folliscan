"""ChEMBL-backed hair-health task builder.

Resolves hair-biology protein targets by name via the ChEMBL REST API (never
hardcoding unstable target IDs), pulls curated bioactivities with pChEMBL
values, thresholds them into binary task labels, and merges a
literature-curated set of known hair-growth actives.

Tasks produced (task id -> rule):
  SULT1A1_active           pChEMBL >= 5 vs SULT1A1
  SRD5A1_inhibitor         pChEMBL >= 5 vs steroid 5-alpha-reductase (isoforms 1/2)
  AR_antagonist            pChEMBL >= 6 vs androgen receptor in antagonist-classified assays
  Wnt_bcatenin_activator   pChEMBL >= 5 vs beta-catenin / TCF-LEF reporter assays
  FGF7_KGF_active          pChEMBL >= 5 vs FGF7 (KGF) or FGFR2(IIIb)
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
import requests

from rdkit import RDLogger
from ml.data.featurize import strip_salts

RDLogger.DisableLog("rdApp.error")
log = logging.getLogger(__name__)

BASE = "https://www.ebi.ac.uk/chembl/api/data"
SESSION = requests.Session()
SESSION.headers.update({
    "Accept": "application/json",
    "User-Agent": "folliscan-data-builder/1.0 (cosmetic-safety research)",
})


def _get(url: str, params: dict | None = None, retries: int = 4) -> dict | None:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException as e:
            log.warning("chembl request failed (%s/%s): %s", attempt + 1, retries, e)
        time.sleep(1.5 * (attempt + 1))
    return None


def resolve_target_chembl_id(query: str) -> str | None:
    """Resolve the best-matching human single-protein ChEMBL target.

    Candidates are scored by keyword overlap with the query against the
    preferred name and synonyms, so generic search ranking cannot silently
    return an unrelated protein.
    """
    data = _get(f"{BASE}/target/search.json", {"q": query, "limit": 30})
    if not data:
        return None

    def keywords(s: str) -> set[str]:
        return {w.strip("()").lower() for w in s.replace("-", " ").replace("/", " ").split()
                if len(w) > 2}

    qkw = keywords(query)
    scored = []
    for t in data.get("targets", []):
        if t.get("target_type") != "SINGLE PROTEIN":
            continue
        names = {t.get("pref_name", "") or ""}
        for comp in t.get("target_components", []) or []:
            for syn in (comp.get("target_component_synonyms") or []):
                names.add(syn.get("component_synonym", "") or "")
        text = " ".join(names).lower()
        score = len(qkw & keywords(text))
        human_bonus = 2 if t.get("organism") == "Homo sapiens" else 0
        scored.append((score + human_bonus, t["target_chembl_id"]))
    if not scored:
        return None
    scored.sort(reverse=True)
    best_score, best_id = scored[0]
    return best_id if best_score >= max(1, min(len(qkw), 2)) else None


def fetch_target_activities(target_chembl_id: str, min_pchembl: float | None = 5.0,
                            max_records: int = 40000) -> pd.DataFrame:
    """Paginated pull of activities for one human target. With
    min_pchembl=None, ALL measured activities are pulled (including
    inactive/unquantified records used as negative evidence)."""
    rows = []
    offset = 0
    page = 500
    while offset < max_records:
        params = {
            "target_chembl_id": target_chembl_id,
            "limit": page, "offset": offset,
        }
        if min_pchembl is not None:
            params["pchembl_value__gte"] = min_pchembl
        data = _get(f"{BASE}/activity.json", params)
        if not data:
            break
        acts = data.get("activities", [])
        rows.extend(acts)
        offset += len(acts)
        if len(acts) < page:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    wanted = [
        "molecule_chembl_id", "canonical_smiles", "pchembl_value",
        "assay_chembl_id", "assay_description", "activity_type",
        "standard_type", "target_pref_name", "data_validity_comment",
    ]
    keep = df[[c for c in wanted if c in df.columns]].copy()
    keep = keep[keep["canonical_smiles"].notna()]
    # drop unreliable flags per ChEMBL curation guidance
    if "data_validity_comment" in keep.columns:
        bad = keep["data_validity_comment"].isin(
            ["Potential transcription errors", "Outside typical range"])
        keep = keep[~bad]
    return keep


def fetch_activities_for_assay_query(term: str, max_records: int = 40000) -> pd.DataFrame:
    """Find ChEMBL assays by description keyword (e.g. 'Wnt', 'TCF/LEF reporter')
    and pull every measured activity for them. Reporter-style endpoints are
    pathway-level and are not attached to a single target."""
    data = _get(f"{BASE}/assay.json", {"q": term, "limit": 25})
    if not data or not data.get("assays"):
        return pd.DataFrame()
    frames = []
    for a in data["assays"][:10]:
        aid = a.get("assay_chembl_id")
        if not aid:
            continue
        rows, offset, page = [], 0, 500
        while offset < max_records:
            d = _get(f"{BASE}/activity.json",
                     {"assay_chembl_id": aid, "limit": page, "offset": offset})
            if not d:
                break
            got = d.get("activities", [])
            rows.extend(got)
            offset += len(got)
            if len(got) < page:
                break
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


_UNITS_TO_MOLAR = {"pm": 1e-12, "nm": 1e-9, "um": 1e-6, "µm": 1e-6, "mm": 1e-3,
                   "cm": None, "M": 1.0}


def _estimate_pchembl(df: pd.DataFrame) -> pd.Series:
    """Fallback pActivity = -log10(molar standard value) when ChEMBL has not
    published a pchembl_value (common for functional antagonist assays)."""
    import numpy as np

    def one(row):
        try:
            if pd.notna(row.get("pchembl_value")):
                return float(row["pchembl_value"])
            rel = str(row.get("relation") or "=")
            if rel not in ("=", "<", "~", ""):
                return float("nan")   # '>', '>>' -> too weak / unquantified
            val = float(row.get("standard_value"))
            unit = str(row.get("standard_units") or "").strip().lower()
            factor = _UNITS_TO_MOLAR.get(unit)
            if factor is None or factor <= 0 or val <= 0:
                return float("nan")
            molar = val * factor
            if molar >= 1:
                return float("nan")
            return -np.log10(molar)
        except (TypeError, ValueError):
            return float("nan")

    return df.apply(one, axis=1)


def _consensus_labels(acts: pd.DataFrame, threshold: float,
                      assay_must_contain: str | None = None) -> pd.DataFrame:
    """Molecule-level consensus over activity records.

    A measurement votes positive when pChEMBL >= threshold. Records without a
    pChEMBL value (tested but unquantified/inactive) vote negative. Molecules
    are positive on majority-positive votes (ties -> positive).
    """
    a = acts.copy()
    if assay_must_contain:
        mask = a["assay_description"].str.contains(
            assay_must_contain, case=False, na=False)
        a = a[mask]
    if a.empty:
        return pd.DataFrame(columns=["smiles", "label"])
    a["smiles"] = a["canonical_smiles"].map(strip_salts)
    a = a.dropna(subset=["smiles"])
    a["pchembl_num"] = pd.to_numeric(a.get("pchembl_value"), errors="coerce")
    est = _estimate_pchembl(a)
    a["effective_p"] = a["pchembl_num"].fillna(est)
    a["vote_pos"] = a["effective_p"].fillna(0.0) >= threshold
    agg = a.groupby("smiles")["vote_pos"].mean().reset_index(name="frac_positive")
    agg["label"] = (agg["frac_positive"] > 0.5).astype(int)
    return agg[["smiles", "label"]]


# Literature-curated hair-growth positives (peer-reviewed evidence of scalp-hair
# growth stimulation or alopecia-pathway modulation). SMILES given only when
# confidently known; otherwise resolved by exact name via PubChem PUG REST.
HAIR_GROWTH_LITERATURE: list[tuple[str, str | None]] = [
    ("minoxidil", "CNC1=NC(N)=C(N)N=C1CO"),
    ("finasteride", None),
    ("dutasteride", None),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("ketoconazole", None),
    ("tretinoin", None),
    ("melatonin", None),
    ("latanoprost", None),
    ("bimatoprost", None),
    ("ricinoleic acid", None),
    ("carnosic acid", None),
    ("ursolic acid", None),
    ("procyanidin B2", None),
    ("alpha-lipoic acid", "OC(=O)CCCCC1SSCC1"),
    ("equol", None),
    ("taxifolin", None),
    ("wedelolactone", None),
    ("biotin", None),
    ("azelaic acid", "OC(=O)CCCCCCCC(=O)O"),
]


def resolve_pubchem_smiles(name: str) -> str | None:
    """Name -> canonical SMILES via PubChem PUG REST (None on miss). Retries
    politely through transient blocks/rate limits."""
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{requests.utils.quote(name)}/property/CanonicalSMILES/JSON")
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=30)
            if r.status_code == 200:
                props = r.json().get("PropertyTable", {}).get("Properties", [])
                return props[0].get("CanonicalSMILES") if props else None
            if r.status_code in (403, 429, 503):
                time.sleep(1.5 * (attempt + 1))   # polite backoff, then retry
                continue
            return None                            # definitive miss (404 etc.)
        except requests.RequestException:
            time.sleep(1.0)
    log.warning("pubchem resolution failed after retries: %s", name)
    return None


def literature_hair_growth_set() -> pd.DataFrame:
    """Curated SMILES for compounds with published hair-growth effects."""
    rows = []
    for name, smi in HAIR_GROWTH_LITERATURE:
        if smi is None:
            smi = resolve_pubchem_smiles(name)
            if smi is None:
                log.warning("could not resolve literature molecule via PubChem: %s", name)
                continue
        canon = strip_salts(smi)
        if canon is None:
            log.warning("literature molecule invalid: %s (%s)", name, smi)
            continue
        rows.append({"name": name, "smiles": canon, "label": 1})
    return pd.DataFrame(rows)


def build_hair_tasks(max_records_per_target: int | None = None) -> dict[str, pd.DataFrame]:
    """Returns {task_id: DataFrame(smiles,label)} for all six hair tasks."""
    if max_records_per_target is None:
        max_records_per_target = int(os.environ.get("FOLLISCAN_MAX_RECORDS", "40000"))
    specs = [
        ("SULT1A1_active", ["cytosolic sulfotransferase 1A1", "SULT1A1"], 5.0, None),
        ("SRD5A1_inhibitor", ["steroid 5-alpha reductase 1", "testosterone 5-alpha reductase"], 5.0, None),
        # functional AR assays: antagonists/inhibitors/blockers of AR signalling
        ("AR_antagonist", ["androgen receptor"], 6.0, "antagon|inhibit|block"),
        ("Wnt_bcatenin_activator", ["beta-catenin"], 5.0, None),
        ("FGF7_KGF_active", ["keratinocyte growth factor", "fibroblast growth factor receptor 2"], 5.0, None),
    ]
    assay_search = {
        "AR_antagonist": ["androgen receptor antagonist"],
        "Wnt_bcatenin_activator": ["Wnt beta-catenin pathway", "TCF LEF reporter Wnt"],
    }
    out: dict[str, pd.DataFrame] = {}
    for task_id, queries, thr, assay_kw in specs:
        frames = []
        for q in queries:
            tid = resolve_target_chembl_id(q)
            if not tid:
                log.warning("no ChEMBL target resolved for %r", q)
                continue
            log.info("task=%s query=%r -> %s", task_id, q, tid)
            acts = fetch_target_activities(tid, min_pchembl=None,
                                           max_records=max_records_per_target)
            if not acts.empty:
                frames.append(acts)
        for term in assay_search.get(task_id, []):
            acts = fetch_activities_for_assay_query(term,
                                                    max_records=max_records_per_target)
            if not acts.empty:
                log.info("task=%s assay-search=%r -> %d records",
                         task_id, term, len(acts))
                frames.append(acts)
        if not frames:
            out[task_id] = pd.DataFrame(columns=["smiles", "label"])
            continue
        merged = pd.concat(frames).drop_duplicates(subset=["molecule_chembl_id", "assay_chembl_id"])
        out[task_id] = _consensus_labels(merged, thr, assay_must_contain=assay_kw)

    lit = literature_hair_growth_set()
    out["hair_growth_lit_positive"] = lit.rename(columns={"name": "source_name"})[
        ["smiles", "label"]]
    return out
