"""Stage 2 report: how much does normalisation raise name similarity on true pairs,
compared with hard negatives?

Pairs are drawn from **val-split** S1 entities only. The lexicon and region map were
learned on the train split, so this is out of sample.
  true pairs      random true pairs, plus extra true pairs with Indic-script candidates so every script has support
  same_building   non-matching records sharing the S1's "house number + street word"
                  (basic_key prefix). Selected on address only, so no bias towards similar names
  same_name       non-matching records with an identical basic_key name (always 100 before, by construction)
Name similarity is the rapidfuzz token-set ratio.
  before  basic_key names (Stage 0/1)
  after   best of {core, alias core} x {core, alias core}: transliterated, legal form and
          titles removed, aliases split out

Usage:  python src/norm_report.py      (writes notes/normalization_report.md)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.metrics import roc_auc_score

from data import NOTES_DIR, load, load_others, load_truth
from memtrack import stage
import pyarrow as pa

from normalize import addr_prefix_hash, hash_key
from normalize_build import load_norm
from prepare import UID_BASE
from splits import load_splits

SEED = 7
N_TRUE, N_PER_INDIC_SCRIPT, N_NEG_S1 = 40_000, 4_000, 40_000
INDIC_GROUPS = {"devanagari", "gurmukhi", "gujarati", "bengali"}
NAME_COLS = ["uid", "name_key"]
NORM_COLS = ["uid", "name_script", "name_core", "alias_core", "legal_form", "house_no", "street_word",
             "city", "region", "postcode", "hs_key", "core_h", "hs_h"]


def _norm(uids: np.ndarray, which: str) -> pd.DataFrame:
    f = [("uid", "in", pd.unique(uids).tolist())]
    if which == "s1":
        return load_norm("train", 1, NORM_COLS, f)
    return pd.concat([load_norm("train", s, NORM_COLS, f) for s in (2, 3)], ignore_index=True)


def _basic(uids: np.ndarray, which: str) -> pd.DataFrame:
    f = [("uid", "in", pd.unique(uids).tolist())]
    return load("train", 1, NAME_COLS + ["country"], f) if which == "s1" else load_others("train", NAME_COLS, f)


def sample_pairs() -> pd.DataFrame:
    sp = load_splits()
    val = sp.uid[sp.split == "val"].to_numpy()
    tr = load_truth(["s1_uid", "m_uid"])
    tr = tr[tr.s1_uid.isin(val)]
    rnd = tr.sample(N_TRUE, random_state=SEED)
    # extra Indic-script true pairs (the candidate script is only known after normalisation)
    scripts = pd.concat([load_norm("train", s, ["uid", "name_script"]) for s in (2, 3)])
    scripts = scripts[scripts.name_script.astype(str).isin(INDIC_GROUPS)]
    ind = tr.merge(scripts, left_on="m_uid", right_on="uid")
    extra = ind.sample(frac=1, random_state=SEED).groupby("name_script", observed=True).head(N_PER_INDIC_SCRIPT)[["s1_uid", "m_uid"]]
    true = pd.concat([rnd, extra]).drop_duplicates().assign(kind="true")

    # negatives: same building / same exact name, for a sample of val S1 entities
    s1 = load("train", 1, ["uid", "country", "name_h", "addr_key"], [("uid", "in", pd.Series(val).sample(N_NEG_S1, random_state=SEED).tolist())])
    s1["ap_h"] = addr_prefix_hash(s1.pop("addr_key"))
    oth = load_others("train", ["uid", "country", "name_h", "addr_key"])
    oth["ap_h"] = addr_prefix_hash(oth.pop("addr_key"))
    owner = tr.set_index("m_uid").s1_uid
    all_owner = load_truth(["s1_uid", "m_uid"]).set_index("m_uid").s1_uid
    negs = []
    for kind, key in (("same_building", "ap_h"), ("same_name", "name_h")):
        grp = oth[oth[key] != 0].groupby(["country", key], observed=True).size()
        small = grp[grp <= 50].rename("g").reset_index()                          # exploration cap
        left = s1[s1[key] != 0].merge(small, on=["country", key])[["uid", "country", key]]
        j = left.merge(oth[oth[key].isin(left[key])][["uid", "country", key]], on=["country", key], suffixes=("", "_m"))
        j = j[j.uid_m.map(all_owner) != j.uid]
        negs.append(pd.DataFrame({"s1_uid": j.uid.values, "m_uid": j.uid_m.values, "kind": kind}))
    del oth
    neg = pd.concat(negs).drop_duplicates(["s1_uid", "m_uid", "kind"])
    neg = neg.sample(frac=1, random_state=SEED).groupby("kind").head(60_000)
    return pd.concat([true, neg], ignore_index=True)


def score_pairs(p: pd.DataFrame) -> pd.DataFrame:
    a, b = _norm(p.s1_uid.values, "s1").set_index("uid"), _norm(p.m_uid.values, "oth").set_index("uid")
    ab, bb = _basic(p.s1_uid.values, "s1").set_index("uid"), _basic(p.m_uid.values, "oth").set_index("uid")
    A, B = a.loc[p.s1_uid], b.loc[p.m_uid]
    p = p.assign(country=ab.loc[p.s1_uid, "country"].astype(str).values,
                 script=B.name_script.astype(str).values,
                 cand_source=np.where(p.m_uid.values // UID_BASE == 2, "S2", "S3"))
    p["script_group"] = np.where(p.script.isin(INDIC_GROUPS), p.script,
                                 np.where(p.script.isin(["latin", "latin_accented"]), p.script, "other_indic"))
    sim = lambda x, y: process.cpdist(list(x), list(y), scorer=fuzz.token_set_ratio, workers=-1)
    p["before"] = sim(ab.loc[p.s1_uid, "name_key"], bb.loc[p.m_uid, "name_key"])
    combos = [sim(A[c1], B[c2]) * (A[c1].values != "") * (B[c2].values != "")
              for c1 in ("name_core", "alias_core") for c2 in ("name_core", "alias_core")]
    p["after"] = np.max(combos, axis=0)
    for f in ("house_no", "street_word", "city", "region", "postcode", "legal_form"):
        both = (A[f].values != "") & (B[f].values != "")
        p[f"{f}_both"] = both
        p[f"{f}_eq"] = both & (A[f].values == B[f].values)
    p["core_eq"] = A.core_h.values == B.core_h.values
    p["hs_eq"] = (A.hs_h.values == B.hs_h.values) & (A.hs_h.values != 0)
    return p


def name_table(p: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = p.groupby(by + ["kind"], observed=True)
    t = pd.DataFrame({
        "pairs": g.size(),
        "mean_before": g.before.mean(), "mean_after": g.after.mean(),
        "share>=80_before": g.before.apply(lambda x: (x >= 80).mean()),
        "share>=80_after": g.after.apply(lambda x: (x >= 80).mean()),
    })
    t["lift"] = t.mean_after - t.mean_before
    return t


def separation(p: pd.DataFrame) -> pd.DataFrame:
    """AUC of name similarity for true pairs vs same-building negatives (1.0 = perfect)."""
    rows = []
    for c, d in p.groupby("country"):
        d = d[d.kind.isin(["true", "same_building"])]
        if d.kind.nunique() < 2:
            continue
        y = (d.kind == "true").to_numpy()
        rows.append({"country": c, "true": int(y.sum()), "same_building": int((~y).sum()),
                     "AUC_before": roc_auc_score(y, d.before), "AUC_after": roc_auc_score(y, d.after)})
    return pd.DataFrame(rows).set_index("country")


def address_table(p: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for (c, k), d in p.groupby(["country", "kind"]):
        r = {"pairs": len(d)}
        for f in ("house_no", "street_word", "city", "region", "postcode"):
            r[f"{f}: both present"] = d[f"{f}_both"].mean()
            r[f"{f}: equal | present"] = d[f"{f}_eq"].sum() / max(d[f"{f}_both"].sum(), 1)
        rows[(c, k)] = r
    return pd.DataFrame(rows).T


def certain_rule_check() -> pd.DataFrame:
    """Precision/recall on val (full pool) of exact-key 'certain match' rules:
      stage1      basic_key name + basic address prefix (Stage 1 baseline)
      core+hs     same core name AND same house number + street word (Stage 2 fields)
      core+hs+lf  ... AND compatible legal form (equal, or missing on one side)
      norm+hs     same full normalised name (legal form kept) AND same house no. + street word"""
    sp = load_splits()
    val = sp.uid[sp.split == "val"].to_numpy()
    tr = load_truth(["s1_uid", "m_uid"])
    truth = tr[tr.s1_uid.isin(val)]
    cols = ["uid", "core_h", "hs_h", "legal_form", "name_norm"]
    f = [("uid", "in", val.tolist())]
    s1 = load_norm("train", 1, cols, f).merge(load("train", 1, ["uid", "country", "name_h", "addr_key"], f), on="uid")
    s1["ap_h"] = addr_prefix_hash(s1.pop("addr_key"))
    s1["nn_h"] = hash_key(pa.array(s1.pop("name_norm"), pa.string()))
    rows = []
    for rule in ("stage1", "core+hs", "core+hs+lf", "norm+hs"):
        parts = []
        for s in (2, 3):
            o = load_norm("train", s, cols).merge(load("train", s, ["uid", "country", "name_h", "addr_key"]), on="uid")
            o["ap_h"] = addr_prefix_hash(o.pop("addr_key"))
            o["nn_h"] = hash_key(pa.array(o.pop("name_norm"), pa.string()))
            keys = {"stage1": ["name_h", "ap_h"], "core+hs": ["core_h", "hs_h"], "core+hs+lf": ["core_h", "hs_h"],
                    "norm+hs": ["nn_h", "hs_h"]}[rule]
            l, r = s1, o
            for k in keys:
                l, r = l[l[k] != 0], r[r[k] != 0]
            j = l.merge(r, on=["country"] + keys, suffixes=("", "_m"))
            if rule == "core+hs+lf":
                j = j[(j.legal_form == j.legal_form_m) | (j.legal_form == "") | (j.legal_form_m == "")]
            parts.append(j[["uid", "uid_m"]])
            del o
        j = pd.concat(parts)
        hit = j.merge(truth, left_on=["uid", "uid_m"], right_on=["s1_uid", "m_uid"])
        rows.append({"rule": rule, "pairs": len(j), "precision": len(hit) / max(len(j), 1),
                     "recall_of_val_pairs": len(hit) / len(truth), "wrong_pairs": len(j) - len(hit)})
    return pd.DataFrame(rows)


def diagnostics(p: pd.DataFrame, k: int = 12) -> dict[str, pd.DataFrame]:
    """Examples: Latin true pairs whose similarity dropped, and India true-pair city mismatches."""
    t = p[(p.kind == "true")]
    drop = t[(t.after < t.before - 15) & t.script.isin(["latin", "latin_accented"])].sample(frac=1, random_state=SEED).head(k)
    cm = t[(t.country == "India") & t.city_both & ~t.city_eq].sample(frac=1, random_state=SEED).head(k)
    out = {}
    for name, d in (("latin_drops", drop), ("india_city_mismatch", cm)):
        a = _norm(d.s1_uid.values, "s1").set_index("uid").loc[d.s1_uid]
        b = _norm(d.m_uid.values, "oth").set_index("uid").loc[d.m_uid]
        ra = load("train", 1, ["uid", "business_name", "business_address"], [("uid", "in", d.s1_uid.tolist())]).set_index("uid").loc[d.s1_uid]
        rb = load_others("train", ["uid", "business_name", "business_address"], [("uid", "in", d.m_uid.tolist())]).set_index("uid").loc[d.m_uid]
        if name == "latin_drops":
            out[name] = pd.DataFrame({"S1 name": ra.business_name.values, "cand name": rb.business_name.values,
                                      "S1 core": a.name_core.values, "cand core": b.name_core.values,
                                      "cand alias": b.alias_core.values, "before": d.before.values, "after": d.after.values})
        else:
            out[name] = pd.DataFrame({"S1 address": ra.business_address.values, "cand address": rb.business_address.values,
                                      "S1 city": a.city.values, "cand city": b.city.values})
    return out


def main() -> None:
    pd.set_option("display.width", 220)
    with stage("norm_report"):
        p = score_pairs(sample_pairs())
        t_country = name_table(p, ["country"])
        t_script = name_table(p, ["country", "script_group"])
        sep = separation(p)
        addr = address_table(p)
        cert = certain_rule_check()
        diag = diagnostics(p)
        for t in (t_country, t_script, sep, addr, cert, *diag.values()):
            print(t.round(4).to_string(), "\n")
        md = ["# Stage 2 — normalisation before/after (generated by `src/norm_report.py`)", "",
              "Val-split S1 entities only. The lexicon and region map were learned on the train split. "
              "Name similarity is the rapidfuzz token-set ratio. *before* = basic_key names, "
              "*after* = best core/alias-core match (transliterated, legal form and titles removed).", "",
              "## Name similarity by country", "", t_country.round(3).to_markdown(), "",
              "## Name similarity by country and candidate script", "", t_script.round(3).to_markdown(), "",
              "## Separation: true pairs vs same-building negatives (ROC AUC of name similarity)", "",
              sep.round(4).to_markdown(), "",
              "## Address fields: share present on both sides, and share equal when present", "",
              addr.round(3).to_markdown(), "",
              "## Certain-match rules on val (full pool)", "", cert.round(5).to_markdown(index=False), "",
              "## Examples: Latin true pairs whose name similarity dropped by more than 15", "",
              diag["latin_drops"].to_markdown(index=False), "",
              "## Examples: India true pairs with different cities", "", diag["india_city_mismatch"].to_markdown(index=False), ""]
        (NOTES_DIR / "normalization_report.md").write_text("\n".join(md))
        p.to_parquet(NOTES_DIR.parent / "artifacts" / "norm_report_pairs.parquet", index=False)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- France check (unseen country)
FR_GROUPS = [  # (label, raw column, regex on the raw text, how many)
    ("legal form SARL/SAS/SA/SASU/EURL/SNC", "business_name", r"(?i)\b(?:s\.?a\.?r\.?l|s\.?a\.?s\.?u?|s\.?a|eurl|snc)\b", 8),
    ("street type rue/avenue/bd/chemin/place/allée/impasse/quai", "business_address",
     r"(?i)\b(?:bd|boulevard|av\.?|avenue|chemin|place|all[ée]e|impasse|quai|rue)\b", 8),
    ("region-level component present", "business_address",
     r"(?i)(?:hauts-de-france|nouvelle-aquitaine|pays de la loire|gironde|loire-atlantique|pas-de-calais|\bnord\b)", 6),
    ("accented name", "business_name", r"[À-ÿ]", 4),
    ("no house number in the raw address", "business_address", r"^[^0-9]*$", 4),
]
# The region names above only *select* examples for review; the pipeline itself learns
# regions from component positions and never uses this list.


def france_check(seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage of normalised fields per country (test S1 and S3), and 30 before/after
    examples of French rows, stratified by the patterns in FR_GROUPS."""
    cols = ["uid", "name_script", "name_core", "legal_form", "alias_core", "titles", "is_domain",
            "addr_norm", "house_no", "street", "city", "region", "localities", "postcode", "landmark"]
    rows, stats = [], []
    for s in (1, 3):
        raw = load("test", s, ["uid", "business_name", "business_address", "country"])
        nrm = load_norm("test", s, cols)
        d = raw.merge(nrm, on="uid")
        for c, g in d.groupby("country", observed=True):
            stats.append({"source": f"S{s}", "country": c, "rows": len(g),
                          "legal_form": (g.legal_form != "").mean(), "alias": (g.alias_core != "").mean(),
                          "house_no": (g.house_no != "").mean(), "street": (g.street != "").mean(),
                          "city": (g.city != "").mean(), "region": (g.region != "").mean(),
                          "postcode": (g.postcode != "").mean(), "landmark": g.landmark.mean(),
                          "non_latin_name": (~g.name_script.astype(str).isin(["latin", "latin_accented"])).mean()})
        fr = d[d.country == "France"].assign(source=f"S{s}")
        rows.append(fr.sample(min(len(fr), 60_000), random_state=seed))
    fr = pd.concat(rows, ignore_index=True)
    picked, used = [], set()
    for label, col, pat, k in FR_GROUPS:
        pool = fr[fr[col].str.contains(pat, regex=True) & ~fr.uid.isin(used)]
        take = pool.sample(min(k, len(pool)), random_state=seed).assign(group=label)
        used.update(take.uid)
        picked.append(take)
    ex = pd.concat(picked, ignore_index=True)
    st = pd.DataFrame(stats).set_index(["source", "country"]).sort_index()
    ex = ex[["group", "source", "business_name", "name_core", "legal_form", "business_address",
             "house_no", "street", "city", "region", "postcode"]]
    return st, ex


def write_france_check(path=NOTES_DIR / "france_check.md") -> None:
    st, ex = france_check()
    md = ["# France check (test set, unseen in training)", "",
          "Generated by `src/norm_report.py::france_check` from the final-mode test norm files.", "",
          "## Share of rows with each normalised field, per source and country", "", st.round(3).to_markdown(), "",
          "## 30 before/after examples (French rows)", ""]
    for g, d in ex.groupby("group", sort=False):
        md += [f"### {g}", "", d.drop(columns="group").to_markdown(index=False), ""]
    path.write_text("\n".join(md))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(st.round(3).to_string())
    print(ex.to_string(index=False))
