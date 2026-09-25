"""Stage 0 exploration functions, imported by notebooks/01_eda.ipynb.

Each function returns DataFrames and reads only the parquet columns it needs.
Hard-negative mining samples S1 entities and caps frequent names. That is fine for
exploration only; blocking must never drop frequent names.

Usage:  python src/eda.py        (prints every table and writes notes/eda_examples.md)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from data import NOTES_DIR, load, load_others, load_truth
from memtrack import stage
from prepare import UID_BASE

SPLITS = {"train": (1, 2, 3), "test": (1, 2, 3)}
FREQ_BINS = [0, 1, 2, 5, 10, 50, 200, np.inf]
FREQ_LABELS = ["1", "2", "3-5", "6-10", "11-50", "51-200", ">200"]


# ---------------------------------------------------------------- counts
def source_counts() -> pd.DataFrame:
    """Rows per split x source x country, with the missing test S2 file shown explicitly."""
    rows = []
    for split, sources in SPLITS.items():
        for s in sources:
            try:
                vc = load(split, s, ["country"])["country"].value_counts()
            except FileNotFoundError:
                rows.append({"split": split, "source": f"S{s}", "country": "(file missing)", "rows": 0})
                continue
            rows += [{"split": split, "source": f"S{s}", "country": c, "rows": n} for c, n in vc.items()]
    df = pd.DataFrame(rows)
    return df.pivot_table(index=["split", "source"], columns="country", values="rows",
                          aggfunc="sum", fill_value=0, margins=True, margins_name="total")


# ---------------------------------------------------------------- labels
def s1_match_counts() -> pd.DataFrame:
    """Every train S1 entity with its number of true matches, split into S2 and S3."""
    s1 = load("train", 1, ["uid", "country"])
    tr = load_truth(["s1_uid", "m_uid"])
    is_s2 = tr.m_uid // UID_BASE == 2
    s1["n_matches"] = s1.uid.map(tr.s1_uid.value_counts()).fillna(0).astype("int16")
    s1["n_s2"] = s1.uid.map(tr.s1_uid[is_s2].value_counts()).fillna(0).astype("int16")
    s1["n_s3"] = s1.n_matches - s1.n_s2
    return s1


def label_summary(mc: pd.DataFrame) -> pd.DataFrame:
    g = mc.groupby("country", observed=True)
    out = pd.DataFrame({
        "s1_entities": g.size(),
        "singleton_share": g.n_matches.apply(lambda x: (x == 0).mean()),
        "mean_matches": g.n_matches.mean(),
        "mean_matches_non_singleton": g.n_matches.apply(lambda x: x[x > 0].mean()),
        "max_matches": g.n_matches.max(),
        "has_S2_and_S3": g.apply(lambda d: ((d.n_s2 > 0) & (d.n_s3 > 0)).mean(), include_groups=False),
    })
    tot = pd.DataFrame([{
        "s1_entities": len(mc), "singleton_share": (mc.n_matches == 0).mean(),
        "mean_matches": mc.n_matches.mean(), "mean_matches_non_singleton": mc.n_matches[mc.n_matches > 0].mean(),
        "max_matches": mc.n_matches.max(), "has_S2_and_S3": ((mc.n_s2 > 0) & (mc.n_s3 > 0)).mean()}], index=["all"])
    return pd.concat([out, tot])


def match_count_distribution(mc: pd.DataFrame) -> pd.DataFrame:
    return pd.crosstab(mc.n_matches, mc.country, margins=True, margins_name="total")


def assignment_checks() -> dict:
    """Is each S2/S3 record owned by at most one S1? Are there cross-country matches?
    What share of S2/S3 records match nothing (pure distractors)?"""
    tr = load_truth(["s1_uid", "m_uid"])
    s1 = load("train", 1, ["uid", "country"]).set_index("uid").country
    oth = load_others("train", ["uid", "country"])
    oth["matched"] = oth.uid.isin(tr.m_uid)
    owner_country = tr.s1_uid.map(s1).astype(str)
    cand_country = tr.m_uid.map(oth.set_index("uid").country).astype(str)
    return {
        "true_pairs": len(tr),
        "S2/S3 ids matched to >1 S1": int(tr.m_uid.duplicated().sum()),
        "truth ids missing from S2/S3 files": int((~tr.m_uid.isin(oth.uid)).sum()),
        "cross-country true pairs": int((owner_country != cand_country).sum()),
        "S2 matched share": oth.matched[oth.uid // UID_BASE == 2].mean(),
        "S3 matched share": oth.matched[oth.uid // UID_BASE == 3].mean(),
    }


# ---------------------------------------------------------------- noise
NOISE_PATTERNS = {
    "devanagari_name": ("business_name", r"[ऀ-ॿ]"),
    "non_ascii_name": ("business_name", r"[^\x00-\x7f]"),
    "junk_prefix_name": ("business_name", r"^[^\wऀ-ॿ]"),
    "domain_like_name": ("business_name", r"\.(?:com|in|net|org|fr|co)\b"),
    "legal_suffix": ("business_name", r"(?i)\b(?:inc|corp|corporation|llc|ltd|limited|pvt|private|llp|co|sa|sarl|sas|snc|eurl|gmbh)\b\.?"),
    "landmark_addr": ("business_address", r"(?i)\b(?:near|nr|opp|opposite|behind|beside|pres de|près de|en face)\b"),
    "5digit_code": ("business_address", r"\b\d{5}\b"),
    "6digit_pin": ("business_address", r"\b\d{6}\b"),
}


def noise_profile() -> pd.DataFrame:
    rows = []
    for split, sources in SPLITS.items():
        for s in sources:
            try:
                df = load(split, s, ["business_name", "business_address", "country"])
            except FileNotFoundError:
                continue
            for country, d in df.groupby("country", observed=True):
                r = {"split": split, "source": f"S{s}", "country": country, "rows": len(d),
                     "empty_addr": (d.business_address == "").mean(),
                     "upper_addr": (d.business_address.str.upper() == d.business_address)[d.business_address != ""].mean(),
                     "name_len": d.business_name.str.len().mean()}
                for k, (col, pat) in NOISE_PATTERNS.items():
                    r[k] = d[col].str.contains(pat, regex=True).mean()
                rows.append(r)
            del df
    return pd.DataFrame(rows).set_index(["split", "source", "country"])


def true_pair_similarity(n: int = 100_000, seed: int = 0) -> pd.DataFrame:
    """Similarity of a sample of true pairs, by country and candidate source."""
    tr = load_truth(["s1_uid", "m_uid"]).sample(n, random_state=seed)
    cols = ["uid", "country", "name_key", "addr_key", "business_name"]
    a = load("train", 1, cols, filters=[("uid", "in", tr.s1_uid.tolist())]).set_index("uid").loc[tr.s1_uid]
    b = load_others("train", cols, filters=[("uid", "in", tr.m_uid.tolist())]).set_index("uid").loc[tr.m_uid]
    na, nb, aa, ab = (x.tolist() for x in (a.name_key, b.name_key, a.addr_key, b.addr_key))
    df = pd.DataFrame({
        "country": a.country.astype(str).values,
        "cand_source": np.where(tr.m_uid.values // UID_BASE == 2, "S2", "S3"),
        "exact_name": np.array(na, dtype=object) == np.array(nb, dtype=object),
        "exact_addr": np.array(aa, dtype=object) == np.array(ab, dtype=object),
        "name_token_set": process.cpdist(na, nb, scorer=fuzz.token_set_ratio, workers=-1),
        "addr_token_set": process.cpdist(aa, ab, scorer=fuzz.token_set_ratio, workers=-1),
        "cand_empty_addr": (b.addr_key == "").values,
        "cand_devanagari": b.business_name.str.contains(r"[ऀ-ॿ]").values,
    })
    return df


def summarise_similarity(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["country", "cand_source"]).agg(
        pairs=("exact_name", "size"),
        exact_name=("exact_name", "mean"), exact_addr=("exact_addr", "mean"),
        name_tokset_median=("name_token_set", "median"),
        name_tokset_below_60=("name_token_set", lambda x: (x < 60).mean()),
        addr_tokset_median=("addr_token_set", "median"),
        cand_empty_addr=("cand_empty_addr", "mean"), cand_devanagari=("cand_devanagari", "mean"))


# ---------------------------------------------------------------- name frequency
def name_frequency(split: str) -> pd.DataFrame:
    """Share of records whose (country, normalised name) group has a given size.
    Computed separately for S1 and for the S2+S3 pool."""
    out = []
    for label, df in (("S1", load(split, 1, ["country", "name_h"])),
                      ("S2+S3", load_others(split, ["country", "name_h"]))):
        df = df[df.name_h != 0]
        size = df.groupby(["country", "name_h"], observed=True).name_h.transform("size")
        b = pd.cut(size, FREQ_BINS, labels=FREQ_LABELS)
        t = pd.crosstab(df.country, b, normalize="index")
        t.index = pd.MultiIndex.from_product([[label], t.index.astype(str)], names=["pool", "country"])
        out.append(t)
    return pd.concat(out)


def top_names(split: str, k: int = 15) -> pd.DataFrame:
    """Most frequent normalised names in the S2+S3 pool, per country."""
    df = load_others(split, ["country", "name_h"])
    df = df[df.name_h != 0]
    vc = df.groupby(["country", "name_h"], observed=True).size().rename("records").reset_index()
    top = vc.sort_values("records", ascending=False).groupby("country", observed=True).head(k)
    keys = load_others(split, ["name_h", "name_key"], filters=[("name_h", "in", top.name_h.tolist())]
                       ).drop_duplicates("name_h")
    return top.merge(keys, on="name_h")[["country", "name_key", "records"]].sort_values(
        ["country", "records"], ascending=[True, False], ignore_index=True)


def same_name_pressure() -> pd.DataFrame:
    """For every train S1 entity, the S2/S3 records sharing its exact normalised name in the
    same country: how many there are, and how many of them are true matches. This bounds
    the precision of exact-name blocking and shows how large name groups get."""
    s1 = load("train", 1, ["uid", "country", "name_h"])
    oth = load_others("train", ["uid", "country", "name_h"])
    grp = oth[oth.name_h != 0].groupby(["country", "name_h"], observed=True).size().rename("same_name_records")
    s1 = s1.join(grp, on=["country", "name_h"]).fillna({"same_name_records": 0})
    tr = load_truth(["s1_uid", "m_uid"])
    tr["m_name_h"] = tr.m_uid.map(oth.set_index("uid").name_h)
    tr["s1_name_h"] = tr.s1_uid.map(s1.set_index("uid").name_h)
    same = tr[tr.m_name_h == tr.s1_name_h].s1_uid.value_counts()
    s1["same_name_true"] = s1.uid.map(same).fillna(0)
    s1["same_name_false"] = s1.same_name_records - s1.same_name_true
    b = pd.cut(s1.same_name_false, [-1, 0, 1, 5, 20, 100, np.inf], labels=["0", "1", "2-5", "6-20", "21-100", ">100"])
    return pd.crosstab(s1.country, b, normalize="index")


# ---------------------------------------------------------------- candidate budget (Stage 3b)
def candidate_budget() -> pd.DataFrame:
    """Per split and country: the all-pairs denominator used by the reduction ratio,
    |S1| x |S2+S3 in the same country|, and for train the true-pair count. The smallest
    average candidate count that can reach a pair completeness c is c x mean true matches,
    because even a perfect pruner must keep every true pair it wants to count."""
    rows = []
    for split in ("train", "test"):
        n1 = load(split, 1, ["country"]).country.value_counts()
        n2 = load_others(split, ["country"]).country.value_counts()
        for c in n1.index:
            rows.append({"split": split, "country": c, "S1": int(n1[c]), "S2+S3": int(n2.get(c, 0)),
                         "all_pairs": int(n1[c]) * int(n2.get(c, 0))})
    df = pd.DataFrame(rows).set_index(["split", "country"])
    mc = s1_match_counts()
    g = mc.groupby("country", observed=True).n_matches
    df.loc[("train", slice(None)), "true_pairs"] = g.sum().rename(lambda c: ("train", c))
    df["mean_true_per_S1"] = df.true_pairs / df.S1
    df["min_avg_cands_for_95pct"] = 0.95 * df.mean_true_per_S1
    df["reduction_ratio_at_4_per_S1"] = 1 - 4 * df.S1 / df.all_pairs
    return df


def oracle_topk_completeness(max_k: int = 10) -> pd.DataFrame:
    """Pair completeness if each S1 kept at most k candidates and the ranking were perfect:
    sum(min(n_i, k)) / sum(n_i). This is the ceiling for any fixed top-k cap. An adaptive
    cutoff can do better at the same average size by spending more slots on entities with
    many matches and fewer on singletons."""
    mc = s1_match_counts()
    n = mc.n_matches.to_numpy()
    return pd.DataFrame({"k": range(1, max_k + 1),
                         "completeness_ceiling": [np.minimum(n, k).sum() / n.sum() for k in range(1, max_k + 1)],
                         "avg_cands_if_oracle": [np.minimum(n, k).mean() for k in range(1, max_k + 1)]}).set_index("k")


# ---------------------------------------------------------------- examples
def example_groups(n_per_country: int = 9, n_singletons: int = 2, seed: int = 7) -> list[dict]:
    mc = s1_match_counts()
    pick = pd.concat(
        [mc[(mc.country == c) & (mc.n_matches > 0)].sample(n_per_country, random_state=seed) for c in ("US", "India")]
        + [mc[mc.n_matches == 0].sample(n_singletons, random_state=seed)])
    cols = ["uid", "entity_id", "business_name", "business_address", "country"]
    s1 = load("train", 1, cols, filters=[("uid", "in", pick.uid.tolist())]).set_index("uid")
    tr = load_truth(["s1_uid", "m_uid"])
    tr = tr[tr.s1_uid.isin(pick.uid)]
    oth = load_others("train", cols, filters=[("uid", "in", tr.m_uid.tolist())]).set_index("uid")
    groups = []
    for u in pick.uid:
        m = tr.m_uid[tr.s1_uid == u]
        groups.append({"s1": s1.loc[u].to_dict(), "matches": oth.loc[m].to_dict("records")})
    return groups


ADDR_PREFIX_RE = r"(\d+\s+[^\s\d]+)"  # first "house number + next word", e.g. "994 miller"


def addr_prefix_hash(addr_key: pd.Series) -> np.ndarray:
    """Hash of the first number+word in a normalised address (0 when there is none).
    Survives component reordering ("Crossville, 994 Miller Ave" vs "994 MILLER AVENUE")."""
    import pyarrow as pa
    from normalize import hash_key
    pref = addr_key.str.extract(ADDR_PREFIX_RE, expand=False).fillna("")
    return hash_key(pa.array(pref, type=pa.string()))


def hard_negatives(sample_n: int = 50_000, name_cap: int = 50, lookalike_min_name: float = 80,
                   seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mine non-matching S2/S3 records that look like a sampled S1 entity:
      same_name     identical normalised name, same country
      same_address  identical normalised address (longer than 10 chars)
      look_alike    same house number + street word AND name token-set >= 80, different
                    normalised name ("Lower Procap LLC" vs "LOWER PROCAP LTD.")
    Key groups larger than `name_cap` are skipped (exploration only).
    Returns (stats per type, all mined negatives with their text)."""
    cols = ["uid", "entity_id", "country", "business_name", "business_address", "name_key", "addr_key", "name_h", "addr_h"]
    s1 = load("train", 1, cols).sample(sample_n, random_state=seed).reset_index(drop=True)
    s1["ap_h"] = addr_prefix_hash(s1.addr_key)
    oth = load_others("train", ["uid", "country", "name_h", "addr_h", "addr_key"])
    oth["ap_h"] = addr_prefix_hash(oth.pop("addr_key"))
    tr = load_truth(["s1_uid", "m_uid"])
    owner_of = tr.set_index("m_uid").s1_uid

    found, stats = [], []
    for kind, key in (("same_name", "name_h"), ("same_address", "addr_h"), ("look_alike", "ap_h")):
        left = s1[s1[key] != 0]
        if kind == "same_address":
            left = left[left.addr_key.str.len() > 10]
        size = oth.groupby(["country", key], observed=True).size()
        small = size[size <= name_cap].rename("grp").reset_index()
        left = left.merge(small, on=["country", key])  # drops keys above the cap
        cand = oth[oth[key].isin(left[key])][["uid", "country", key]]
        j = left[["uid", "country", key, "name_key"]].merge(cand, on=["country", key], suffixes=("", "_c"))
        j["owner"] = j.uid_c.map(owner_of)
        j["is_match"] = j.owner == j.uid
        if kind == "look_alike":  # keep pairs with similar but not identical names
            ck = load_others("train", ["uid", "name_key"], filters=[("uid", "in", j.uid_c.unique().tolist())])
            j["name_key_c"] = j.uid_c.map(ck.set_index("uid").name_key)
            j["name_sim"] = process.cpdist(j.name_key.tolist(), j.name_key_c.tolist(),
                                           scorer=fuzz.token_set_ratio, workers=-1)
            j = j[(j.name_sim >= lookalike_min_name) & (j.name_key != j.name_key_c)]
        stats.append({"type": kind, "sampled_s1": len(s1), "s1_with_key_group": left.uid.nunique(),
                      "joined_pairs": len(j), "true_share": j.is_match.mean(),
                      "negatives": int((~j.is_match).sum()),
                      "neg_unowned_distractor": int(j.owner[~j.is_match].isna().sum()),
                      "s1_with_>=1_negative": j.uid[~j.is_match].nunique() / len(s1)})
        found.append(j[~j.is_match].assign(type=kind)[["type", "uid", "uid_c", "owner"]])
    neg = pd.concat(found, ignore_index=True)

    ctext = load_others("train", ["uid", "business_name", "business_address"],
                        filters=[("uid", "in", neg.uid_c.unique().tolist())])
    neg = (neg.merge(s1[["uid", "entity_id", "country", "business_name", "business_address"]], on="uid")
              .merge(ctext.rename(columns=lambda c: c + "_c"), on="uid_c"))
    owners = load("train", 1, ["uid", "business_name", "business_address"],
                  filters=[("uid", "in", neg.owner.dropna().astype("int64").unique().tolist())])
    neg = neg.merge(owners.rename(columns={"uid": "owner", "business_name": "owner_name",
                                           "business_address": "owner_address"}), on="owner", how="left")
    return pd.DataFrame(stats).set_index("type"), neg


def pick_negative_examples(neg: pd.DataFrame, per_type=(7, 6, 7), seed: int = 5) -> pd.DataFrame:
    parts = [neg[neg.type == t].sample(min(k, (neg.type == t).sum()), random_state=seed)
             for t, k in zip(("same_name", "same_address", "look_alike"), per_type)]
    return pd.concat(parts, ignore_index=True)


def write_examples_md(groups: list[dict], negs: pd.DataFrame, path=NOTES_DIR / "eda_examples.md") -> None:
    def rec(r):
        return f"{r['business_name']} \\| {r['business_address'] or '*(empty)*'}"
    lines = ["# Stage 0 examples", "", "## 20 matched groups (train)", "",
             "Sampled with seed 7: 9 US and 9 India entities with matches, plus 2 singletons.", ""]
    for g in groups:
        s = g["s1"]
        lines.append(f"**{s['entity_id']}** [{s['country']}] {rec(s)}  — {len(g['matches'])} match(es)")
        lines += [f"- `{m['entity_id']}` {rec(m)}" for m in g["matches"]] or ["- *(singleton: no matches)*"]
        lines.append("")
    lines += ["## 20 hard negatives (train)", "",
              "Non-matching S2/S3 records that collide with a sampled S1 entity on name or address. "
              "`owner` is the S1 entity the record really belongs to (blank = matches no S1).", "",
              "| type | S1 entity | S1 name \\| address | negative | negative name \\| address | real owner name \\| address |",
              "|---|---|---|---|---|---|"]
    for _, r in negs.iterrows():
        owner = "*(none — distractor)*" if pd.isna(r.owner) else f"{r.owner_name} \\| {r.owner_address}"
        cid = f"S{int(r.uid_c // UID_BASE)}-{int(r.uid_c % UID_BASE)}"
        lines.append(f"| {r.type} | {r.entity_id} | {r.business_name} \\| {r.business_address} | {cid} | "
                     f"{r.business_name_c} \\| {r.business_address_c or '*(empty)*'} | {owner} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    with stage("eda_counts_labels"):
        print(source_counts().to_string())
        mc = s1_match_counts()
        print(label_summary(mc).round(4).to_string())
        print(match_count_distribution(mc).to_string())
        print(assignment_checks())
        del mc
    with stage("eda_noise"):
        print(noise_profile().round(3).to_string())
        print(summarise_similarity(true_pair_similarity()).round(3).to_string())
    with stage("eda_name_frequency"):
        for sp in ("train", "test"):
            print(sp, "\n", name_frequency(sp).round(4).to_string())
            print(top_names(sp).to_string())
        print(same_name_pressure().round(4).to_string())
    with stage("eda_candidate_budget"):
        print(candidate_budget().to_string())
        print(oracle_topk_completeness().round(4).to_string())
    with stage("eda_examples_hard_negatives"):
        groups = example_groups()
        st, neg = hard_negatives()
        print(st.round(4).to_string())
        write_examples_md(groups, pick_negative_examples(neg))
