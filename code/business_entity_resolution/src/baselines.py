"""Rule baselines that need no training. They exercise the evaluation end to end and
set a floor for every later stage.

  empty            predict no match for anyone (score = singleton share)
  exact_name       every S2/S3 record in the same country with an identical basic_key name
  name+addr_prefix identical name AND the same "house number + street word" address prefix
  certain          Stage 2 "certain match": identical normalised name (name_norm, legal form kept)
                   AND the same canonical house number + first street word (hs_key)

Rules have no training step, so the country hold-out columns are simply the score on each
country's S1 entities. Later stages train on one country and score the other.

Every evaluation runs twice: on the full pool and on the test-like pool (splits.pool_mask).

Usage:  python src/baselines.py [--full]    (--full also scores the 441k val split)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data import load, load_others, load_truth
from evaluate import candidate_stats, log_experiment, score_report
from memtrack import stage
import pyarrow as pa

from normalize import addr_prefix_hash, hash_key
from splits import POOL_MODES, load_split_ids, pool_mask

RULES = ("empty", "exact_name", "name+addr_prefix", "certain")


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    df["ap_h"] = addr_prefix_hash(df.pop("addr_key"))
    return df


def _norm_keys(uids: np.ndarray | None, sources=(2, 3)) -> pd.DataFrame:
    """uid, nn_h (hash of name_norm), hs_h from the Stage 2 norm files."""
    from normalize_build import load_norm
    f = None if uids is None else [("uid", "in", uids.tolist())]
    parts = []
    for s in sources:
        d = load_norm("train", s, ["uid", "name_norm", "hs_h"], f)
        d["nn_h"] = hash_key(pa.array(d.pop("name_norm"), pa.string()))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def rule_pairs(rule: str, s1_uids: np.ndarray, pool: pd.DataFrame) -> pd.DataFrame:
    """Predicted pairs (s1, m) as int64 uids for one rule, restricted to s1_uids."""
    if rule == "empty":
        return pd.DataFrame({"s1": pd.Series(dtype="int64"), "m": pd.Series(dtype="int64")})
    if rule == "certain":
        s1 = load("train", 1, ["uid", "country"], filters=[("uid", "in", s1_uids.tolist())]).merge(
            _norm_keys(s1_uids, sources=(1,)), on="uid")
        on = ["country", "nn_h", "hs_h"]
        left, right = s1[(s1.nn_h != 0) & (s1.hs_h != 0)], pool[(pool.nn_h != 0) & (pool.hs_h != 0)]
        j = left[["uid"] + on].merge(right[["uid"] + on], on=on, suffixes=("", "_m"))
        return pd.DataFrame({"s1": j.uid.to_numpy(), "m": j.uid_m.to_numpy()})
    s1 = _keys(load("train", 1, ["uid", "country", "name_h", "addr_key"],
                    filters=[("uid", "in", s1_uids.tolist())]))
    on = ["country", "name_h"] + (["ap_h"] if rule == "name+addr_prefix" else [])
    left, right = s1[s1.name_h != 0], pool[pool.name_h != 0]
    if "ap_h" in on:
        left, right = left[left.ap_h != 0], right[right.ap_h != 0]
    j = left[["uid"] + on].merge(right[["uid"] + on], on=on, suffixes=("", "_m"))
    return pd.DataFrame({"s1": j.uid.to_numpy(), "m": j.uid_m.to_numpy()})


def main(full: bool = False, rules=RULES) -> None:
    with stage("baselines"):
        pool = _keys(load_others("train", ["uid", "country", "name_h", "addr_key"]))
        if "certain" in rules:
            pool = pool.merge(_norm_keys(None), on="uid", how="left").fillna({"nn_h": 0, "hs_h": 0})
        truth = load_truth(["s1_uid", "m_uid"]).rename(columns={"s1_uid": "s1", "m_uid": "m"})
        country = load("train", 1, ["uid", "country"]).set_index("uid").country.astype(str)
        sets = ["dev_val"] + (["val"] if full else [])
        countries = sorted(country.unique())
        for mode in POOL_MODES:
            for rule in rules:
                res = {}
                for name in sets + [f"dev_country={c}" for c in countries]:
                    ids = load_split_ids(name)
                    sub = pool[pool_mask(pool.uid.to_numpy(), ids, mode)]
                    pred = rule_pairs(rule, ids, sub)
                    rep = score_report(pred, truth, ids, groups=country)
                    st = candidate_stats(pred, truth, ids)
                    res[name] = (rep.loc["all", "macro_f05"], st)
                    if name in sets:
                        print(f"== [{mode}] {rule} on {name}: macro F0.5 = {rep.loc['all', 'macro_f05']:.4f} | "
                              f"pairs/S1 {st['avg_cands']:.2f} | pair completeness {st['pair_completeness']:.4f} | "
                              f"singletons {rep.loc['singletons', 'macro_f05']:.4f} | "
                              f"mean precision {rep.loc['all', 'mean_precision']:.4f} | pool {len(sub):,}")
                for name in sets:
                    f, st = res[name]
                    log_experiment(run=f"B-{rule}", change=f"rule baseline: {rule}", **{
                        "eval set": f"{name} [{mode} pool]", "blocking recall": st["pair_completeness"],
                        "avg cands": st["avg_cands"], "val F0.5": f,
                        "hold-out US→India": res["dev_country=India"][0] if "India" in countries else None,
                        "hold-out India→US": res["dev_country=US"][0] if "US" in countries else None,
                        "notes": "no training: hold-out = score on that country's dev S1"})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--rules", nargs="+", default=list(RULES), choices=RULES)
    a = ap.parse_args()
    main(a.full, tuple(a.rules))
