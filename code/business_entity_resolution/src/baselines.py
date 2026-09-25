"""Rule baselines that need no training. They exercise the evaluation end to end and
set a floor for every later stage.

  empty            predict no match for anyone (score = singleton share)
  exact_name       every S2/S3 record in the same country with an identical basic_key name
  name+addr_prefix identical name AND the same "house number + street word" address prefix

Rules have no training step, so the country hold-out columns are simply the score on each
country's S1 entities. Later stages train on one country and score the other.

Usage:  python src/baselines.py [--full]    (--full also scores the 441k val split)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data import load, load_others, load_truth
from evaluate import candidate_stats, log_experiment, score_report
from memtrack import stage
from normalize import addr_prefix_hash
from splits import load_split_ids

RULES = ("empty", "exact_name", "name+addr_prefix")


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    df["ap_h"] = addr_prefix_hash(df.pop("addr_key"))
    return df


def rule_pairs(rule: str, s1_uids: np.ndarray, pool: pd.DataFrame) -> pd.DataFrame:
    """Predicted pairs (s1, m) as int64 uids for one rule, restricted to s1_uids."""
    if rule == "empty":
        return pd.DataFrame({"s1": pd.Series(dtype="int64"), "m": pd.Series(dtype="int64")})
    s1 = _keys(load("train", 1, ["uid", "country", "name_h", "addr_key"],
                    filters=[("uid", "in", s1_uids.tolist())]))
    on = ["country", "name_h"] + (["ap_h"] if rule == "name+addr_prefix" else [])
    left, right = s1[s1.name_h != 0], pool[pool.name_h != 0]
    if "ap_h" in on:
        left, right = left[left.ap_h != 0], right[right.ap_h != 0]
    j = left[["uid"] + on].merge(right[["uid"] + on], on=on, suffixes=("", "_m"))
    return pd.DataFrame({"s1": j.uid.to_numpy(), "m": j.uid_m.to_numpy()})


def main(full: bool = False) -> None:
    with stage("baselines"):
        pool = _keys(load_others("train", ["uid", "country", "name_h", "addr_key"]))
        truth = load_truth(["s1_uid", "m_uid"]).rename(columns={"s1_uid": "s1", "m_uid": "m"})
        country = load("train", 1, ["uid", "country"]).set_index("uid").country.astype(str)
        sets = ["dev_val"] + (["val"] if full else [])
        countries = sorted(country.unique())
        for rule in RULES:
            res = {}
            for name in sets + [f"dev_country={c}" for c in countries]:
                ids = load_split_ids(name)
                pred = rule_pairs(rule, ids, pool)
                rep = score_report(pred, truth, ids, groups=country)
                st = candidate_stats(pred, truth, ids)
                res[name] = (rep.loc["all", "macro_f05"], st)
                if name in sets:
                    print(f"\n== {rule} on {name}: macro F0.5 = {rep.loc['all', 'macro_f05']:.4f} | "
                          f"pairs/S1 {st['avg_cands']:.2f} | pair completeness {st['pair_completeness']:.4f}")
                    print(rep.round(4).to_string())
            for name in sets:
                f, st = res[name]
                log_experiment(run=f"B-{rule}", change=f"rule baseline: {rule}", **{
                    "eval set": name, "blocking recall": st["pair_completeness"], "avg cands": st["avg_cands"],
                    "val F0.5": f,
                    "hold-out US→India": res["dev_country=India"][0] if "India" in countries else None,
                    "hold-out India→US": res["dev_country=US"][0] if "US" in countries else None,
                    "notes": "no training: hold-out = score on that country's dev S1"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    main(ap.parse_args().full)
