"""Reproducible subsets and validation splits of the training data.

All splits are by **S1 entity**. Every S2/S3 record belongs to at most one S1 entity
(checked in Stage 0), so a split by S1 keeps each entity's true candidates on the same
side. S2/S3 are never subsampled: every evaluation searches the full pool of its
countries, as the test set will.

artifacts/splits.parquet, one row per train S1 entity:
  uid, country, n_matches
  split   'train' (80%) or 'val' (20%), stratified by country x match count (0..5, 6+)
  fold    0..4 for group-aware CV inside the train part, -1 for val rows
  in_dev  True for the ~150k dev-subset entities

Dev subset: 150k S1 entities, stratified by country, seed 42. Iterate on it and run the
full data only for final checks. It inherits split/fold from the table above, so
dev_val = dev ∩ val. Records owned by S1 entities outside the evaluated set act as extra
distractors, which makes dev and val precision slightly pessimistic.

Country hold-out (the stand-in for unseen France): train on one country's S1 entities,
validate on the other's.

Usage:  python src/splits.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import ARTIFACTS_DIR, load, load_truth
from memtrack import stage

DEV_SIZE = 150_000
SEED = 42
VAL_FRAC = 0.2
N_FOLDS = 5
SPLITS_PATH = ARTIFACTS_DIR / "splits.parquet"


def make_dev_subset(n: int = DEV_SIZE, seed: int = SEED) -> pd.DataFrame:
    s1 = load("train", 1, ["uid", "entity_id", "country"])
    frac = n / len(s1)
    dev = (s1.groupby("country", observed=True, group_keys=False)
             .sample(frac=frac, random_state=seed)
             .sort_values("uid", ignore_index=True))
    truth = load_truth()
    dev_truth = truth[truth.s1_uid.isin(dev.uid)].reset_index(drop=True)
    dev.to_parquet(ARTIFACTS_DIR / "dev_s1.parquet", index=False)
    dev_truth.to_parquet(ARTIFACTS_DIR / "dev_truth.parquet", index=False)
    print(f"  dev S1: {len(dev):,} ({dev.country.value_counts().to_dict()}), true pairs: {len(dev_truth):,}")
    return dev


def make_splits(val_frac: float = VAL_FRAC, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Stratified 80/20 split plus CV folds. Deterministic for a given seed."""
    s1 = load("train", 1, ["uid", "country"]).sort_values("uid", ignore_index=True)
    counts = load_truth(["s1_uid"]).s1_uid.value_counts()
    s1["n_matches"] = s1.uid.map(counts).fillna(0).astype("int16")
    strat = s1.country.astype(str) + "_" + s1.n_matches.clip(upper=6).astype(str)
    rnd = pd.Series(np.random.default_rng(seed).random(len(s1)))
    # Within each stratum, the first round(size * val_frac) entities in random order go to val.
    pos = rnd.groupby(strat).rank(method="first")
    size = strat.map(strat.value_counts())
    s1["split"] = np.where(pos <= (size * val_frac).round(), "val", "train")
    # CV folds for the train part: round-robin over the same random order within each stratum.
    tr = s1.split == "train"
    fold_pos = rnd[tr].groupby(strat[tr]).rank(method="first").astype(int) - 1
    s1["fold"] = np.int8(-1)
    s1.loc[tr, "fold"] = (fold_pos % n_folds).astype("int8")
    dev = pd.read_parquet(ARTIFACTS_DIR / "dev_s1.parquet", columns=["uid"])
    s1["in_dev"] = s1.uid.isin(dev.uid)
    s1.to_parquet(SPLITS_PATH, index=False)
    return s1


def load_splits() -> pd.DataFrame:
    return pd.read_parquet(SPLITS_PATH)


def country_holdout(train_country: str, dev: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(train uids, validation uids) for training on `train_country` and validating on all
    other countries. The country list comes from the data, never hard-coded."""
    sp = load_splits()
    if dev:
        sp = sp[sp.in_dev]
    is_tr = (sp.country == train_country).to_numpy()
    return sp.uid.to_numpy()[is_tr], sp.uid.to_numpy()[~is_tr]


def load_split_ids(name: str) -> np.ndarray:
    """S1 uids for a named evaluation set:
    all_train | train | val | dev | dev_train | dev_val |
    country=<C> | dev_country=<C>   (all train S1 entities of country C, optionally within dev)"""
    sp = load_splits()
    if name.startswith(("country=", "dev_country=")):
        prefix, c = name.split("=", 1)
        m = sp.country == c
        if prefix == "dev_country":
            m &= sp.in_dev
        return sp.uid[m].to_numpy()
    masks = {
        "all_train": np.ones(len(sp), bool),
        "train": sp.split == "train", "val": sp.split == "val",
        "dev": sp.in_dev, "dev_train": sp.in_dev & (sp.split == "train"),
        "dev_val": sp.in_dev & (sp.split == "val"),
    }
    return sp.uid[masks[name]].to_numpy()


POOL_MODES = ("full", "testlike")


def pool_mask(pool_uids: np.ndarray, eval_s1_uids: np.ndarray, mode: str) -> np.ndarray:
    """Which S2/S3 records to keep in the candidate pool when evaluating `eval_s1_uids`.

    full      every record (the pool the rest of train leaves behind: records owned by
              non-evaluated S1 entities stay as extra, unclaimable look-alikes)
    testlike  unmatched distractors, plus records whose true S1 is in the evaluated set.
              Records owned by S1 entities outside the evaluated set (e.g. the train split)
              are dropped. In the test set every S1 is scored, so those records would
              have a competing owner rather than being free decoys.
    """
    if mode == "full":
        return np.ones(len(pool_uids), dtype=bool)
    if mode != "testlike":
        raise ValueError(f"unknown pool mode {mode!r}")
    truth = load_truth(["s1_uid", "m_uid"])
    owned_elsewhere = truth.m_uid[~truth.s1_uid.isin(eval_s1_uids)]
    return ~np.isin(pool_uids, owned_elsewhere.to_numpy())


def split_summary() -> pd.DataFrame:
    sp = load_splits()
    rows = []
    for name in ("all_train", "train", "val", "dev", "dev_train", "dev_val"):
        d = sp[sp.uid.isin(load_split_ids(name))]
        rows.append({"set": name, "S1": len(d), **d.country.value_counts().to_dict(),
                     "singleton_share": (d.n_matches == 0).mean(), "mean_matches": d.n_matches.mean(),
                     "true_pairs": int(d.n_matches.sum())})
    return pd.DataFrame(rows).set_index("set")


if __name__ == "__main__":
    with stage("splits"):
        if not (ARTIFACTS_DIR / "dev_s1.parquet").exists():
            make_dev_subset()
        make_splits()
        print(split_summary().round(4).to_string())
