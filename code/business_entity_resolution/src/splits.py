"""Reproducible subsets and validation splits of the training data.

Dev subset: about 150k S1 entities, stratified by country with a fixed seed, plus their
true pairs. S2 and S3 are NOT subsampled. The dev subset keeps every S2/S3 record of the
same countries, so blocking and matching face the full distractor pool. Records owned by
S1 entities outside the subset therefore act as extra distractors, which makes dev
precision slightly pessimistic compared with the full set.

Usage:  python src/splits.py
"""
from __future__ import annotations

import pandas as pd

from data import ARTIFACTS_DIR, load, load_truth
from memtrack import stage

DEV_SIZE = 150_000
SEED = 42


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


def load_dev() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(ARTIFACTS_DIR / "dev_s1.parquet"),
            pd.read_parquet(ARTIFACTS_DIR / "dev_truth.parquet"))


if __name__ == "__main__":
    with stage("dev_subset"):
        make_dev_subset()
