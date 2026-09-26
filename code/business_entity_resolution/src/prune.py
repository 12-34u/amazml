"""Stage 3b — shrink the candidate set.

1. A cheap scorer: LightGBM on 12 fast pair features, trained out-of-fold on the dev-train S1
   entities (5 folds by S1 entity, from splits.parquet), then refit on all of dev-train to
   score the other sets.
2. One-to-one pressure: each S2/S3 record keeps only its top-2 S1 entities by score.
3. An adaptive cutoff per S1: keep candidate j when p_j >= floor AND p_j >= p_best - margin,
   at most max_k. An S1 whose best candidate is below the floor gets an empty list.
Certain matches are always kept (score 1).

`candidate_pairs.tsv` = certain matches + the pruned candidates. That is exactly what the
Stage 4 matcher scores.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from blocking import (BLOCKERS, explode_localities, load_region_compat, load_side, locality_overlap,
                      region_compatible)
from prepare import UID_BASE

FEATURE_COLS = ["cos", "name_tsr", "name_ratio", "legal_compat", "hs_eq", "house_eq", "street_eq",
                "loc_overlap", "region_compat", "n_blockers", "cos_rank", "cos_gap", "cand_source"] + \
               [f"r_{b}" for b in BLOCKERS]   # blocker ranks (0 = not proposed by that blocker)
FEAT_SIDE_COLS = ["uid", "name_core", "alias_core", "legal_form", "house_no", "street_word", "hs_h",
                  "localities", "region"]
LGB_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=100,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=42, num_threads=8, deterministic=True, force_row_wise=True)
N_ROUNDS = 200


def pair_features(cands: pd.DataFrame, split: str, country: str) -> pd.DataFrame:
    """Add the fast features to the candidates of one country (row order kept)."""
    from data import available_sources
    s1 = load_side(split, [1], country, uids=cands.s1.unique(), columns=FEAT_SIDE_COLS)
    pool = load_side(split, [s for s in available_sources(split) if s != 1], country,
                     uids=cands.m.unique(), columns=FEAT_SIDE_COLS)
    ai, bi = pd.Index(s1.uid).get_indexer(cands.s1), pd.Index(pool.uid).get_indexer(cands.m)
    col = lambda df, c, ix: df[c].to_numpy()[ix]
    out = cands.copy()
    a_core, b_core, b_alias = col(s1, "name_core", ai), col(pool, "name_core", bi), col(pool, "alias_core", bi)
    tsr = process.cpdist(list(a_core), list(b_core), scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    ha = b_alias != ""
    if ha.any():
        tsr[ha] = np.maximum(tsr[ha], process.cpdist(list(a_core[ha]), list(b_alias[ha]), scorer=fuzz.token_set_ratio, workers=-1))
    out["name_tsr"] = tsr
    out["name_ratio"] = process.cpdist(list(a_core), list(b_core), scorer=fuzz.ratio, workers=-1).astype(np.float32)
    la, lb = col(s1, "legal_form", ai), col(pool, "legal_form", bi)
    out["legal_compat"] = np.where((la == "") | (lb == ""), 0.5, (la == lb).astype(np.float32)).astype(np.float32)
    out["hs_eq"] = ((col(s1, "hs_h", ai) == col(pool, "hs_h", bi)) & (col(s1, "hs_h", ai) != 0)).astype(np.int8)
    for f, c in (("house_eq", "house_no"), ("street_eq", "street_word")):
        a, b = col(s1, c, ai), col(pool, c, bi)
        out[f] = np.where((a == "") | (b == ""), 0.5, (a == b).astype(np.float32)).astype(np.float32)
    out["loc_overlap"] = np.minimum(locality_overlap(cands, explode_localities(s1), explode_localities(pool)), 3).astype(np.int8)
    out["region_compat"] = region_compatible(col(s1, "region", ai), col(pool, "region", bi), country,
                                             load_region_compat()).astype(np.float32)
    # context inside the S1's candidate list
    out["cos_rank"] = out.groupby("s1").cos.rank(ascending=False, method="min").astype(np.float32)
    out["cos_gap"] = (out.groupby("s1").cos.transform("max") - out.cos).astype(np.float32)
    out["cand_source"] = source_of(out.m.to_numpy())
    return out


def label(cands: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    return cands[["s1", "m"]].merge(truth[["s1", "m"]].assign(y=1), how="left").y.fillna(0).to_numpy(np.int8)


def train_oof(X: np.ndarray, y: np.ndarray, fold: np.ndarray) -> tuple[np.ndarray, lgb.Booster]:
    """Out-of-fold scores plus a model refit on all rows. X is binned ONCE into a LightGBM
    Dataset; each fold trains on a subset of it (no extra copies of the feature matrix).
    `fold` is per row and group-aware: all candidates of an S1 share its fold."""
    from memtrack import phase
    phase("oof: bin dataset")
    full = lgb.Dataset(X, y, params={"verbose": -1}, free_raw_data=False).construct()
    oof = np.zeros(len(y), np.float32)
    for k in np.unique(fold):
        tr = np.nonzero(fold != k)[0]
        te = fold == k
        phase(f"oof: fold {k}")
        m = lgb.train(LGB_PARAMS, full.subset(tr).construct(), N_ROUNDS)
        oof[te] = m.predict(X[te])
    phase("oof: refit")
    model = lgb.train(LGB_PARAMS, full, N_ROUNDS)
    return oof, model


def _group_rank(key: np.ndarray, order: np.ndarray) -> np.ndarray:
    """0-based rank of each element within its `key` group, following `order`."""
    k = key[order]
    start = np.ones(len(k), bool)
    start[1:] = k[1:] != k[:-1]
    idx = np.arange(len(k))
    rank_sorted = idx - np.maximum.accumulate(np.where(start, idx, 0))
    rank = np.empty(len(k), np.int64)
    rank[order] = rank_sorted
    return rank


def record_top2(c: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Each S2/S3 record keeps only its k best S1 entities by score (one-to-one pressure)."""
    s1, m, p = c.s1.to_numpy(), c.m.to_numpy(), c.p.to_numpy()
    order = np.lexsort((s1, -p, m))                  # by record, best score first, S1 uid tie-break
    return c[_group_rank(m, order) < k]


def adaptive_cutoff(c: pd.DataFrame, floor: float, margin: float, max_k: int) -> pd.DataFrame:
    """Keep p >= floor AND p >= best - margin, at most max_k per S1 (empty list allowed)."""
    s1, m, p = c.s1.to_numpy(), c.m.to_numpy(), c.p.to_numpy()
    order = np.lexsort((m, -p, s1))
    rank = _group_rank(s1, order)
    best = pd.Series(p).groupby(s1).transform("max").to_numpy()
    return c[(p >= floor) & (p >= best - margin) & (rank < max_k)]


def prune(c: pd.DataFrame, certain: pd.DataFrame, floor: float, margin: float, max_k: int, top_records: int = 2) -> pd.DataFrame:
    """Certain pairs plus the pruned candidates (columns s1, m, p)."""
    kept = adaptive_cutoff(record_top2(c[["s1", "m", "p"]], top_records), floor, margin, max_k)
    return pd.concat([certain[["s1", "m"]].assign(p=np.float32(1.0)), kept], ignore_index=True)


def f05_upper_bound(kept: pd.DataFrame, truth: pd.DataFrame, s1_uids) -> float:
    """Macro F0.5 of a perfect matcher restricted to these candidates: it predicts exactly the
    true pairs among them (precision 1), so singletons score 1 and others 1.25R/(0.25+R)."""
    t = truth[truth.s1.isin(s1_uids)]
    n_true = t.s1.value_counts()
    hit = t.merge(kept[["s1", "m"]]).s1.value_counts()
    r = (hit.reindex(n_true.index, fill_value=0) / n_true).to_numpy()
    f = np.where(r > 0, 1.25 * r / (0.25 + r), 0.0)
    n_single = len(pd.unique(np.asarray(s1_uids))) - len(n_true)
    return float((f.sum() + n_single) / (len(f) + n_single))


def source_of(uids: np.ndarray) -> np.ndarray:
    return (uids // UID_BASE).astype(np.int8)


# ================================================================ driver
S1_CHUNK = 15_000   # ~1M pairs per chunk: rapidfuzz needs Python string lists, keep them small


def featurize(cands: pd.DataFrame, split: str):
    """Yield feature frames per country and S1 chunk (bounded memory)."""
    from memtrack import check_budget, phase
    for c, g in cands.groupby("country", sort=True):
        phase(f"featurize {c}")
        s1s = g.s1.unique()
        for i in range(0, len(s1s), S1_CHUNK):
            part = g[g.s1.isin(s1s[i:i + S1_CHUNK])]
            yield pair_features(part, split, c)
            check_budget(where="featurize")


def score(cands: pd.DataFrame, split: str, model: lgb.Booster) -> pd.DataFrame:
    """(s1, m, p) for every candidate."""
    out = []
    for f in featurize(cands, split):
        out.append(pd.DataFrame({"s1": f.s1.to_numpy(), "m": f.m.to_numpy(),
                                 "p": model.predict(f[FEATURE_COLS].to_numpy(np.float32)).astype(np.float32)}))
    return pd.concat(out, ignore_index=True)


def score_file(path, split: str, model: lgb.Booster, s1_filter=None, min_p: float | None = None) -> pd.DataFrame:
    """Score a blocking output file one row group (= one S1 batch of one country) at a time.
    `min_p`: drop pairs below it while streaming. Exact for pruning with floor >= min_p: the
    cutoff removes them anyway, and a record's best two S1 above the floor are the same with
    or without them (keeps test scoring at ~5 pairs per S1 instead of ~70)."""
    import pyarrow.parquet as pq
    from blocking import candidate_files
    out = []
    for f in candidate_files(path):
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            rg = pf.read_row_group(i).to_pandas()
            if s1_filter is not None:
                rg = rg[rg.s1.isin(s1_filter)]
            if len(rg):
                sc = score(rg, split, model)
                out.append(sc[sc.p >= min_p] if min_p is not None else sc)
            del rg
    return pd.concat(out, ignore_index=True)


def train_pruner(cands: pd.DataFrame, split: str, truth: pd.DataFrame, folds: pd.Series):
    """Features + labels for the training S1 entities, OOF scores and the refit model.
    Only the feature matrix (float32) and ids are kept from each chunk."""
    Xs, ids = [], []
    for f in featurize(cands, split):
        Xs.append(f[FEATURE_COLS].to_numpy(np.float32))
        ids.append(f[["s1", "m"]].copy())
        del f
    X = np.concatenate(Xs)
    del Xs
    ids = pd.concat(ids, ignore_index=True)
    ids["y"] = label(ids, truth)
    oof, model = train_oof(X, ids.y.to_numpy(), ids.s1.map(folds).to_numpy())
    return ids.assign(p=oof), model


def tradeoff(scored: pd.DataFrame, certain: pd.DataFrame, truth: pd.DataFrame, s1_uids,
             floors=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2), margins=(0.3, 0.5, 0.7, 0.9, 1.0),
             max_ks=(4, 6, 8)) -> pd.DataFrame:
    """Average/median/p90 candidates, pair completeness and the F0.5 upper bound for a grid
    of cutoffs. Labels are joined once; each grid point is pure array arithmetic."""
    s1_uids = pd.unique(np.asarray(s1_uids))
    t = truth[truth.s1.isin(s1_uids)]
    n_true = t.s1.value_counts().reindex(s1_uids, fill_value=0).to_numpy()
    base = record_top2(scored[["s1", "m", "p"]]).reset_index(drop=True)
    base["y"] = label(base, t)
    cert = certain[certain.s1.isin(s1_uids)]
    cert_hit = cert[["s1", "m"]].merge(t[["s1", "m"]]).s1.value_counts().reindex(s1_uids, fill_value=0).to_numpy()
    cert_n = cert.s1.value_counts().reindex(s1_uids, fill_value=0).to_numpy()
    s1_code = pd.Index(s1_uids).get_indexer(base.s1)
    s1v, mv, pv, yv = base.s1.to_numpy(), base.m.to_numpy(), base.p.to_numpy(), base.y.to_numpy()
    order = np.lexsort((mv, -pv, s1v))
    rank = _group_rank(s1v, order)
    best = pd.Series(pv).groupby(s1v).transform("max").to_numpy()
    n = len(s1_uids)
    rows = []
    for mk in max_ks:
        for fl in floors:
            for mg in margins:
                keep = (pv >= fl) & (pv >= best - mg) & (rank < mk)
                cnt = np.bincount(s1_code[keep], minlength=n) + cert_n
                hit = np.bincount(s1_code[keep], weights=yv[keep], minlength=n) + cert_hit
                r = np.divide(hit, n_true, out=np.zeros(n), where=n_true > 0)
                f = np.where(n_true == 0, 1.0, np.where(r > 0, 1.25 * r / (0.25 + r), 0.0))
                rows.append({"max_k": mk, "floor": fl, "margin": mg, "avg_cands": cnt.mean(),
                             "median_cands": float(np.median(cnt)), "p90_cands": float(np.quantile(cnt, 0.9)),
                             "empty_lists": float((cnt == 0).mean()), "pair_completeness": hit.sum() / n_true.sum(),
                             "f05_upper_bound": f.mean()})
    return pd.DataFrame(rows)
