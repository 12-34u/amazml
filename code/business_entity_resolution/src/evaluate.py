"""Macro F0.5 exactly as the challenge defines it, plus candidate-set (blocking) metrics.

Scoring rules, per S1 entity in the evaluation set:
  * no true matches (singleton): 1.0 if the prediction is empty, else 0.0
  * true matches but empty prediction, or no correct prediction: 0.0
  * otherwise F0.5 = 1.25*P*R / (0.25*P + R)
The final score is the unweighted mean over ALL S1 entities in the evaluation set. An S1
entity with no prediction row counts as an empty prediction.

Predictions and truth are long pair tables with columns (s1, m), one row per pair. Any
hashable ids work (entity_id strings or the int64 uids). Duplicate pairs are ignored.

CLI (score a submission-format file against the train labels):
  python src/evaluate.py --pred path/to/matching_results.tsv [--s1-subset val|dev_val|...]
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from data import NOTES_DIR, read_tsv_arrow

BETA2 = 0.25  # beta = 0.5


# ---------------------------------------------------------------- core metric
def f05(pred: set | list, truth: set | list) -> float:
    """Reference single-entity implementation (used by the unit tests)."""
    pred, truth = set(pred), set(truth)
    if not truth:
        return 1.0 if not pred else 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def per_entity_scores(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids) -> pd.DataFrame:
    """Vectorised per-entity precision, recall and F0.5.

    pred, truth: DataFrames with columns s1, m. s1_ids: every S1 entity in the evaluation
    set. Pairs whose s1 is outside s1_ids are ignored."""
    s1_ids = pd.Index(pd.unique(np.asarray(s1_ids)), name="s1")
    pred = pred[["s1", "m"]].drop_duplicates()
    truth = truth[["s1", "m"]].drop_duplicates()
    pred, truth = pred[pred.s1.isin(s1_ids)], truth[truth.s1.isin(s1_ids)]
    tp = pred.merge(truth, on=["s1", "m"]).s1.value_counts()
    out = pd.DataFrame(index=s1_ids)
    out["n_pred"] = pred.s1.value_counts().reindex(s1_ids, fill_value=0)
    out["n_true"] = truth.s1.value_counts().reindex(s1_ids, fill_value=0)
    out["tp"] = tp.reindex(s1_ids, fill_value=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(out.n_pred > 0, out.tp / out.n_pred, 0.0)
        r = np.where(out.n_true > 0, out.tp / out.n_true, 0.0)
        f = np.where(out.tp > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0)
    singleton = out.n_true.to_numpy() == 0
    f = np.where(singleton, (out.n_pred.to_numpy() == 0).astype(float), f)
    out["precision"], out["recall"], out["f05"] = p, r, f
    out["singleton"] = singleton
    return out


def macro_f05(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids) -> float:
    return float(per_entity_scores(pred, truth, s1_ids).f05.mean())


def score_report(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids, groups: pd.Series | None = None) -> pd.DataFrame:
    """Macro F0.5 with a breakdown: overall, singletons vs non-singletons, and optionally
    by a grouping Series indexed by s1 (e.g. country)."""
    e = per_entity_scores(pred, truth, s1_ids)
    parts = {"all": e, "singletons": e[e.singleton], "non-singletons": e[~e.singleton]}
    if groups is not None:
        g = groups.reindex(e.index)
        parts.update({f"country={k}": e[g == k] for k in pd.unique(g.dropna())})
    rows = {}
    for k, d in parts.items():
        rows[k] = {"entities": len(d), "macro_f05": d.f05.mean(),
                   "mean_precision": d.precision[d.n_pred > 0].mean() if (d.n_pred > 0).any() else np.nan,
                   "mean_recall": d.recall[~d.singleton].mean() if (~d.singleton).any() else np.nan,
                   "pred_pairs": int(d.n_pred.sum()), "true_pairs": int(d.n_true.sum()), "tp_pairs": int(d.tp.sum())}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- candidate-set metrics
def candidate_stats(cands: pd.DataFrame, truth: pd.DataFrame, s1_ids,
                    pool_size_by_s1: pd.Series | None = None) -> dict:
    """Blocking / pruning quality for a candidate table (columns s1, m).

    pair_completeness  share of true pairs that are in the candidate set
    avg/median         candidates per S1 over ALL evaluated S1 (empty lists count as 0)
    reduction_ratio    1 - |candidates| / sum over S1 of |S2+S3 in the same country|,
                       when pool_size_by_s1 (indexed by s1) is given
    """
    s1_ids = pd.Index(pd.unique(np.asarray(s1_ids)))
    cands = cands[["s1", "m"]].drop_duplicates()
    cands = cands[cands.s1.isin(s1_ids)]
    truth = truth[truth.s1.isin(s1_ids)][["s1", "m"]].drop_duplicates()
    per = cands.s1.value_counts().reindex(s1_ids, fill_value=0)
    kept = len(truth.merge(cands, on=["s1", "m"]))
    out = {"s1_entities": len(s1_ids), "candidate_pairs": len(cands),
           "avg_cands": per.mean(), "median_cands": float(per.median()),
           "p90_cands": float(per.quantile(0.9)), "empty_lists": float((per == 0).mean()),
           "pair_completeness": kept / len(truth) if len(truth) else np.nan,
           "entity_full_recall": _entity_full_recall(cands, truth)}
    if pool_size_by_s1 is not None:
        denom = float(pool_size_by_s1.reindex(s1_ids).sum())
        out["reduction_ratio"] = 1 - len(cands) / denom
    return out


def _entity_full_recall(cands: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Share of non-singleton S1 entities whose every true match is a candidate."""
    if truth.empty:
        return np.nan
    hit = truth.merge(cands.assign(_hit=1), on=["s1", "m"], how="left")._hit.fillna(0)
    return float(hit.groupby(truth.s1.to_numpy()).min().mean())


# ---------------------------------------------------------------- file I/O
def read_id_list_tsv(path) -> pd.DataFrame:
    """Read a submission-format TSV (s1 <tab> comma-separated ids) into long pairs, plus
    the list of S1 ids that had a row."""
    import pyarrow.compute as pc
    t = read_tsv_arrow(path)
    s1col, lcol = t.column_names[:2]
    lists = pc.split_pattern(t[lcol], ",")
    flat = pc.utf8_trim_whitespace(pc.list_flatten(lists))
    s1 = pc.take(t[s1col], pc.list_parent_indices(lists))
    keep = pc.not_equal(flat, "")
    pairs = pd.DataFrame({"s1": pc.filter(s1, keep).to_numpy(zero_copy_only=False),
                          "m": pc.filter(flat, keep).to_numpy(zero_copy_only=False)})
    pairs.attrs["s1_rows"] = t[s1col].to_pylist()   # a list, not ndarray: pandas compares attrs on merge
    return pairs


def write_id_list_tsv(pairs: pd.DataFrame, s1_ids, path, list_col: str = "matched_entity_ids") -> None:
    """Write long pairs (string ids) as a submission TSV: one row per S1 in s1_ids, in that
    order, with an empty list where there are no pairs."""
    lists = pairs.groupby("s1", sort=False).m.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": list(s1_ids)})
    out[list_col] = out.source1_entity_id.map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")


# ---------------------------------------------------------------- experiment log
EXP_LOG = NOTES_DIR / "experiments.md"
EXP_COLS = ["date", "run", "change", "eval set", "blocking recall", "avg cands", "val F0.5",
            "hold-out US→India", "hold-out India→US", "notes"]


def log_experiment(**row) -> None:
    """Append one row to notes/experiments.md (the header is created on first use)."""
    row.setdefault("date", f"{dt.date.today():%Y-%m-%d}")
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else ("–" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    if not EXP_LOG.exists():
        EXP_LOG.parent.mkdir(parents=True, exist_ok=True)
        EXP_LOG.write_text("# Experiment log\n\nOne row per run. F0.5 is macro F0.5 over every S1 entity in the eval set "
                           "(singletons included). Hold-out columns: train on one country, validate on the other.\n\n"
                           "| " + " | ".join(EXP_COLS) + " |\n|" + "---|" * len(EXP_COLS) + "\n")
    with open(EXP_LOG, "a") as f:
        f.write("| " + " | ".join(fmt(row.get(c)) for c in EXP_COLS) + " |\n")


# ---------------------------------------------------------------- CLI
def main() -> None:
    from data import load, load_truth
    from splits import load_split_ids
    ap = argparse.ArgumentParser(description="Score a matching_results-style TSV against the train labels.")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--s1-subset", default="all_train",
                    help="all_train | val | train | dev | dev_val | dev_train | holdout_US | holdout_India")
    args = ap.parse_args()
    s1_ids = load_split_ids(args.s1_subset)
    s1 = load("train", 1, ["uid", "entity_id", "country"]).set_index("uid")
    ids = s1.loc[s1_ids, "entity_id"]
    truth = load_truth(["s1_id", "matched_id"]).rename(columns={"s1_id": "s1", "matched_id": "m"})
    pred = read_id_list_tsv(args.pred)
    missing = set(ids) - set(pred.attrs["s1_rows"])
    if missing:
        print(f"note: {len(missing):,} S1 entities have no row in the file; scored as empty predictions")
    country = pd.Series(s1.loc[s1_ids, "country"].astype(str).values, index=ids.values)
    print(score_report(pred, truth, ids.values, groups=country).round(4).to_string())


if __name__ == "__main__":
    main()
