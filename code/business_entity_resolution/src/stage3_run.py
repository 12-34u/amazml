"""Stage 3/3b runner (validation): block the val split, report blockers, score with the
dev-trained pruner, and build the trade-off table. Test-set run: see submit_baseline.py.

Usage:  python src/stage3_run.py block      # blocking for val -> artifacts/cands_val.parquet
        python src/stage3_run.py train      # OOF pruner on dev-train -> artifacts/pruner.txt (+ OOF scores)
        python src/stage3_run.py prune      # score val, trade-off table, pruned candidates for val and dev-train
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from blocking import BLOCKERS, BlockConfig, blocking_report, run_blocking
from data import ARTIFACTS_DIR, NOTES_DIR, load, load_truth
from memtrack import PeakSampler, stage
from splits import load_split_ids


def truth_long() -> pd.DataFrame:
    return load_truth(["s1_uid", "m_uid"]).rename(columns={"s1_uid": "s1", "m_uid": "m"})


def pool_size_by_s1(s1_uids: np.ndarray) -> pd.Series:
    """|S2+S3 in the same country| for each S1: the reduction-ratio denominator."""
    from data import load_others
    n = load_others("train", ["country"]).country.astype(str).value_counts()
    c = load("train", 1, ["uid", "country"]).set_index("uid").country.astype(str).reindex(s1_uids)
    return c.map(n).astype(float)


def block_val() -> None:
    ids = load_split_ids("val")
    with stage("stage3_block_val"), PeakSampler() as ps:
        path, certs = run_blocking("train", ids, "val", BlockConfig(), resume="--resume" in sys.argv)
    print("PEAKS", ps.report(), flush=True)
    report_val_blocking()


def report_val_blocking() -> None:
    """Per-country blocker report (one country's candidates in memory at a time); the 'all'
    row is combined from per-country counts, so memory stays flat."""
    from blocking import candidate_files
    ids = load_split_ids("val")
    certs = pd.read_parquet(ARTIFACTS_DIR / "certain_val.parquet")
    truth = truth_long()
    truth = truth[truth.s1.isin(ids)]
    s1c = load("train", 1, ["uid", "country"]).set_index("uid").country.astype(str)
    reps, stats = [], []
    with stage("stage3_report_val"):
        for f in candidate_files(ARTIFACTS_DIR / "cands_val"):
            cands = pd.read_parquet(f, columns=["s1", "m", "country"] + [f"b_{b}" for b in BLOCKERS])
            c = cands.country.iloc[0]
            ids_c = ids[s1c.reindex(ids).to_numpy() == c]
            rep, st = blocking_report(cands, certs[certs.country == c], truth, ids_c,
                                      pool_size_by_s1=pool_size_by_s1(ids_c))
            n_true = int(truth.s1.isin(ids_c).sum())
            reps.append(rep.assign(country=c, n_true=n_true, n_s1=len(ids_c)))
            stats.append({"set": c, "n_true": n_true, **st})
            del cands
    reps, stats = pd.concat(reps), pd.DataFrame(stats)
    w = lambda col, by: (reps[col] * reps[by]).groupby(level=0).sum() / reps.groupby(level=0)[by].sum()
    all_rep = pd.DataFrame({"pairs": reps.groupby(level=0).pairs.sum(),
                            "pairs_per_S1": reps.groupby(level=0).pairs.sum() / len(ids),
                            "recall": w("recall", "n_true"), "unique_recall": w("unique_recall", "n_true")})
    tot = stats.n_true.sum()
    all_st = {"set": "all", "n_true": tot, "s1_entities": stats.s1_entities.sum(), "candidate_pairs": stats.candidate_pairs.sum(),
              "avg_cands": stats.candidate_pairs.sum() / stats.s1_entities.sum(),
              "pair_completeness": (stats.pair_completeness * stats.n_true).sum() / tot,
              "entity_full_recall": (stats.entity_full_recall * stats.n_true).sum() / tot,
              "reduction_ratio": 1 - stats.candidate_pairs.sum() / (stats.candidate_pairs / (1 - stats.reduction_ratio)).sum()}
    stats = pd.concat([pd.DataFrame([all_st]), stats], ignore_index=True)
    pd.concat([all_rep.assign(country="all"), reps.drop(columns=["n_true", "n_s1"])]).to_csv(ARTIFACTS_DIR / "blocking_report_val.csv")
    stats.to_csv(ARTIFACTS_DIR / "blocking_union_val.csv", index=False)
    pd.set_option("display.width", 220)
    print(all_rep.round(4).to_string()); print(stats.round(7).to_string(index=False), flush=True)


# ---------------------------------------------------------------- pruner
# Recommended operating point (chosen on val: ~3.5-5 candidates, completeness >= 95%); see notes/blocking.md
OPERATING_POINT = dict(floor=0.05, margin=1.0, max_k=8)


def train_pruner_dev() -> None:
    import lightgbm as lgb  # noqa: F401
    from prune import FEATURE_COLS, train_pruner
    from splits import load_splits
    sp = load_splits()
    folds = sp[sp.in_dev].set_index("uid").fold
    dtr = load_split_ids("dev_train")
    truth = truth_long()
    truth = truth[truth.s1.isin(dtr)].reset_index(drop=True)
    with stage("stage3b_train_pruner"), PeakSampler() as ps:
        tr = pd.read_parquet(ARTIFACTS_DIR / "cands_dev.parquet", filters=[("s1", "in", dtr.tolist())])
        oof, model = train_pruner(tr, "train", truth, folds)
    print("PEAKS", ps.report(), flush=True)
    from sklearn.metrics import average_precision_score, roc_auc_score
    print(f"OOF dev-train: {len(oof):,} pairs, positives {oof.y.mean():.4f}, AUC {roc_auc_score(oof.y, oof.p):.4f}, "
          f"AP {average_precision_score(oof.y, oof.p):.4f}")
    imp = pd.Series(model.feature_importance("gain"), index=FEATURE_COLS)
    (imp / imp.sum()).sort_values(ascending=False).to_csv(ARTIFACTS_DIR / "pruner_importance.csv")
    model.save_model(str(ARTIFACTS_DIR / "pruner.txt"))
    oof.to_parquet(ARTIFACTS_DIR / "pruner_oof_dev_train.parquet", index=False)


def prune_val() -> None:
    import lightgbm as lgb
    from prune import prune, score_file, tradeoff
    model = lgb.Booster(model_file=str(ARTIFACTS_DIR / "pruner.txt"))
    ids = load_split_ids("val")
    truth = truth_long()
    truth = truth[truth.s1.isin(ids)].reset_index(drop=True)
    certs = pd.read_parquet(ARTIFACTS_DIR / "certain_val.parquet")
    with stage("stage3b_score_val"), PeakSampler() as ps:
        sv = score_file(ARTIFACTS_DIR / "cands_val", "train", model)
        sv.to_parquet(ARTIFACTS_DIR / "pruner_scores_val.parquet", index=False)
    print("PEAKS", ps.report(), flush=True)
    with stage("stage3b_tradeoff_val"):
        t = tradeoff(sv, certs, truth, ids)
    t.to_csv(ARTIFACTS_DIR / "prune_tradeoff_val.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.sort_values("avg_cands").round(4).to_string(index=False))
    # pruned candidates at the operating point: val (validation) and dev-train (OOF, for Stage 4 training)
    kept = prune(sv, certs, **OPERATING_POINT)
    kept.to_parquet(ARTIFACTS_DIR / "pruned_val.parquet", index=False)
    oof = pd.read_parquet(ARTIFACTS_DIR / "pruner_oof_dev_train.parquet")
    dcert = pd.read_parquet(ARTIFACTS_DIR / "certain_dev.parquet")
    dtr = load_split_ids("dev_train")
    prune(oof[["s1", "m", "p"]], dcert[dcert.s1.isin(dtr)], **OPERATING_POINT).to_parquet(
        ARTIFACTS_DIR / "pruned_dev_train.parquet", index=False)
    from evaluate import candidate_stats
    st = candidate_stats(kept, truth, ids, pool_size_by_s1(ids))
    print("operating point", OPERATING_POINT, {k: round(float(v), 7) for k, v in st.items()}, flush=True)


def country_holdout() -> None:
    """Pruner trained on ONE country's dev-train, applied to the OTHER country's val candidates
    (stand-in for unseen France). Reports pruned completeness/size at the operating point and
    the baseline F0.5 (certain + top-1 >= threshold), next to the both-countries model."""
    import json
    from evaluate import candidate_stats, macro_f05
    from prune import prune, score_file, train_pruner
    from splits import load_splits
    from submit_baseline import THRESH_PATH, top1_matches
    t = json.loads(THRESH_PATH.read_text())["threshold"]
    sp = load_splits()
    folds = sp[sp.in_dev].set_index("uid").fold
    s1c = load("train", 1, ["uid", "country"]).set_index("uid").country.astype(str)
    dtr, ids = load_split_ids("dev_train"), load_split_ids("val")
    truth = truth_long()
    certs = pd.read_parquet(ARTIFACTS_DIR / "certain_val.parquet")
    both = pd.read_parquet(ARTIFACTS_DIR / "pruner_scores_val.parquet")
    rows = []
    for train_c, test_c in (("US", "India"), ("India", "US")):
        tr_ids = dtr[s1c.reindex(dtr).to_numpy() == train_c]
        te_ids = ids[s1c.reindex(ids).to_numpy() == test_c]
        tr_truth = truth[truth.s1.isin(tr_ids)].reset_index(drop=True)
        te_truth = truth[truth.s1.isin(te_ids)]
        with stage(f"holdout_train_{train_c}"):
            tr = pd.read_parquet(ARTIFACTS_DIR / "cands_dev.parquet", filters=[("s1", "in", tr_ids.tolist())])
            _, model = train_pruner(tr, "train", tr_truth, folds)
            del tr
        with stage(f"holdout_score_{test_c}"):
            sc = score_file(ARTIFACTS_DIR / "cands_val" / f"{test_c}.parquet", "train", model, min_p=OPERATING_POINT["floor"])
        ce = certs[certs.s1.isin(te_ids)]
        for label_, s in ((f"{train_c}->{test_c} (hold-out)", sc), (f"both->{test_c} (in-country)", both[both.s1.isin(te_ids)])):
            kept = prune(s, ce, **OPERATING_POINT)
            st = candidate_stats(kept, te_truth, te_ids)
            rows.append({"model": label_, "avg_cands": st["avg_cands"], "median_cands": st["median_cands"],
                         "pair_completeness": st["pair_completeness"],
                         "baseline_f05": macro_f05(top1_matches(kept[~kept.m.isin(ce.m)], ce, t), te_truth, te_ids)})
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS_DIR / "holdout_stage3.csv", index=False)
    print(res.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    {"block": block_val, "report": report_val_blocking, "train": train_pruner_dev, "prune": prune_val,
     "holdout": country_holdout}[sys.argv[1]]()
