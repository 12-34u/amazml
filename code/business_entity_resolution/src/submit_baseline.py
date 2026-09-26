"""Baseline submission: certain matches + the pruner's top-1 above a high threshold.

  1. choose the threshold on val (full pool, macro F0.5), using the Stage 3b scores;
  2. test: block (certain matches first), score candidates with the dev-trained pruner,
     prune at the Stage 3b operating point -> candidate_pairs.tsv;
  3. matches = certain pairs + each S1's best candidate if p >= threshold, with each S2/S3
     record assigned only to the S1 that scores it highest -> matching_results.tsv.
Every matched pair is a candidate by construction. IDs are mapped back through the stored
entity_id column (never rebuilt from the numeric uid, which would drop leading zeros).

Usage:  python src/submit_baseline.py threshold      # pick the threshold on val
        python src/submit_baseline.py test           # full test run -> output/*.tsv
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import ARTIFACTS_DIR, RESOURCE_DIR, load
from evaluate import macro_f05, write_id_list_tsv
from memtrack import PeakSampler, stage
from prune import prune, score_file
from stage3_run import OPERATING_POINT, truth_long
from splits import load_split_ids

OUT_DIR = RESOURCE_DIR / "output"
THRESH_PATH = ARTIFACTS_DIR / "baseline_threshold.json"


def top1_matches(scored: pd.DataFrame, certain: pd.DataFrame, t: float) -> pd.DataFrame:
    """Certain pairs + for each S1 its best candidate with p >= t, after giving every record
    only to its best-scoring S1 (records already certain-matched are excluded)."""
    c = scored[(scored.p >= t) & ~scored.m.isin(certain.m)]
    c = c.sort_values(["m", "p", "s1"], ascending=[True, False, True]).drop_duplicates("m")      # one S1 per record
    c = c.sort_values(["s1", "p", "m"], ascending=[True, False, True]).drop_duplicates("s1")     # top-1 per S1
    return pd.concat([certain[["s1", "m"]], c[["s1", "m"]]], ignore_index=True)


def choose_threshold() -> float:
    ids = load_split_ids("val")
    truth = truth_long()
    truth = truth[truth.s1.isin(ids)]
    sv = pd.read_parquet(ARTIFACTS_DIR / "pruner_scores_val.parquet")
    cert = pd.read_parquet(ARTIFACTS_DIR / "certain_val.parquet")
    rows = [{"threshold": 0.0, "rule": "certain only", "val_f05": macro_f05(cert, truth, ids)}]
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99):
        rows.append({"threshold": t, "rule": "certain + top-1", "val_f05": macro_f05(top1_matches(sv, cert, t), truth, ids)})
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    best = res[res.rule == "certain + top-1"].sort_values("val_f05", ascending=False).iloc[0]
    THRESH_PATH.write_text(json.dumps({"threshold": float(best.threshold), "val_f05": float(best.val_f05)}))
    return float(best.threshold)


def id_lookup(split: str, sources) -> pd.Series:
    return pd.concat([load(split, s, ["uid", "entity_id"]) for s in sources]).set_index("uid").entity_id


def run_test() -> None:
    from blocking import BlockConfig, run_blocking
    from data import available_sources
    t = json.loads(THRESH_PATH.read_text())["threshold"]
    test_s1 = load("test", 1, ["uid", "entity_id"])
    others = [s for s in available_sources("test") if s != 1]
    print(f"test sources available: S1 + {['S%d' % s for s in others]} (test_source2.tsv present: {2 in others})", flush=True)
    path, cert_path = ARTIFACTS_DIR / "cands_test", ARTIFACTS_DIR / "certain_test.parquet"
    if "--reuse-blocking" in sys.argv and cert_path.exists() and any(path.glob("*.parquet")):
        cert = pd.read_parquet(cert_path)
        print(f"reusing blocking output: {len(list(path.glob('*.parquet')))} files, {len(cert):,} certain pairs", flush=True)
    else:
        with stage("baseline_test_block"), PeakSampler() as ps:
            path, cert = run_blocking("test", test_s1.uid.to_numpy(), "test", BlockConfig())
        print("PEAKS", ps.report(), flush=True)
    model = lgb.Booster(model_file=str(ARTIFACTS_DIR / "pruner.txt"))
    with stage("baseline_test_score"), PeakSampler() as ps:
        st = score_file(path, "test", model, min_p=OPERATING_POINT["floor"])
    print("PEAKS", ps.report(), f"| scored pairs kept (p >= floor): {len(st):,}", flush=True)
    kept = prune(st, cert, **OPERATING_POINT)
    kept.to_parquet(ARTIFACTS_DIR / "pruned_test.parquet", index=False)
    matches = top1_matches(kept[~kept.m.isin(cert.m)], cert, t)   # top-1 chosen within the candidate set
    ids_s1 = test_s1.set_index("uid").entity_id
    ids_m = id_lookup("test", others)
    OUT_DIR.mkdir(exist_ok=True)
    for pairs, name, col in ((kept, "candidate_pairs.tsv", "candidate_entity_ids"),
                             (matches, "matching_results.tsv", "matched_entity_ids")):
        p = pd.DataFrame({"s1": ids_s1.loc[pairs.s1].to_numpy(), "m": ids_m.loc[pairs.m].to_numpy()}).drop_duplicates()
        write_id_list_tsv(p, test_s1.entity_id.tolist(), OUT_DIR / name, list_col=col)
    n = len(test_s1)
    print(f"test: {n:,} S1 | candidates {len(kept):,} ({len(kept)/n:.2f}/S1) | matches {len(matches):,} "
          f"({len(matches)/n:.2f}/S1), certain {len(cert):,}, threshold {t}", flush=True)


if __name__ == "__main__":
    {"threshold": choose_threshold, "test": run_test}[sys.argv[1]]()
