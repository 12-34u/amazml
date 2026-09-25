"""Unit tests for src/evaluate.py.  Run:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import (candidate_stats, f05, macro_f05, per_entity_scores,  # noqa: E402
                      read_id_list_tsv, write_id_list_tsv)


def pairs(d: dict) -> pd.DataFrame:
    rows = [(s, m) for s, ms in d.items() for m in ms]
    return pd.DataFrame(rows, columns=["s1", "m"])


# ---------------------------------------------------------------- official example
def test_official_example():
    pred = ["S2-00047", "S2-00193", "S3-00812"]
    truth = ["S2-00047", "S3-00812"]
    assert f05(pred, truth) == pytest.approx(0.7142857, abs=1e-6)
    assert round(f05(pred, truth), 3) == 0.714
    v = macro_f05(pairs({"S1-00001": pred}), pairs({"S1-00001": truth}), ["S1-00001"])
    assert v == pytest.approx(0.7142857, abs=1e-6)


# ---------------------------------------------------------------- edge cases
@pytest.mark.parametrize("pred,truth,expected", [
    ([], [], 1.0),                       # singleton, correctly empty
    (["S2-1"], [], 0.0),                 # singleton, any prediction scores 0
    ([], ["S2-1"], 0.0),                 # non-singleton, empty prediction
    (["S2-9"], ["S2-1"], 0.0),           # no overlap
    (["S2-1", "S3-2"], ["S2-1", "S3-2"], 1.0),
    (["S2-1"], ["S2-1", "S3-2"], 1.25 * 1 * 0.5 / (0.25 * 1 + 0.5)),   # P=1, R=.5 -> 0.8333
    (["S2-1", "S2-2"], ["S2-1"], 1.25 * 0.5 * 1 / (0.25 * 0.5 + 1)),   # P=.5, R=1 -> 0.5556
])
def test_single_entity_cases(pred, truth, expected):
    assert f05(pred, truth) == pytest.approx(expected)
    s1 = ["S1-x"]
    assert macro_f05(pairs({"S1-x": pred}), pairs({"S1-x": truth}), s1) == pytest.approx(expected)


def test_precision_weighted_more_than_recall():
    # the same number of errors costs more as a false positive than as a miss
    assert f05(["a", "b", "c"], ["a", "b"]) < f05(["a"], ["a", "b"])


def test_macro_includes_missing_rows_and_singletons():
    truth = pairs({"A": ["x", "y"], "B": ["z"]})          # C is a singleton (no truth rows)
    pred = pairs({"A": ["x", "y"]})                        # B and C have no prediction rows
    # A=1.0, B=0.0 (missed), C=1.0 (correct empty)
    assert macro_f05(pred, truth, ["A", "B", "C"]) == pytest.approx(2 / 3)


def test_duplicates_and_out_of_scope_pairs_ignored():
    truth = pairs({"A": ["x"]})
    pred = pairs({"A": ["x", "x"], "Z": ["q"]})            # duplicate pair; Z not evaluated
    assert macro_f05(pred, truth, ["A"]) == pytest.approx(1.0)


def test_vectorised_matches_reference_on_random_data():
    rng = np.random.default_rng(0)
    s1 = [f"S1-{i}" for i in range(400)]
    pool = [f"S2-{i}" for i in range(60)]
    truth = {s: list(rng.choice(pool, rng.integers(0, 5), replace=False)) for s in s1}
    pred = {s: list(rng.choice(pool, rng.integers(0, 5), replace=False)) for s in s1}
    ref = np.mean([f05(pred[s], truth[s]) for s in s1])
    got = per_entity_scores(pairs(pred), pairs(truth), s1)
    assert got.f05.mean() == pytest.approx(ref, abs=1e-12)
    assert all(got.loc[s, "f05"] == pytest.approx(f05(pred[s], truth[s])) for s in s1[:50])


def test_int_uids_work_too():
    truth = pd.DataFrame({"s1": [1, 1], "m": [20, 30]})
    pred = pd.DataFrame({"s1": [1, 1, 1], "m": [20, 30, 40]})
    assert macro_f05(pred, truth, [1, 2]) == pytest.approx((f05([20, 30, 40], [20, 30]) + 1.0) / 2)


# ---------------------------------------------------------------- candidate metrics
def test_candidate_stats():
    truth = pairs({"A": ["x", "y"], "B": ["z"]})
    cands = pairs({"A": ["x", "q", "r"], "C": ["w"]})     # B gets an empty list
    st = candidate_stats(cands, truth, ["A", "B", "C"],
                         pool_size_by_s1=pd.Series({"A": 100, "B": 100, "C": 100}))
    assert st["pair_completeness"] == pytest.approx(1 / 3)
    assert st["avg_cands"] == pytest.approx(4 / 3)
    assert st["median_cands"] == 1.0
    assert st["empty_lists"] == pytest.approx(1 / 3)
    assert st["entity_full_recall"] == 0.0                # A misses y, B misses z
    assert st["reduction_ratio"] == pytest.approx(1 - 4 / 300)


# ---------------------------------------------------------------- file round trip
def test_tsv_round_trip(tmp_path):
    p = pairs({"S1-1": ["S2-47", "S3-812"], "S1-2": ["S3-4"]})
    path = tmp_path / "m.tsv"
    write_id_list_tsv(p, ["S1-1", "S1-2", "S1-3"], path)
    text = path.read_text()
    assert text.splitlines() == ["source1_entity_id\tmatched_entity_ids",
                                 "S1-1\tS2-47,S3-812", "S1-2\tS3-4", "S1-3\t"]
    back = read_id_list_tsv(path)
    assert sorted(map(tuple, back.values.tolist())) == sorted(map(tuple, p.values.tolist()))
    assert list(back.attrs["s1_rows"]) == ["S1-1", "S1-2", "S1-3"]
