"""Unit tests for the Stage 3b pruning logic (src/prune.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prune import adaptive_cutoff, f05_upper_bound, prune, record_top2  # noqa: E402


def cands(rows):
    return pd.DataFrame(rows, columns=["s1", "m", "p"])


def test_record_top2_keeps_best_two_s1_per_record():
    c = cands([(1, 100, 0.9), (2, 100, 0.8), (3, 100, 0.7), (4, 100, 0.95), (1, 200, 0.1)])
    kept = record_top2(c, k=2)
    assert set(map(tuple, kept[kept.m == 100][["s1", "m"]].values)) == {(4, 100), (1, 100)}
    assert (1, 200) in set(map(tuple, kept[["s1", "m"]].values))


def test_adaptive_cutoff_floor_margin_and_cap():
    c = cands([(1, 10, 0.9), (1, 11, 0.5), (1, 12, 0.05), (1, 13, 0.3),
               (2, 20, 0.02),                        # best below the floor -> empty list
               (3, 30, 0.6), (3, 31, 0.59), (3, 32, 0.58)])
    kept = adaptive_cutoff(c, floor=0.04, margin=0.5, max_k=2)
    got = {s: sorted(g.m) for s, g in kept.groupby("s1")}
    assert got[1] == [10, 11]          # 0.3 is within margin but cut by max_k=2; 0.05 is outside the margin
    assert 2 not in got                # empty list allowed
    assert got[3] == [30, 31]          # cap


def test_prune_always_keeps_certain_pairs():
    c = cands([(1, 10, 0.01)])
    certain = pd.DataFrame({"s1": [1], "m": [99]})
    kept = prune(c, certain, floor=0.5, margin=1.0, max_k=8)
    assert set(map(tuple, kept[["s1", "m"]].values)) == {(1, 99)}


def test_f05_upper_bound():
    truth = pd.DataFrame({"s1": [1, 1, 2], "m": [10, 11, 20]})
    kept = cands([(1, 10, 0.9), (1, 12, 0.8)])     # S1 1: recall 1/2; S1 2: recall 0; S1 3: singleton
    r = 0.5
    expected = (1.25 * r / (0.25 + r) + 0.0 + 1.0) / 3
    assert f05_upper_bound(kept, truth, [1, 2, 3]) == pytest.approx(expected)
