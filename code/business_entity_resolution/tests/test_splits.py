"""Invariants of the saved splits (needs artifacts/splits.parquet; skipped otherwise)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import splits  # noqa: E402

pytestmark = pytest.mark.skipif(not splits.SPLITS_PATH.exists(), reason="run `python src/splits.py` first")


def test_split_partition_and_fractions():
    sp = splits.load_splits()
    assert sp.uid.is_unique
    assert set(sp.split) == {"train", "val"}
    assert sp.split.eq("val").mean() == pytest.approx(0.2, abs=0.001)
    # stratified: val keeps the country mix and the singleton share
    for col in ("country",):
        a = sp[sp.split == "train"][col].value_counts(normalize=True)
        b = sp[sp.split == "val"][col].value_counts(normalize=True)
        assert (a - b).abs().max() < 0.002
    s_tr = (sp[sp.split == "train"].n_matches == 0).mean()
    s_va = (sp[sp.split == "val"].n_matches == 0).mean()
    assert abs(s_tr - s_va) < 0.001


def test_folds():
    sp = splits.load_splits()
    assert (sp.fold[sp.split == "val"] == -1).all()
    f = sp.fold[sp.split == "train"].value_counts(normalize=True)
    assert set(f.index) == set(range(splits.N_FOLDS))
    assert f.max() - f.min() < 0.001


def test_deterministic():
    a = splits.load_splits()
    b = splits.make_splits()
    assert a.equals(b)


def test_country_holdout_is_disjoint_and_complete():
    sp = splits.load_splits()
    dev = sp[sp.in_dev]
    for c in dev.country.unique():
        tr, va = splits.country_holdout(c)
        assert len(set(tr) & set(va)) == 0
        assert len(tr) + len(va) == len(dev)
        assert set(dev.country[dev.uid.isin(tr)]) == {c}


def test_dev_val_is_intersection():
    dv = set(splits.load_split_ids("dev_val"))
    assert dv == set(splits.load_split_ids("dev")) & set(splits.load_split_ids("val"))
