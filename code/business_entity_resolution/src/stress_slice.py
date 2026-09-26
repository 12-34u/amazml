"""Test-like unknown-token stress slice (Stage 3).

On test, 17.3% of Indic-script names contain a token that is no English S1 word (test-only
vocabulary the lexicon cannot know). On val it is only ~5%. To see how blocking copes, hide a
random share of the lexicon's CONTENT-word mappings (legal words such as private/limited stay
mapped: test knows them too) until ~17% of val Indic names have an unknown token, re-normalise
all Indic-script pool records with that reduced lexicon, and re-block the val S1 entities that
own at least one Indic-script record. Compare with normal blocking on the same true pairs.

Usage:  python src/stress_slice.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from blocking import BlockConfig, load_side, run_blocking
from data import ARTIFACTS_DIR, NOTES_DIR, load, load_truth
from memtrack import PeakSampler, stage
from normalize import LEGAL_TOKENS, TITLE_TOKENS, hash_key, is_indic, detect_script, normalize_names
from normalize_build import load_cache, load_lexicon
from splits import load_split_ids

TARGET_UNKNOWN_NAMES = 0.173
INDIC = ["devanagari", "bengali", "gurmukhi", "gujarati", "oriya", "tamil", "telugu", "kannada", "malayalam"]
SEED = 7


def s1_vocab() -> pd.Index:
    """Tokens of the full normalised S1 names (train + test) - the same vocabulary as the test
    measurement in normalize_build.indic_token_coverage (4.6% tokens / 17.3% names unknown)."""
    from normalize_build import load_norm
    toks = pd.concat([load_norm(sp, 1, ["name_norm"]).name_norm.str.split().explode() for sp in ("train", "test")])
    return pd.Index(toks.dropna().unique())


def stressed_names(raw: pd.Series, cache, lexicon: pa.Table, hide_frac: float, rng_seed: int = SEED) -> tuple[pd.DataFrame, pa.Table]:
    """Normalise `raw` names with a lexicon missing `hide_frac` of its content-word entries."""
    lex = lexicon.to_pandas()
    legal = set(LEGAL_TOKENS) | set(LEGAL_TOKENS.values()) | set(TITLE_TOKENS) | {"private", "limited"}
    content = lex[~lex.dst.isin(legal)]
    hide = content.sample(frac=hide_frac, random_state=rng_seed).src
    reduced = pa.Table.from_pandas(lex[~lex.src.isin(hide)][["src", "dst"]], preserve_index=False)
    out = normalize_names(pa.array(raw.to_numpy(), pa.string()), cache, reduced)
    return pd.DataFrame({k: out[k].to_pandas().to_numpy() for k in ("name_core", "alias_core", "name_norm")}), reduced


def unknown_share(names: pd.Series, vocab: pd.Index) -> float:
    toks = names.str.split().explode()
    unk = ~toks.isin(vocab)
    return float(unk.groupby(level=0).any().mean())


def main() -> None:
    ids = load_split_ids("val")
    truth = load_truth(["s1_uid", "m_uid"]).rename(columns={"s1_uid": "s1", "m_uid": "m"})
    truth = truth[truth.s1.isin(ids)]
    # all India Indic-script pool records (decoys too), and the val-owned ones for calibration
    pool = pd.concat([load("train", s, ["uid", "business_name", "country"]) for s in (2, 3)], ignore_index=True)
    pool = pool[pool.country.astype(str) == "India"]
    codes = detect_script(pa.array(pool.business_name.to_numpy(), pa.string()))
    pool = pool[is_indic(codes)].reset_index(drop=True)
    owned = pool.uid.isin(truth.m).to_numpy()
    cache, lexicon, vocab = load_cache(), load_lexicon(), s1_vocab()
    base, _ = stressed_names(pool.business_name[owned].reset_index(drop=True), cache, lexicon, 0.0)
    print(f"Indic India pool records {len(pool):,} (val-owned {owned.sum():,}); unknown-name share with full lexicon "
          f"{unknown_share(base.name_norm, vocab):.3f}", flush=True)
    lo, hi, best = 0.0, 1.0, None
    for _ in range(10):                                        # bisection on the hidden fraction
        f = (lo + hi) / 2
        st, _ = stressed_names(pool.business_name[owned].reset_index(drop=True), cache, lexicon, f)
        u = unknown_share(st.name_norm, vocab)
        best = (f, u)
        if abs(u - TARGET_UNKNOWN_NAMES) < 0.005:
            break
        lo, hi = (f, hi) if u < TARGET_UNKNOWN_NAMES else (lo, f)
    f, u = best
    print(f"hidden content-word share {f:.3f} -> unknown-name share {u:.3f} (target {TARGET_UNKNOWN_NAMES})", flush=True)
    names, _ = stressed_names(pool.business_name.reset_index(drop=True), cache, lexicon, f)
    # plain arrays: Series would be ALIGNED to the uid index (all NaN)
    ov = pd.DataFrame({"name_core": names.name_core.to_numpy(), "alias_core": names.alias_core.to_numpy(),
                       "core_h": hash_key(pa.array(names.name_core, pa.string())),
                       "alias_h": hash_key(pa.array(names.alias_core, pa.string())),
                       "nn_h": hash_key(pa.array(names.name_norm, pa.string()))}, index=pool.uid.to_numpy())
    assert not ov.isna().any().any()
    # the slice: true pairs whose candidate is an Indic-script record; S1 = their owners
    slice_pairs = truth[truth.m.isin(pool.uid)]
    s1_slice = slice_pairs.s1.unique()
    print(f"slice: {len(slice_pairs):,} true pairs, {len(s1_slice):,} val India S1", flush=True)
    with stage("stress_slice_block"), PeakSampler() as ps:
        path_st, cert_st = run_blocking("train", s1_slice, "val_stress", BlockConfig(), countries_=["India"], pool_overrides=ov)
        c_st = pd.read_parquet(path_st, columns=["s1", "m"])
    print("PEAKS", ps.report(), flush=True)
    c_no = pd.read_parquet(ARTIFACTS_DIR / "cands_val", columns=["s1", "m"], filters=[("s1", "in", s1_slice.tolist())])
    cert_no = pd.read_parquet(ARTIFACTS_DIR / "certain_val.parquet")
    rows = []
    for label_, c, ce in (("normal lexicon", c_no, cert_no), ("stressed (test-like unknowns)", c_st, cert_st)):
        found = pd.concat([c[["s1", "m"]], ce[["s1", "m"]]]).drop_duplicates()
        hit = slice_pairs.merge(found)
        all_s1 = truth[truth.s1.isin(s1_slice)]
        rows.append({"run": label_, "slice true pairs": len(slice_pairs),
                     "slice completeness": len(hit) / len(slice_pairs),
                     "same S1, all their true pairs": all_s1.merge(found).shape[0] / len(all_s1),
                     "certain on slice": slice_pairs.merge(ce[["s1", "m"]]).shape[0] / len(slice_pairs),
                     "cands/S1": len(c[c.s1.isin(s1_slice)]) / len(s1_slice)})
    res = pd.DataFrame(rows)
    res.to_csv(ARTIFACTS_DIR / "stress_slice.csv", index=False)
    print(res.round(4).to_string(index=False))
    import shutil
    shutil.rmtree(ARTIFACTS_DIR / "cands_val_stress", ignore_errors=True)   # rebuildable intermediate


if __name__ == "__main__":
    main()
