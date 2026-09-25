"""Stage 2 build: learn the normalisation resources, then normalise every source file.

Steps (each cached in artifacts/, skipped when present unless --force):
  1. translit_cache.parquet    unique Indic-script names/addresses -> Latin (indic-transliteration, MIT)
  2. translit_lexicon.parquet  transliterated token -> English token, learned from train-split
                               true pairs whose candidate name is in an Indic script
  3. norm_{split}_s{n}.parquet normalised name + address fields per record (normalize.py),
                               in 500k-row chunks, files in parallel
  4. regions.parquet           region-level address components per country: frequent AND
                               usually the last component (normalize.REGION_*)
     region_map.parquet        region spelling -> S1 spelling, learned from train-split true pairs
  5. geo pass                  adds city, region and localities to every norm file (from addr_norm)

Only train-split labels (splits.parquet) are used for learning, so val stays clean.
Unlabelled text (all sources, both splits) is used for component frequencies. That is
the provided data, not external data.

Usage:  python src/normalize_build.py [--force] [--from-step N]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data import ARTIFACTS_DIR, available_sources, load, load_truth, prepared_path
from memtrack import stage
from normalize import (REGION_MIN_LAST_RATIO, REGION_MIN_SHARE, SCRIPT_LABELS, _clean_name_text, _fold, _np,
                       apply_translit_cache, city_region, component_table, detect_script, hash_key, is_indic,
                       normalize_addresses, normalize_names, transliterate_unique)

CHUNK = 500_000
JOBS = 2            # parallel file workers: 3 peaked at 4.67 GB total PSS, 2 leaves margin under 5 GB
CACHE_PATH = ARTIFACTS_DIR / "translit_cache.parquet"
LEXICON_PATH = ARTIFACTS_DIR / "translit_lexicon.parquet"
REGIONS_PATH = ARTIFACTS_DIR / "regions.parquet"
REGION_MAP_PATH = ARTIFACTS_DIR / "region_map.parquet"
MAP_MIN_COUNT, MAP_MIN_SHARE = 20, 0.8
LEX_MIN_COUNT, LEX_MIN_SHARE = 3, 0.6
_SHARED: dict = {}  # resources inherited by forked workers


def norm_path(split: str, source: int):
    return ARTIFACTS_DIR / f"norm_{split}_s{source}.parquet"


def load_norm(split: str, source: int, columns=None, filters=None) -> pd.DataFrame:
    return pq.read_table(norm_path(split, source), columns=columns, filters=filters).to_pandas()


def _files():
    return [(sp, s) for sp in ("train", "test") for s in available_sources(sp)]


def _iter_chunks(split, source, columns):
    for batch in pq.ParquetFile(prepared_path(split, source)).iter_batches(batch_size=CHUNK, columns=columns):
        yield pa.Table.from_batches([batch])


def _train_split_truth() -> pd.DataFrame:
    from splits import load_splits
    sp = load_splits()
    tr = load_truth(["s1_uid", "m_uid"])
    return tr[tr.s1_uid.isin(sp.uid[sp.split == "train"])]


# ---------------------------------------------------------------- 1. transliteration cache
def _translit_worker(args):
    return transliterate_unique(*args)


def build_translit_cache(workers: int = 8) -> None:
    raw, scripts, seen = [], [], set()
    for split, s in _files():
        for t in _iter_chunks(split, s, ["business_name", "business_address"]):
            for col in t.column_names:
                arr = t[col].combine_chunks()
                codes = detect_script(arr)
                m = is_indic(codes)
                if not m.any():
                    continue
                sub, sc = pc.filter(arr, pa.array(m)), codes[m]
                u = pc.unique(sub)
                first = pc.index_in(u, value_set=sub)  # first occurrence -> script
                for v, c in zip(u.to_pylist(), sc[first.to_numpy()]):
                    if v not in seen:
                        seen.add(v)
                        raw.append(v)
                        scripts.append(SCRIPT_LABELS[c])
    print(f"  unique Indic strings: {len(raw):,}")
    step = 20_000
    jobs = [(raw[i:i + step], scripts[i:i + step]) for i in range(0, len(raw), step)]
    with mp.get_context("fork").Pool(workers) as pool:
        latin = [x for part in pool.map(_translit_worker, jobs) for x in part]
    pq.write_table(pa.table({"raw": raw, "latin": latin, "script": scripts}), CACHE_PATH, compression="zstd")


def load_cache() -> pa.Table:
    return pq.read_table(CACHE_PATH, columns=["raw", "latin"])


# ---------------------------------------------------------------- 2. lexicon
def build_lexicon(cache: pa.Table) -> None:
    tr = _train_split_truth()
    parts = []
    for s in (2, 3):
        o = load("train", s, ["uid", "business_name"])
        o = o[o.uid.isin(tr.m_uid)]
        arr = pa.array(o.business_name, pa.string())
        codes = detect_script(arr)
        keep = is_indic(codes)
        latin = apply_translit_cache(pc.filter(arr, pa.array(keep)), codes[keep], cache)
        parts.append(pd.DataFrame({"m_uid": o.uid.values[keep], "cand": _clean_name_text(_fold(latin)).to_pandas().values}))
    cand = pd.concat(parts).merge(tr, on="m_uid")
    s1 = load("train", 1, ["uid", "business_name"])
    s1 = s1[s1.uid.isin(cand.s1_uid)]
    s1n = pd.Series(_clean_name_text(_fold(pa.array(s1.business_name, pa.string()))).to_pandas().values, index=s1.uid.values)
    cand["eng"] = cand.s1_uid.map(s1n)
    a = pc.split_pattern(pa.array(cand.cand, pa.string()), " ")
    b = pc.split_pattern(pa.array(cand.eng, pa.string()), " ")
    same = pc.list_value_length(a).to_numpy() == pc.list_value_length(b).to_numpy()  # align token by token
    a, b = pc.filter(a, pa.array(same)), pc.filter(b, pa.array(same))
    tok = pd.DataFrame({"src": pc.list_flatten(a).to_pandas().values, "dst": pc.list_flatten(b).to_pandas().values})
    cnt = tok.value_counts().rename("n").reset_index()
    tot = cnt.groupby("src").n.transform("sum")
    lex = cnt[(cnt.n >= LEX_MIN_COUNT) & (cnt.n / tot >= LEX_MIN_SHARE) & (cnt.src != cnt.dst)]
    lex = lex.sort_values("n", ascending=False).drop_duplicates("src")
    lex.to_parquet(LEXICON_PATH, index=False)
    print(f"  aligned Indic true pairs: {int(same.sum()):,} of {len(same):,}; lexicon entries: {len(lex):,}; top:",
          lex.head(12)[["src", "dst"]].values.tolist())


def load_lexicon() -> pa.Table:
    lex = pd.read_parquet(LEXICON_PATH)
    return pa.table({"src": lex.src.tolist(), "dst": lex.dst.tolist()})


# ---------------------------------------------------------------- 3. normalise all files
def normalize_file(split: str, source: int, cache, lexicon) -> None:
    writer = None
    for t in _iter_chunks(split, source, ["uid", "business_name", "business_address"]):
        nm = normalize_names(t["business_name"], cache, lexicon)
        ad = normalize_addresses(t["business_address"], cache=cache)
        cols = {"uid": t["uid"], **nm, **ad,
                "core_h": pa.array(hash_key(nm["name_core"])), "hs_h": pa.array(hash_key(ad["hs_key"]))}
        out = pa.table({k: (v.combine_chunks() if isinstance(v, pa.ChunkedArray) else v) for k, v in cols.items()})
        if writer is None:
            writer = pq.ParquetWriter(norm_path(split, source), out.schema, compression="zstd")
        writer.write_table(out)
    writer.close()
    print(f"  {split} S{source} -> {norm_path(split, source).name}")


def _normalize_worker(file):
    normalize_file(*file, _SHARED["cache"], _SHARED["lexicon"])


# ---------------------------------------------------------------- 4. regions
def _country_codes(split, source):
    c = load(split, source, ["country"]).country
    return np.asarray(c.cat.categories, dtype=object), c.cat.codes.to_numpy()


def build_regions() -> None:
    parts = []
    for split, s in _files():
        cats, codes = _country_codes(split, s)
        off = 0
        for b in pq.ParquetFile(norm_path(split, s)).iter_batches(1_000_000, columns=["addr_norm"]):
            n = b.num_rows
            flat, parents, usable, is_last = component_table(b.column(0), n)
            ctry = cats[codes[off:off + n][parents]]
            df = pd.DataFrame({"country": ctry[usable], "comp": _np(pc.filter(flat, pa.array(usable))), "last": is_last[usable]})
            parts.append(df.groupby(["country", "comp"]).agg(n=("last", "size"), n_last=("last", "sum")).reset_index())
            off += n
    cnt = pd.concat(parts).groupby(["country", "comp"], as_index=False)[["n", "n_last"]].sum()
    cnt["last_ratio"] = cnt.n_last / cnt.n
    cnt["share"] = cnt.n / cnt.groupby("country").n.transform("sum")
    reg = cnt[(cnt.share >= REGION_MIN_SHARE) & (cnt.last_ratio >= REGION_MIN_LAST_RATIO)]
    reg.sort_values(["country", "n"], ascending=[True, False]).to_parquet(REGIONS_PATH, index=False)
    print("  region-level components per country:", reg.groupby("country").size().to_dict())

    # spelling map: the region a candidate writes -> the region its true S1 writes
    unmapped = {"region_keys": pa.array((reg.country.str.lower() + "|" + reg.comp).tolist(), pa.string()),
                "region_map": pa.table({"src": pa.array([], pa.string()), "dst": pa.array([], pa.string())})}
    tr = _train_split_truth()
    region_of = {}
    for s in (1, 2, 3):
        uids = tr.s1_uid if s == 1 else tr.m_uid
        n = load_norm("train", s, ["uid", "addr_norm"])
        c = load("train", s, ["uid", "country"])
        keep = n.uid.isin(uids).to_numpy()
        n, c = n[keep], c[keep]
        r = city_region(pa.array(n.addr_norm, pa.string()), pa.array(c.country.astype(str), pa.string()), unmapped)["region"]
        region_of[s] = pd.Series(r.to_pandas().values, index=n.uid.values)
        if s == 1:
            s1_country = pd.Series(c.country.astype(str).values, index=c.uid.values)
    cand = pd.concat([region_of[2], region_of[3]])
    pairs = pd.DataFrame({"country": tr.s1_uid.map(s1_country).values, "dst": tr.s1_uid.map(region_of[1]).values,
                          "cand": tr.m_uid.map(cand).values})
    pairs = pairs[(pairs.cand != "") & (pairs.dst != "") & pairs.cand.notna() & pairs.dst.notna()]
    m = pairs.value_counts().rename("n").reset_index()
    tot = m.groupby(["country", "cand"]).n.transform("sum")
    m = m[(m.n >= MAP_MIN_COUNT) & (m.n / tot >= MAP_MIN_SHARE) & (m.cand != m.dst)]
    m.to_parquet(REGION_MAP_PATH, index=False)
    print(f"  region spelling map: {len(m):,} entries, e.g.", m.sort_values("n", ascending=False).head(10)[["cand", "dst"]].values.tolist())


def load_regions() -> dict:
    reg, mp_ = pd.read_parquet(REGIONS_PATH), pd.read_parquet(REGION_MAP_PATH)
    return {"region_keys": pa.array((reg.country.str.lower() + "|" + reg.comp).tolist(), pa.string()),
            "region_map": pa.table({"src": (mp_.country.str.lower() + "|" + mp_.cand).tolist(), "dst": mp_.dst.tolist()})}


# ---------------------------------------------------------------- 5. geo pass
def add_city_region(split: str, source: int, regions: dict) -> None:
    """Rewrite a norm file with fresh city/region columns (streamed via a temp file)."""
    path, tmp = norm_path(split, source), norm_path(split, source).with_suffix(".tmp")
    cats, codes = _country_codes(split, source)
    writer, off = None, 0
    for b in pq.ParquetFile(path).iter_batches(CHUNK):
        t = pa.Table.from_batches([b])
        t = t.drop_columns([c for c in ("city", "region", "localities") if c in t.column_names])
        country = pa.array(cats[codes[off:off + t.num_rows]], pa.string())
        geo = city_region(t["addr_norm"], country, regions)
        for k in ("city", "region", "localities"):
            t = t.append_column(k, geo[k])
        if writer is None:
            writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t)
        off += t.num_rows
    writer.close()
    os.replace(tmp, path)
    print(f"  {split} S{source}: city/region added")


def _geo_worker(file):
    add_city_region(*file, _SHARED["regions"])


def main(force: bool = False, from_step: int = 1) -> None:
    f = lambda step: force or from_step <= step
    with stage("normalize_1_translit_cache"):
        if f(1) or not CACHE_PATH.exists():
            build_translit_cache()
    cache = load_cache()
    with stage("normalize_2_lexicon"):
        if f(2) or not LEXICON_PATH.exists():
            build_lexicon(cache)
    _SHARED.update(cache=cache, lexicon=load_lexicon())
    with stage("normalize_3_all_files"):
        todo = [x for x in _files() if f(3) or not norm_path(*x).exists()]
        if todo:
            with mp.get_context("fork").Pool(JOBS) as pool:
                pool.map(_normalize_worker, todo)
    with stage("normalize_4_regions"):
        if f(4) or not REGION_MAP_PATH.exists():
            build_regions()
    _SHARED["regions"] = load_regions()
    with stage("normalize_5_city_region"):
        # sequential: each rewrite needs a temp copy and free disk is tight
        for x in _files():
            _geo_worker(x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from-step", type=int, default=99, help="rebuild this step and every later one")
    a = ap.parse_args()
    main(a.force, a.from_step)
