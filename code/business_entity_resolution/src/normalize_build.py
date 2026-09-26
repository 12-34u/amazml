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
  6. region_compat.parquet     label-free region hierarchy: region A is compatible with B when
                               >= 80% of A's records sit in cities that also occur with B
                               (French departments inside regions: gironde -> nouvelle aquitaine)

Validation mode (default) learns the lexicon and region map from train-split labels only,
so val stays clean. Final mode (--final, for the test-set run) re-learns both from ALL
train labels (*_final.parquet) and re-normalises the test files with them. The region
list itself is label-free: component frequencies over all sources, which is the provided
data, not external data.

Usage:  python src/normalize_build.py [--force] [--from-step N]      validation resources, all files
        python src/normalize_build.py --final                        then: final resources, test files
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
REGION_COMPAT_PATH = ARTIFACTS_DIR / "region_compat.parquet"
COMPAT_MIN_CONTAINMENT, COMPAT_MIN_CITY_RECORDS = 0.8, 20
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


def _learning_truth(final: bool = False) -> pd.DataFrame:
    """Labels used for learning: the train split (validation mode) or all train data (final mode)."""
    tr = load_truth(["s1_uid", "m_uid"])
    if final:
        return tr
    from splits import load_splits
    sp = load_splits()
    return tr[tr.s1_uid.isin(sp.uid[sp.split == "train"])]


def _lexicon_path(final: bool):
    return LEXICON_PATH.with_name("translit_lexicon_final.parquet") if final else LEXICON_PATH


def _region_map_path(final: bool):
    return REGION_MAP_PATH.with_name("region_map_final.parquet") if final else REGION_MAP_PATH


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
def build_lexicon(cache: pa.Table, final: bool = False) -> None:
    tr = _learning_truth(final)
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
    lex.to_parquet(_lexicon_path(final), index=False)
    print(f"  aligned Indic true pairs: {int(same.sum()):,} of {len(same):,}; lexicon entries: {len(lex):,}; top:",
          lex.head(12)[["src", "dst"]].values.tolist())


def load_lexicon(final: bool = False) -> pa.Table:
    lex = pd.read_parquet(_lexicon_path(final))
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
    build_region_map(final=False)


def build_region_map(final: bool = False) -> None:
    """Spelling map: the region a candidate writes -> the region its true S1 writes."""
    reg = pd.read_parquet(REGIONS_PATH)
    unmapped = {"region_keys": pa.array((reg.country.str.lower() + "|" + reg.comp).tolist(), pa.string()),
                "region_map": pa.table({"src": pa.array([], pa.string()), "dst": pa.array([], pa.string())})}
    tr = _learning_truth(final)
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
    m.to_parquet(_region_map_path(final), index=False)
    print(f"  region spelling map: {len(m):,} entries, e.g.", m.sort_values("n", ascending=False).head(10)[["cand", "dst"]].values.tolist())


def load_regions(final: bool = False) -> dict:
    reg, mp_ = pd.read_parquet(REGIONS_PATH), pd.read_parquet(_region_map_path(final))
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


# ---------------------------------------------------------------- 6. region hierarchy
def build_region_compat() -> None:
    """Directional containment between region labels via shared cities, over all sources and
    both splits (no labels). A city 'links' A to B when it occurs with B at least 20 times;
    A -> B is compatible when >= 80% of A's region-tagged records are in such cities.
    Stored both ways (A,B) and (B,A) for lookup."""
    parts = []
    for split, s in _files():
        n = load_norm(split, s, ["region", "city"])
        c = load(split, s, ["country"]).country.astype(str).values
        d = pd.DataFrame({"country": c, "region": n.region.values, "city": n.city.values})
        parts.append(d[(d.region != "") & (d.city != "")].value_counts().rename("n").reset_index())
    co = pd.concat(parts).groupby(["country", "city", "region"], as_index=False).n.sum()
    rows = []
    for country, g in co.groupby("country"):
        totals = g.groupby("region").n.sum()
        strong = g[g.n >= COMPAT_MIN_CITY_RECORDS]
        cities_of = strong.groupby("region").city.apply(set)
        for a, na in totals.items():
            ga = g[g.region == a]
            for b, cb in cities_of.items():
                if a == b:
                    continue
                share = ga.n[ga.city.isin(cb)].sum() / na
                if share >= COMPAT_MIN_CONTAINMENT:
                    rows.append({"country": country, "region_a": a, "region_b": b, "containment": share})
    m = pd.DataFrame(rows, columns=["country", "region_a", "region_b", "containment"])
    both = pd.concat([m, m.rename(columns={"region_a": "region_b", "region_b": "region_a"})]).drop_duplicates(
        ["country", "region_a", "region_b"])
    both.to_parquet(REGION_COMPAT_PATH, index=False)
    print(f"  region hierarchy links: {len(m)} directional, e.g.", m.sort_values("containment", ascending=False)
          .head(8)[["country", "region_a", "region_b"]].values.tolist())


# ---------------------------------------------------------------- unknown Indic tokens
def indic_token_coverage(split: str, sources=(2, 3), uids=None) -> dict:
    """Among names written in an Indic script: the share of tokens (after transliteration
    and lexicon) that are not English vocabulary, where the vocabulary is every token of
    the Latin S1 names (train + test). Also the share of names with at least one such token."""
    vocab = pa.concat_arrays([pc.unique(pc.list_flatten(pc.split_pattern(
        pa.array(load_norm(sp, 1, ["name_norm"]).name_norm, pa.string()), " "))) for sp in ("train", "test")])
    vocab = pc.unique(vocab)
    tot_tok = unk_tok = names = names_unk = 0
    for s in sources:
        if not norm_path(split, s).exists():      # test S2 may be absent
            continue
        d = load_norm(split, s, ["uid", "name_script", "name_norm"])
        d = d[d.name_script.astype(str).isin(["devanagari", "bengali", "gurmukhi", "gujarati", "oriya", "tamil",
                                               "telugu", "kannada", "malayalam"])]
        if uids is not None:
            d = d[d.uid.isin(uids)]
        toks = pc.split_pattern(pa.array(d.name_norm, pa.string()), " ")
        flat, par = pc.list_flatten(toks), _np(pc.list_parent_indices(toks))
        unk = ~_np(pc.is_in(flat, value_set=vocab))
        tot_tok += len(flat)
        unk_tok += int(unk.sum())
        names += len(d)
        names_unk += len(np.unique(par[unk]))
    return {"indic_names": names, "tokens": tot_tok, "unknown_token_share": unk_tok / max(tot_tok, 1),
            "names_with_unknown_share": names_unk / max(names, 1)}


def final_main() -> None:
    """Final-pipeline resources from ALL train labels; re-normalise the test files with them."""
    cache = load_cache()
    with stage("normalize_final_lexicon_regionmap"):
        build_lexicon(cache, final=True)
        build_region_map(final=True)
    _SHARED.update(cache=cache, lexicon=load_lexicon(final=True), regions=load_regions(final=True))
    test_files = [f for f in _files() if f[0] == "test"]
    with stage("normalize_final_test_files"):
        for f in test_files:                      # sequential: disk is tight
            _normalize_worker(f)
            _geo_worker(f)
    with stage("normalize_final_region_compat"):
        build_region_compat()
    with stage("normalize_final_indic_coverage"):
        print("  test S2/S3, final lexicon:", indic_token_coverage("test"))


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
    with stage("normalize_6_region_compat"):
        build_region_compat()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from-step", type=int, default=99, help="rebuild this step and every later one")
    ap.add_argument("--final", action="store_true", help="final-pipeline resources from all train labels; re-normalise test")
    a = ap.parse_args()
    final_main() if a.final else main(a.force, a.from_step)
