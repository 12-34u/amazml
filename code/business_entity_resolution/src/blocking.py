"""Stage 3 — candidate generation (blocking).

Per country and in batches of S1 queries; never all-pairs. Everything is an indexed or
key lookup:
  certain  exact name_norm AND hs_key (house number + street word). These pairs are saved,
           and their S2/S3 records leave the pool (one-to-one).
  tfidf    (a) char_wb 3-gram TF-IDF on the transliterated core name, sparse top-K cosine
           (sparse_dot_topn, Apache-2.0)
  name     (b) exact core-name blocks, ranked by address/locality overlap, top-M per S1
  addr     (c) address-key blocks (house number + street word; house number + locality),
           ranked by name similarity, top-M per S1. Catches renamed businesses.
  alias    (d) S2/S3 alias names ("X fka Y") are indexed as extra documents and extra
           name keys that point back to their record. S1 has no aliases, so aliases only
           appear on the pool side; any S1 alias would be queried the same way.
Output: one row per (s1, m) candidate with a flag per blocker and cheap scores, which the
Stage 3b pruner uses as features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from data import available_sources, prepared_path
from normalize import hash_key
from normalize_build import REGION_COMPAT_PATH, norm_path

SIDE_COLS = ["uid", "name_core", "alias_core", "name_norm", "legal_form", "house_no", "street_word",
             "hs_h", "core_h", "localities", "region", "city", "name_script"]


# ---------------------------------------------------------------- loading
def load_side(split: str, sources, country: str, uids: np.ndarray | None = None,
              columns: list[str] = SIDE_COLS, exclude: np.ndarray | None = None) -> pd.DataFrame:
    """Norm fields for one country (and optionally a uid subset). The norm and prepared
    files share row order, so the country mask from the prepared file applies directly."""
    parts = []
    for s in sources:
        if not norm_path(split, s).exists():
            continue
        if uids is not None and exclude is None:
            # uid subset: push the filter into the parquet scan (only matching rows are materialised).
            # The country is checked too: callers may pass uids spanning several countries.
            meta = pq.read_table(prepared_path(split, s), columns=["uid", "country"],
                                 filters=[("uid", "in", pa.array(uids))])
            keep = pc.filter(meta["uid"], pc.equal(pc.cast(meta["country"], pa.string()), country))
            if len(keep) == 0:
                continue
            t = pq.read_table(norm_path(split, s), columns=columns, filters=[("uid", "in", keep)])
            parts.append(t.to_pandas())
            continue
        meta = pq.read_table(prepared_path(split, s), columns=["uid", "country"])
        mask = pc.equal(pc.cast(meta["country"], pa.string()), country)
        if uids is not None:
            mask = pc.and_(mask, pc.is_in(meta["uid"], value_set=pa.array(uids)))
        if exclude is not None and len(exclude):
            mask = pc.and_(mask, pc.invert(pc.is_in(meta["uid"], value_set=pa.array(exclude))))
        t = pq.read_table(norm_path(split, s), columns=columns).filter(mask)
        parts.append(t.to_pandas())
    if not parts:
        return pd.DataFrame(columns=columns)
    df = pd.concat(parts, ignore_index=True)
    if "name_norm" in df:
        df["nn_h"] = hash_key(pa.array(df.pop("name_norm"), pa.string()))   # only the hash is needed
    if "alias_core" in df:
        df["alias_h"] = hash_key(pa.array(df.alias_core, pa.string()))
    return df


def countries(split: str) -> list[str]:
    c = pq.read_table(prepared_path(split, 1), columns=["country"])["country"]
    return sorted(pc.unique(pc.cast(c, pa.string())).to_pylist())


# ---------------------------------------------------------------- certain matches
def certain_matches(s1: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Exact name_norm + hs_key. A record claimed by several S1 entities is ambiguous and
    left in the pool."""
    l = s1[(s1.nn_h != 0) & (s1.hs_h != 0)][["uid", "nn_h", "hs_h"]]
    r = pool[(pool.nn_h != 0) & (pool.hs_h != 0)][["uid", "nn_h", "hs_h"]]
    j = l.merge(r, on=["nn_h", "hs_h"], suffixes=("", "_m"))
    j = j[~j.uid_m.duplicated(keep=False)]
    return pd.DataFrame({"s1": j.uid.to_numpy(), "m": j.uid_m.to_numpy()})


# ---------------------------------------------------------------- shared helpers
def load_region_compat() -> set:
    """{(country, region_a, region_b)} learned label-free in Stage 2 (both directions)."""
    if not REGION_COMPAT_PATH.exists():
        return set()
    m = pd.read_parquet(REGION_COMPAT_PATH)
    return set(zip(m.country, m.region_a, m.region_b))


def explode_localities(df: pd.DataFrame) -> pd.DataFrame:
    """uid -> one row per locality hash (the '|'-separated `localities` field)."""
    d = df.loc[df.localities != "", ["uid", "localities"]]
    toks = pc.split_pattern(pa.array(d.localities, pa.string()), "|")
    flat, par = pc.list_flatten(toks), pc.list_parent_indices(toks).to_numpy()
    return pd.DataFrame({"uid": d.uid.to_numpy()[par], "loc_h": hash_key(flat)})


def locality_overlap(pairs: pd.DataFrame, s1_locs: pd.DataFrame, pool_locs: pd.DataFrame) -> np.ndarray:
    """Number of shared localities for each (s1, m) row of `pairs` (row order kept)."""
    p = pairs[["s1", "m"]].reset_index(drop=True).rename_axis("pid").reset_index()
    a = p.merge(s1_locs, left_on="s1", right_on="uid")[["pid", "loc_h"]]
    b = p.merge(pool_locs, left_on="m", right_on="uid")[["pid", "loc_h"]]
    hit = a.merge(b, on=["pid", "loc_h"]).pid.value_counts()
    return hit.reindex(p.pid, fill_value=0).to_numpy()


def region_compatible(a: np.ndarray, b: np.ndarray, country: str, compat: set) -> np.ndarray:
    """1 = same region or linked in the learned hierarchy, 0.5 = one side missing, 0 = conflict."""
    out = np.where((a == "") | (b == ""), 0.5, 0.0)
    same = (a == b) & (a != "")
    linked = np.fromiter(((country, x, y) in compat for x, y in zip(a, b)), bool, len(a)) if compat else np.zeros(len(a), bool)
    return np.where(same | linked, 1.0, out)


def name_similarity(s1_core: np.ndarray, m_core: np.ndarray, m_alias: np.ndarray) -> np.ndarray:
    """Token-set ratio of the S1 core name against the candidate's core or alias (best)."""
    a = process.cpdist(list(s1_core), list(m_core), scorer=fuzz.token_set_ratio, workers=-1)
    has_alias = m_alias != ""
    if has_alias.any():
        b = np.zeros(len(a), dtype=a.dtype)
        b[has_alias] = process.cpdist(list(s1_core[has_alias]), list(m_alias[has_alias]),
                                      scorer=fuzz.token_set_ratio, workers=-1)
        a = np.maximum(a, b)
    return a.astype(np.float32)


def address_score(s1: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame, loc_overlap: np.ndarray,
                  region_c: np.ndarray) -> np.ndarray:
    """Cheap address agreement used to rank name blocks:
    4*hs_key + 2*house_no + 1*street_word + 2*min(shared localities, 2) + 1*region_compatible."""
    a, b = s1.set_index("uid").loc[pairs.s1], pool.set_index("uid").loc[pairs.m]
    eq = lambda col: (a[col].to_numpy() == b[col].to_numpy()) & (a[col].to_numpy() != "")
    hs = (a.hs_h.to_numpy() == b.hs_h.to_numpy()) & (a.hs_h.to_numpy() != 0)
    return (4 * hs + 2 * eq("house_no") + eq("street_word") + 2 * np.minimum(loc_overlap, 2) + region_c).astype(np.float32)


def top_per_s1(pairs: pd.DataFrame, score: str, m: int) -> pd.DataFrame:
    """Keep the m best candidates per S1 by `score` (ties broken by candidate uid, deterministic)."""
    pairs = pairs.sort_values(["s1", score, "m"], ascending=[True, False, True], kind="stable")
    return pairs[pairs.groupby("s1").cumcount() < m]


# ================================================================ blockers
from dataclasses import dataclass, asdict  # noqa: E402

import scipy.sparse as sp  # noqa: E402
from sparse_dot_topn import sp_matmul_topn  # noqa: E402  (Apache-2.0)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from memtrack import check_budget, phase  # noqa: E402

BLOCKERS = ("tfidf", "name", "loc_tfidf", "hs", "num_loc", "addr_tfidf", "noaddr_tfidf")
RUN_COLS = ["uid", "name_core", "alias_core", "name_norm", "house_no", "street_word", "hs_h", "core_h",
            "localities", "region", "addr_norm"]   # addr_norm is dropped right after number extraction


@dataclass
class BlockConfig:
    tfidf_k: int = 30             # (a) global name TF-IDF top-K
    tfidf_min_cos: float = 0.3
    max_df: float = 0.02          # trigram vocabulary: drop trigrams in > 2% of names
    global_max_df: float = 0.005  # (a) global search uses only trigrams in <= 0.5% of names (~10x faster)
    name_m: int = 20              # (b) exact core/alias-name block, top-M by address score
    loc_m: int = 20               # (c) name TF-IDF within each shared locality, top-M
    loc_min_cos: float = 0.1
    hs_m: int = 20                # (c) house number + street word block, top-M by name similarity
    num_loc_m: int = 20           # (c) any address number + locality block, top-M by name similarity
    num_loc_max_block: int = 200  # address blocks bigger than this carry no identity signal
    addr_m: int = 10              # (e) address-text TF-IDF inside each shared locality, top-M (renamed businesses)
    addr_min_cos: float = 0.3
    noaddr_k: int = 10            # (f) full-vocabulary name TF-IDF over pool records with no locality, top-K
    batch: int = 5000             # S1 queries per batch


def address_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """uid -> every number in the address (zero-stripped, up to 6 digits). Catches house
    numbers hidden behind an injected prefix number ('H.no 777 2759')."""
    arr = pa.array(df.addr_norm, pa.string())
    toks = pc.split_pattern_regex(pc.replace_substring_regex(arr, r"[^0-9]+", " "), r" ")
    flat, par = pc.list_flatten(toks), pc.list_parent_indices(toks).to_numpy()
    flat = pc.replace_substring_regex(flat, r"^0+(\d)", r"\1")
    ok = pc.and_(pc.greater(pc.utf8_length(flat), 0), pc.less_equal(pc.utf8_length(flat), 6)).to_numpy(zero_copy_only=False)
    nums = pc.cast(pc.filter(flat, pa.array(ok)), pa.int64()).to_numpy()   # int64, not Python strings (~1 GB saved on US)
    return pd.DataFrame({"uid": df.uid.to_numpy()[par[ok]], "num": nums}).drop_duplicates()


def _key_join(s1_keys: pd.DataFrame, pool_keys: pd.DataFrame, batch: int) -> list[pd.DataFrame]:
    """Exact key join in S1 batches (never materialises more than one batch of pairs)."""
    uids = s1_keys.uid.unique()
    for i in range(0, len(uids), batch):
        q = s1_keys[s1_keys.uid.isin(uids[i:i + batch])]
        j = q.merge(pool_keys, on="k", suffixes=("", "_m"))
        check_budget(where="key join")
        yield pd.DataFrame({"s1": j.uid.to_numpy(), "m": j.uid_m.to_numpy()}).drop_duplicates()


FIT_SAMPLE, TRANSFORM_CHUNK = 1_500_000, 500_000


def fit_transform_chunked(vec: TfidfVectorizer, docs: np.ndarray, seed: int = 42):
    """Fit vocabulary + IDF on a seeded sample, transform in chunks: same result shape as
    fit_transform but without materialising every n-gram of every document at once."""
    rng = np.random.default_rng(seed)
    sample = docs if len(docs) <= FIT_SAMPLE else docs[np.sort(rng.choice(len(docs), FIT_SAMPLE, replace=False))]
    vec.fit(sample)
    return sp.vstack([vec.transform(docs[i:i + TRANSFORM_CHUNK]) for i in range(0, len(docs), TRANSFORM_CHUNK)], format="csr")


class CountryBlocker:
    """All blockers for one country. The pool-side index (name and address TF-IDF, locality
    and number tables, hashed fields) is fitted ONCE; S1 queries are then answered in batches
    with `set_queries(batch)` + `run()`, so memory does not grow with the number of S1."""

    def __init__(self, pool: pd.DataFrame, country: str, cfg: BlockConfig):
        self.pool, self.country, self.cfg = pool, country, cfg
        self.compat = load_region_compat()
        self.pool_ix = pd.Index(pool.uid)
        # TF-IDF documents: every core name, plus alias names pointing back to their record
        has_alias = pool.alias_core.to_numpy() != ""
        self.doc_row = np.concatenate([np.arange(len(pool)), np.nonzero(has_alias)[0]])
        docs = np.concatenate([pool.name_core.to_numpy(), pool.alias_core.to_numpy()[has_alias]])
        phase("init: name tfidf")
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=cfg.max_df,
                                   dtype=np.float32, sublinear_tf=True)
        self.D = fit_transform_chunked(self.vec, docs)           # doc rows
        del docs
        # (a) global search: same trigram space, restricted to rare trigrams (column slice)
        df_ = np.bincount(self.D.indices, minlength=self.D.shape[1])
        self.rare = np.nonzero(df_ <= cfg.global_max_df * self.D.shape[0])[0]
        phase("init: localities/numbers")
        self.pool_locs = explode_localities(pool)
        self.pool_nums = address_numbers(pool)
        # (e) word-level address TF-IDF (postcode-free body); small because addresses are ~8 words
        phase("init: address tfidf")
        self.avec = TfidfVectorizer(analyzer="word", token_pattern=r"[^\s,]+", min_df=2, max_df=cfg.max_df,
                                    dtype=np.float32, sublinear_tf=True)
        self.PA = fit_transform_chunked(self.avec, self._body(pool))
        self._slim(pool)

    @staticmethod
    def _body(df: pd.DataFrame) -> np.ndarray:
        from normalize import PC_STRIP
        return pc.replace_substring_regex(pa.array(df.addr_norm, pa.string()), PC_STRIP, r"\1").to_numpy(zero_copy_only=False)

    @staticmethod
    def _slim(df: pd.DataFrame) -> None:
        """Keep only what the blockers need: text for names, int hashes for address fields."""
        for c in ("house_no", "street_word", "region"):
            df[c + "_h"] = hash_key(pa.array(df[c], pa.string()))
        df.drop(columns=["addr_norm", "localities", "house_no", "street_word"], inplace=True, errors="ignore")

    def set_queries(self, s1: pd.DataFrame) -> None:
        """Load one batch of S1 queries (replaces the previous batch)."""
        phase("queries")
        self.s1 = s1
        self.s1_ix = pd.Index(s1.uid)
        self.Q = self.vec.transform(s1.name_core.to_numpy())
        self.QA = self.avec.transform(self._body(s1))
        self.s1_locs = explode_localities(s1)
        self.s1_nums = address_numbers(s1)
        self._slim(s1)

    # ---- (a) global name TF-IDF
    def tfidf(self) -> pd.DataFrame:
        DT, out, c = self.D[:, self.rare].T.tocsr(), [], self.cfg
        Qr = self.Q[:, self.rare]
        for i in range(0, Qr.shape[0], 2000):
            # 4 threads: each thread holds a dense accumulator as wide as the whole pool (~100 MB on US)
            m = sp_matmul_topn(Qr[i:i + 2000], DT, top_n=c.tfidf_k * 2, threshold=c.tfidf_min_cos, n_threads=4).tocoo()
            out.append(pd.DataFrame({"s1": self.s1.uid.to_numpy()[i + m.row],
                                     "m": self.pool.uid.to_numpy()[self.doc_row[m.col]], "score": m.data}))
            check_budget(where="tfidf")
        del DT
        r = pd.concat(out).sort_values("score", ascending=False).drop_duplicates(["s1", "m"])
        return top_per_s1(r, "score", c.tfidf_k)

    def _within_locality(self, Q, D, doc_row, top_n, threshold, where) -> pd.DataFrame:
        """For each locality: the top_n documents by cosine among pool records sharing it.
        Indexed search (inverted index on localities); never all-pairs. Uses sorted arrays
        and searchsorted boundaries (no merges, no dict of arrays) to keep memory flat."""
        n_rows = len(self.pool)
        prow = self.pool_ix.get_indexer(self.pool_locs.uid)
        loc_parts, doc_parts = [self.pool_locs.loc_h.to_numpy()], [prow]   # core document of a row = the row
        if len(doc_row) > n_rows:                                           # alias documents point back to their row
            alias_doc = np.full(n_rows, -1, np.int64)
            alias_doc[doc_row[n_rows:]] = np.arange(n_rows, len(doc_row))
            a = alias_doc[prow]
            loc_parts.append(self.pool_locs.loc_h.to_numpy()[a >= 0])
            doc_parts.append(a[a >= 0])
        loc_h, doc = np.concatenate(loc_parts), np.concatenate(doc_parts)
        o = np.argsort(loc_h, kind="stable")
        loc_h, doc = loc_h[o], doc[o]
        ql, qr = self.s1_locs.loc_h.to_numpy(), self.s1_ix.get_indexer(self.s1_locs.uid)
        o = np.argsort(ql, kind="stable")
        ql, qr = ql[o], qr[o]
        uq, qstart = np.unique(ql, return_index=True)
        qend = np.append(qstart[1:], len(ql))
        dstart, dend = np.searchsorted(loc_h, uq, "left"), np.searchsorted(loc_h, uq, "right")
        out = []
        for n in range(len(uq)):
            if dend[n] == dstart[n]:
                continue
            q, d = qr[qstart[n]:qend[n]], doc[dstart[n]:dend[n]]
            m = sp_matmul_topn(Q[q], D[d].T.tocsr(), top_n=top_n, threshold=threshold, n_threads=8).tocoo()
            out.append(pd.DataFrame({"s1": self.s1.uid.to_numpy()[q[m.row]],
                                     "m": self.pool.uid.to_numpy()[doc_row[d[m.col]]], "score": m.data}))
            if n % 2000 == 0:
                check_budget(where=where)
        r = pd.concat(out).sort_values("score", ascending=False).drop_duplicates(["s1", "m"])
        return top_per_s1(r, "score", top_n)

    # ---- (c) name TF-IDF inside each shared locality
    def loc_tfidf(self) -> pd.DataFrame:
        return self._within_locality(self.Q, self.D, self.doc_row, self.cfg.loc_m, self.cfg.loc_min_cos, "loc_tfidf")

    # ---- (e) address-text TF-IDF inside each shared locality (renamed businesses)
    def addr_tfidf(self) -> pd.DataFrame:
        return self._within_locality(self.QA, self.PA, np.arange(len(self.pool)), self.cfg.addr_m,
                                     self.cfg.addr_min_cos, "addr_tfidf")

    # ---- (f) records without any address: full-vocabulary name search over that small subset
    def noaddr_tfidf(self) -> pd.DataFrame:
        has_loc = np.zeros(len(self.pool), bool)
        has_loc[self.pool_ix.get_indexer(self.pool_locs.uid.unique())] = True
        docs = np.nonzero(~has_loc[self.doc_row])[0]            # core + alias documents of address-less records
        DT, out, c = self.D[docs].T.tocsr(), [], self.cfg
        for i in range(0, self.Q.shape[0], 5000):
            m = sp_matmul_topn(self.Q[i:i + 5000], DT, top_n=c.noaddr_k * 2, threshold=c.tfidf_min_cos, n_threads=4).tocoo()
            out.append(pd.DataFrame({"s1": self.s1.uid.to_numpy()[i + m.row],
                                     "m": self.pool.uid.to_numpy()[self.doc_row[docs[m.col]]], "score": m.data}))
            check_budget(where="noaddr_tfidf")
        r = pd.concat(out).sort_values("score", ascending=False).drop_duplicates(["s1", "m"])
        return top_per_s1(r, "score", c.noaddr_k)

    # ---- (b) exact core/alias name blocks, ranked by address
    def name(self) -> pd.DataFrame:
        pk = pd.concat([self.pool.loc[self.pool.core_h != 0, ["uid", "core_h"]].rename(columns={"core_h": "k"}),
                        self.pool.loc[self.pool.alias_h != 0, ["uid", "alias_h"]].rename(columns={"alias_h": "k"})])
        qk = self.s1.loc[self.s1.core_h != 0, ["uid", "core_h"]].rename(columns={"core_h": "k"})
        out = []
        for pairs in _key_join(qk, pk, self.cfg.batch):
            if pairs.empty:
                continue
            pairs["score"] = self._address_score(pairs)
            out.append(top_per_s1(pairs, "score", self.cfg.name_m))
        return pd.concat(out, ignore_index=True)

    # ---- (c) address-key blocks, ranked by name similarity
    def hs(self) -> pd.DataFrame:
        pk = self.pool.loc[self.pool.hs_h != 0, ["uid", "hs_h"]].rename(columns={"hs_h": "k"})
        qk = self.s1.loc[self.s1.hs_h != 0, ["uid", "hs_h"]].rename(columns={"hs_h": "k"})
        return self._ranked_by_name(qk, pk, self.cfg.hs_m)

    def num_loc(self) -> pd.DataFrame:
        def keys(nums, locs):
            k = nums.merge(locs, on="uid")
            # arithmetic int64 key (wraps on overflow, fine for hashing): no per-row strings
            with np.errstate(over="ignore"):
                kk = k.loc_h.to_numpy() * np.int64(1_000_003) + k.num.to_numpy()
            return pd.DataFrame({"uid": k.uid.to_numpy(), "k": kk}).drop_duplicates()
        qk = keys(self.s1_nums, self.s1_locs)
        wanted = pd.Index(qk.k.unique())
        # pool keys in uid chunks, keeping only keys some S1 query has (block sizes stay exact)
        uids = self.pool_nums.uid.unique()
        parts = []
        for i in range(0, len(uids), 1_000_000):
            sel = uids[i:i + 1_000_000]
            k = keys(self.pool_nums[self.pool_nums.uid.isin(sel)], self.pool_locs[self.pool_locs.uid.isin(sel)])
            parts.append(k[k.k.isin(wanted)])
            check_budget(where="num_loc keys")
        pk = pd.concat(parts, ignore_index=True)
        size = pk.k.value_counts()
        pk = pk[pk.k.map(size) <= self.cfg.num_loc_max_block]
        return self._ranked_by_name(qk, pk, self.cfg.num_loc_m)

    # ---- scoring helpers
    def _ranked_by_name(self, qk, pk, m) -> pd.DataFrame:
        out = []
        for pairs in _key_join(qk, pk, self.cfg.batch):
            if pairs.empty:
                continue
            a = self.s1.name_core.to_numpy()[self.s1_ix.get_indexer(pairs.s1)]
            bi = self.pool_ix.get_indexer(pairs.m)
            pairs["score"] = name_similarity(a, self.pool.name_core.to_numpy()[bi], self.pool.alias_core.to_numpy()[bi])
            out.append(top_per_s1(pairs, "score", m))
        return pd.concat(out, ignore_index=True)

    def _address_score(self, pairs) -> np.ndarray:
        ai, bi = self.s1_ix.get_indexer(pairs.s1), self.pool_ix.get_indexer(pairs.m)
        A, B = self.s1, self.pool
        eq = lambda col: (A[col].to_numpy()[ai] == B[col].to_numpy()[bi]) & (A[col].to_numpy()[ai] != 0)
        lo = locality_overlap(pairs, self.s1_locs, self.pool_locs)
        ra, rb = A.region_h.to_numpy()[ai], B.region_h.to_numpy()[bi]
        rc = np.where((ra == 0) | (rb == 0), 0.5, (ra == rb).astype(np.float32))   # hierarchy is used by the pruner
        return (4 * eq("hs_h") + 2 * eq("house_no_h") + eq("street_word_h") + 2 * np.minimum(lo, 2) + rc).astype(np.float32)

    def cosine(self, pairs) -> np.ndarray:
        """TF-IDF cosine of the S1 core name vs the candidate's core name, for any pairs."""
        ai, bi = self.s1_ix.get_indexer(pairs.s1), self.pool_ix.get_indexer(pairs.m)
        out = np.empty(len(pairs), np.float32)
        for i in range(0, len(pairs), 1_000_000):
            out[i:i + 1_000_000] = np.asarray(self.Q[ai[i:i + 1_000_000]].multiply(self.D[bi[i:i + 1_000_000]]).sum(axis=1)).ravel()
        return out

    def run(self, blockers=BLOCKERS) -> pd.DataFrame:
        """Union of the blockers: one row per (s1, m) with, per blocker, the pair's rank in
        that blocker's list for the S1 (r_<blocker>, 0 = not proposed) and a 0/1 flag."""
        import time
        from memtrack import current_rss_gb
        parts = []
        for bi, b in enumerate(blockers):
            t = time.time()
            phase(b)
            r = getattr(self, b)()
            r = r.sort_values(["s1", "score", "m"], ascending=[True, False, True], kind="stable")
            parts.append(pd.DataFrame({"s1": r.s1.to_numpy(), "m": r.m.to_numpy(), "b": np.int8(bi),
                                       "rank": (r.groupby("s1").cumcount() + 1).to_numpy(np.int16)}))
            print(f"    {b:10s} {len(r):>10,} pairs ({len(r) / len(self.s1):.1f}/S1) {time.time()-t:6.0f}s  rss {current_rss_gb():.2f} GB", flush=True)
            del r
            check_budget(where=b)
        phase("union")
        u = pd.concat(parts, ignore_index=True)
        del parts
        order = np.lexsort((u.m.to_numpy(), u.s1.to_numpy()))
        s1v, mv, bv, rv = (u[c].to_numpy()[order] for c in ("s1", "m", "b", "rank"))
        del u, order
        new = np.ones(len(s1v), bool)
        new[1:] = (s1v[1:] != s1v[:-1]) | (mv[1:] != mv[:-1])
        pid = np.cumsum(new) - 1
        ranks = np.zeros((int(pid[-1]) + 1, len(blockers)), np.int16)
        ranks[pid, bv] = rv
        flags = pd.DataFrame({"s1": s1v[new], "m": mv[new]})
        for bi, b in enumerate(blockers):
            flags[f"r_{b}"] = ranks[:, bi]
            flags[f"b_{b}"] = (ranks[:, bi] > 0).astype(np.int8)
        flags["n_blockers"] = (ranks > 0).sum(axis=1).astype(np.int8)
        del ranks, s1v, mv, bv, rv, pid, new
        phase("cosine")
        flags["cos"] = self.cosine(flags)
        return flags


# ================================================================ driver
S1_BATCH = 40_000   # queries per batch; 60k exceeded the budget on the 6M-record US pool


def apply_overrides(df: pd.DataFrame, ov: pd.DataFrame | None) -> pd.DataFrame:
    """Replace name-derived columns for the uids in `ov` (index uid). Used by the stress slice."""
    if ov is None or df.empty:
        return df
    mask = df.uid.isin(ov.index).to_numpy()
    for col in ov.columns:
        if col in df.columns:
            df.loc[mask, col] = ov.loc[df.uid.to_numpy()[mask], col].to_numpy()
    return df


def _block_country(split, c, s1_uids, out_dir, cfg, blockers, s1_batch, pool_overrides) -> None:
    """One country, run in its own child process so all of its memory returns to the OS.
    Writes {out_dir}/{c}.parquet (one row group per S1 batch) and {out_dir}/{c}.certain.parquet."""
    import time
    t = time.time()
    others = [s for s in available_sources(split) if s != 1]
    phase(f"{c}: certain")
    keys = ["uid", "name_norm", "hs_h"]
    s1k = load_side(split, [1], c, uids=s1_uids, columns=keys)
    if s1k.empty:
        return
    cert = certain_matches(s1k, apply_overrides(load_side(split, others, c, columns=keys), pool_overrides))
    c_uids = s1k.uid.to_numpy()
    del s1k
    phase(f"{c}: load pool")
    pool = load_side(split, others, c, columns=RUN_COLS, exclude=cert.m.to_numpy())  # certain records leave the pool
    pool = apply_overrides(pool, pool_overrides)
    print(f"  [{c}] S1 {len(c_uids):,}  pool {len(pool):,}  certain {len(cert):,}", flush=True)
    cb = CountryBlocker(pool, c, cfg)
    writer, n_c = None, 0
    for i in range(0, len(c_uids), s1_batch):
        cb.set_queries(load_side(split, [1], c, uids=c_uids[i:i + s1_batch], columns=RUN_COLS))
        tbl = pa.Table.from_pandas(cb.run(blockers).assign(country=c), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_dir / f"{c}.parquet", tbl.schema, compression="zstd")
        writer.write_table(tbl)
        n_c += tbl.num_rows
        del tbl
    writer.close()
    cert.assign(country=c).to_parquet(out_dir / f"{c}.certain.parquet", index=False)   # written last = country complete
    print(f"  [{c}] {n_c:,} candidates ({n_c/len(c_uids):.1f}/S1) in {time.time()-t:.0f}s", flush=True)


def _block_country_logged(*args):
    """Child-process entry: logs this country's own time and peak RSS to notes/resource_log.tsv."""
    from memtrack import PeakSampler, stage
    *rest, name = args
    with stage(f"block_{name}_{rest[1]}"), PeakSampler() as ps:
        _block_country(*rest)
    print(f"  [{rest[1]}] PEAKS {ps.report()}", flush=True)


def run_blocking(split: str, s1_uids: np.ndarray, name: str, cfg: BlockConfig = BlockConfig(),
                 blockers=BLOCKERS, countries_=None, s1_batch: int = S1_BATCH,
                 pool_overrides: pd.DataFrame | None = None, resume: bool = False):
    """Block every country for the given S1 uids. Per country (in a child process): certain
    matches on key columns, fit the pool index once, then query S1 in batches.
    Output: artifacts/cands_{name}/{country}.parquet (one row group per S1 batch, so every row
    group holds complete candidate lists) and artifacts/certain_{name}.parquet.
    Returns (candidates directory, certain pairs); candidates are never loaded whole."""
    import multiprocessing as mp
    import shutil
    from data import ARTIFACTS_DIR
    out_dir = ARTIFACTS_DIR / f"cands_{name}"
    if not resume:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in countries_ or countries(split):
        if resume and (out_dir / f"{c}.parquet").exists() and (out_dir / f"{c}.certain.parquet").exists():
            print(f"  [{c}] already blocked, reusing", flush=True)
            continue
        proc = mp.get_context("fork").Process(target=_block_country_logged,
                                              args=(split, c, s1_uids, out_dir, cfg, blockers, s1_batch, pool_overrides, name))
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"blocking failed for {c} (exit code {proc.exitcode})")
    parts = sorted(out_dir.glob("*.certain.parquet"))
    certs = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    for f in parts:
        f.unlink()
    certs.to_parquet(ARTIFACTS_DIR / f"certain_{name}.parquet", index=False)
    return out_dir, certs


def candidate_files(path) -> list:
    """Candidate parquet files of a blocking output (a directory of per-country files, or one file)."""
    import pathlib
    path = pathlib.Path(path)
    return sorted(path.glob("*.parquet")) if path.is_dir() else [path]


def blocking_report(cands: pd.DataFrame, certs: pd.DataFrame, truth: pd.DataFrame, s1_uids, blockers=BLOCKERS,
                    pool_size_by_s1: pd.Series | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-blocker recall and unique contribution, and union completeness (certain matches
    counted as found). Recall is over ALL true pairs of the evaluated S1 entities."""
    from evaluate import candidate_stats
    truth = truth[truth.s1.isin(s1_uids)][["s1", "m"]]
    t = truth.merge(cands, on=["s1", "m"], how="left").fillna(0)
    t["certain"] = truth.merge(certs.assign(c=1)[["s1", "m", "c"]], on=["s1", "m"], how="left").c.fillna(0).to_numpy()
    rows = []
    cols = [f"b_{b}" for b in blockers]
    for b in blockers:
        others = [c for c in cols if c != f"b_{b}"]
        only = (t[f"b_{b}"] == 1) & (t[others].sum(axis=1) == 0) & (t.certain == 0)
        rows.append({"blocker": b, "pairs": int(cands[f"b_{b}"].sum()), "pairs_per_S1": cands[f"b_{b}"].sum() / len(s1_uids),
                     "recall": t[f"b_{b}"].mean(), "unique_recall": only.mean()})
    rows.append({"blocker": "certain", "pairs": len(certs), "pairs_per_S1": len(certs) / len(s1_uids),
                 "recall": t.certain.mean(), "unique_recall": ((t.certain == 1) & (t[cols].sum(axis=1) == 0)).mean()})
    union = pd.concat([cands[["s1", "m"]], certs[["s1", "m"]]], ignore_index=True)
    st = candidate_stats(union, truth, s1_uids, pool_size_by_s1)
    return pd.DataFrame(rows).set_index("blocker"), st
