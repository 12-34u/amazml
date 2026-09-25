"""Vectorised text normalisation built on Arrow compute (RE2 regex, no Python loops).

Stage 0 needs only `basic_key`. Stage 2 adds legal-suffix separation and address
parsing to this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

# Combining diacritics produced by NFKD (é -> e + U+0301). Only this block is removed,
# because Devanagari vowel signs are also "marks" (\p{M}) and must be kept.
_DIACRITICS = r"[\x{0300}-\x{036f}]"
_NON_WORD = r"[^\p{L}\p{M}\p{N}]+"


def basic_key(arr: pa.Array | pa.ChunkedArray) -> pa.ChunkedArray:
    """Language-agnostic comparison key: NFKD, drop accents, lowercase, '&'->'and',
    all punctuation to single spaces, trimmed. Works for any script."""
    x = pc.utf8_normalize(arr, form="NFKD")
    x = pc.replace_substring_regex(x, _DIACRITICS, "")
    x = pc.utf8_lower(x)
    x = pc.replace_substring(x, "&", " and ")
    x = pc.replace_substring_regex(x, _NON_WORD, " ")
    return pc.utf8_trim_whitespace(x)


def hash_key(arr: pa.Array | pa.ChunkedArray, chunk: int = 1_000_000) -> np.ndarray:
    """64-bit hash of a string column, for cheap joins and value counts.
    Returned as int64 (uint64 bits reinterpreted) so Arrow/parquet filters accept it.
    Empty strings hash to 0, so callers can drop them with `h != 0`."""
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    out = np.empty(len(arr), dtype=np.uint64)
    for i in range(0, len(arr), chunk):
        s = pd.Series(arr.slice(i, chunk), dtype=pd.ArrowDtype(pa.string()))
        out[i:i + chunk] = pd.util.hash_pandas_object(s, index=False).to_numpy()
    out[pc.equal(arr, "").to_numpy(zero_copy_only=False)] = 0
    return out.view(np.int64)


ADDR_PREFIX_RE = r"(\d+\s+[^\s\d]+)"  # first "house number + next word", e.g. "994 miller"


def addr_prefix_hash(addr_key: pd.Series) -> np.ndarray:
    """Hash of the first number+word in a normalised address (0 when there is none).
    Survives component reordering ("Crossville, 994 Miller Ave" vs "994 MILLER AVENUE")."""
    pref = addr_key.str.extract(ADDR_PREFIX_RE, expand=False).fillna("")
    return hash_key(pa.array(pref, type=pa.string()))
