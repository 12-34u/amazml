"""Convert the raw TSVs into compact parquet in artifacts/. This runs once and is the first pipeline step.

Per source file it writes artifacts/{split}_s{n}.parquet with these columns:
  entity_id, business_name, business_address  raw strings
  country                                     dictionary-encoded (pandas categorical)
  uid                                         int64 = source * 1e10 + numeric id, for cheap joins
  name_key, addr_key                          basic_key() normalised text
  name_h, addr_h                              int64 hashes of the keys (0 = empty)
Train labels become artifacts/train_truth.parquet in long format (s1_id, matched_id, s1_uid, m_uid).

Usage:  python src/prepare.py            (skips outputs that already exist)
        python src/prepare.py --force
"""
from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data import ARTIFACTS_DIR, prepared_path, raw_path, read_tsv_arrow
from memtrack import stage
from normalize import basic_key, hash_key

UID_BASE = 10**10


def id_to_uid(ids: pa.ChunkedArray | pa.Array) -> pa.Array:
    """'S2-681193310' -> 2*1e10 + 681193310. The numeric part must be < 1e10."""
    src = pc.cast(pc.utf8_slice_codeunits(ids, 1, 2), pa.int64())
    num = pc.cast(pc.utf8_slice_codeunits(ids, 3), pa.int64())
    assert pc.max(num).as_py() < UID_BASE, "entity id numeric part too long for uid packing"
    return pc.add(pc.multiply(src, UID_BASE), num)


def prepare_source(split: str, source: int) -> None:
    t = read_tsv_arrow(raw_path(split, source))
    assert pc.all(pc.starts_with(t["entity_id"], f"S{source}-")).as_py(), "unexpected id prefix"
    name_key, addr_key = basic_key(t["business_name"]), basic_key(t["business_address"])
    out = pa.table({
        "entity_id": t["entity_id"],
        "business_name": t["business_name"],
        "business_address": t["business_address"],
        "country": pc.dictionary_encode(t["country"]),
        "uid": id_to_uid(t["entity_id"]),
        "name_key": name_key,
        "addr_key": addr_key,
        "name_h": pa.array(hash_key(name_key)),
        "addr_h": pa.array(hash_key(addr_key)),
    })
    pq.write_table(out, prepared_path(split, source), compression="zstd")
    print(f"  {split} S{source}: {out.num_rows:,} rows -> {prepared_path(split, source).name}")


def prepare_truth() -> None:
    """Explode the comma-separated label column in Arrow (list_flatten), no Python lists."""
    t = read_tsv_arrow(raw_path("train", 1).parent / "train_ground_truth.tsv")
    lists = pc.split_pattern(t["matched_entity_ids"], ",")
    flat = pc.list_flatten(lists)
    s1 = pc.take(t["source1_entity_id"], pc.list_parent_indices(lists))
    keep = pc.not_equal(pc.utf8_trim_whitespace(flat), "")
    s1, flat = pc.filter(s1, keep), pc.utf8_trim_whitespace(pc.filter(flat, keep))
    out = pa.table({"s1_id": s1, "matched_id": flat, "s1_uid": id_to_uid(s1), "m_uid": id_to_uid(flat)})
    pq.write_table(out, ARTIFACTS_DIR / "train_truth.parquet", compression="zstd")
    print(f"  truth: {t.num_rows:,} S1 rows -> {out.num_rows:,} true pairs")


def main(force: bool = False) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with stage("prepare"):
        for split in ("train", "test"):
            for s in (1, 2, 3):
                if not raw_path(split, s).exists():
                    print(f"  {split} S{s}: raw file missing, skipped")
                    continue
                if force or not prepared_path(split, s).exists():
                    prepare_source(split, s)
        if force or not (ARTIFACTS_DIR / "train_truth.parquet").exists():
            prepare_truth()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
