"""Paths and loaders shared by every stage.

Raw challenge files are TSVs that must be read as plain strings (tab separator, no
quoting, no NA coercion). `prepare.py` reads them once and writes compact parquet to
artifacts/. Everything downstream reads the parquet, and only the columns it needs.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# code/business_entity_resolution/src -> student_resource/
PROJECT_DIR = Path(__file__).resolve().parents[1]
RESOURCE_DIR = PROJECT_DIR.parents[1]
DATA_DIR = Path(os.environ.get("BER_DATA_DIR", RESOURCE_DIR / "dataset"))
ARTIFACTS_DIR = Path(os.environ.get("BER_ARTIFACTS_DIR", PROJECT_DIR / "artifacts"))
NOTES_DIR = PROJECT_DIR / "notes"

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def raw_path(split: str, source: int) -> Path:
    return DATA_DIR / split / f"{split}_source{source}.tsv"


def read_tsv_arrow(path: Path | str) -> pa.Table:
    """Read a challenge TSV as an Arrow table of non-null strings.

    quote_char=False: quotes are literal data in names such as `Joe's "Best" Deli`.
    """
    with open(path, encoding="utf-8") as f:
        cols = f.readline().rstrip("\n").split("\t")
    return pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={c: pa.string() for c in cols},
            strings_can_be_null=False, quoted_strings_can_be_null=False,
        ),
    )


def prepared_path(split: str, source: int) -> Path:
    return ARTIFACTS_DIR / f"{split}_s{source}.parquet"


def available_sources(split: str) -> list[int]:
    """Sources that exist for a split. The test S2 file may be absent."""
    return [s for s in (1, 2, 3) if prepared_path(split, s).exists()]


def load(split: str, source: int, columns: list[str] | None = None, filters=None) -> pd.DataFrame:
    """Load a prepared source. `country` comes back as a pandas categorical."""
    return pq.read_table(prepared_path(split, source), columns=columns, filters=filters).to_pandas()


def load_others(split: str, columns: list[str] | None = None, filters=None) -> pd.DataFrame:
    """Concatenate the prepared S2 and S3 tables (whichever exist) for a split."""
    parts = [load(split, s, columns, filters) for s in available_sources(split) if s != 1]
    df = pd.concat(parts, ignore_index=True)
    if "country" in df:  # concat of categoricals with different categories gives object dtype
        df["country"] = df["country"].astype("category")
    return df


def load_truth(columns: list[str] | None = None) -> pd.DataFrame:
    """Long-format train labels: one row per true pair (s1_id, matched_id, s1_uid, m_uid)."""
    return pd.read_parquet(ARTIFACTS_DIR / "train_truth.parquet", columns=columns)
