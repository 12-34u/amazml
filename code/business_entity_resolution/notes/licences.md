# Licence register

The challenge requires every **model** to be MIT or Apache-2.0 licensed and to have at most 8B parameters.
This file records every model and every third-party library the pipeline uses. It is the source for the licence
table in `Documentation_template.md`.

## Models and model-like components

| component | used for | version | licence | parameters | data it uses |
|---|---|---|---|---|---|
| indic-transliteration (sanscript) | rule-based Indic script → Latin (IAST) transliteration in `normalize.py` | 2.3.82 | **MIT** | 0 (character tables, not a learned model) | none downloaded; transliteration tables ship with the package |
| learned transliteration lexicon | transliterated token → English token (`artifacts/translit_lexicon.parquet`) | built by `normalize_build.py` | our code | 583 token pairs | train-split true pairs only |
| learned region resources | region-level address components and spellings (`regions.parquet`, `region_map.parquet`) | built by `normalize_build.py` | our code | – | provided source files; the map uses train-split true pairs only |

No neural models are used yet. Candidates for later stages: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(Apache-2.0, ~118M parameters) and LightGBM (MIT). These will be added here when they are used.

## Libraries (Python 3.12)

| library | version | licence | role |
|---|---|---|---|
| indic-transliteration | 2.3.82 | MIT | transliteration |
| ↳ regex | 2026.9.10 | Apache-2.0 AND CNRI-Python | dependency of indic-transliteration |
| ↳ typer | 0.27.2 | MIT | dependency (its CLI, unused by us) |
| ↳ toml | 0.10.2 | MIT | dependency |
| ↳ roman | 5.2 | ZPL-2.1 | dependency (Roman numerals, unused by us) |
| ↳ tqdm | 4.70.1 | MPL-2.0 AND MIT | dependency |
| ↳ backports.functools_lru_cache | 2.0.0 | MIT | dependency |
| pandas | 3.0.6 | BSD-3-Clause | data frames |
| numpy | 2.5.3 | BSD-3-Clause (and others) | arrays |
| pyarrow | 25.0.1 | Apache-2.0 | parquet and vectorised string compute |
| rapidfuzz | 3.14.6 | MIT | string similarity |
| scikit-learn | 1.9.1 | BSD-3-Clause | metrics (later: TF-IDF) |
| matplotlib | 3.11.2 | PSF-based | notebook plots |
| tabulate | 0.10.0 | MIT | markdown tables in reports |
| pytest | 9.1.1 | MIT | tests |

No external data or lookups are used. Every learned resource comes from the provided `dataset/` files.
