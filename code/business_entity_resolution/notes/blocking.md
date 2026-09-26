# Stage 3 / 3b — Blocking and pruning

Code:
- `src/blocking.py` holds the blockers, the per-country child processes and the report.
- `src/prune.py` holds the pruner features, OOF LightGBM, top-2 per record, the adaptive cutoff and the trade-off table.
- `src/stage3_run.py` runs the val steps (`block | report | train | prune | holdout`).
- `src/stress_slice.py` runs the unknown-token stress slice.
- `src/submit_baseline.py` produces the baseline test submission.
- Tests: `tests/test_prune.py` (62 tests in the whole suite).

Development ran on the dev split (150k S1). **All numbers below are on the full val split (441,365 S1, 1,527,534
true pairs) against the full pool (10.3M train S2+S3)**, not the test-like pool.

## Design: indexed lookups only, never all-pairs
Per country and in batches of 40k S1 queries. Each country runs in its own child process, so its memory returns to
the OS before the next one starts.

0. **Certain matches first:** same `name_norm` and same house number + street word. A record claimed by two S1 is
   left in the pool. These pairs are saved (`certain_val.parquet`) and their records **leave the pool**. This
   covers 297,208 pairs, 19.4% of all true pairs.
1. **(a) `tfidf`:** char_wb 3-gram TF-IDF on the transliterated core name (alias names are indexed as extra documents
   pointing back to their record). Global sparse top-30, cosine ≥ 0.3, using only trigrams in ≤ 0.5% of names
   (about 10× faster than the full vocabulary).
2. **(b) `name`:** exact core-name blocks (S1 core name vs pool core **and alias** names), ranked by an address score
   (4·same house+street, 2·house no., 1·street word, 2·shared locality, 1·region), top-20. Frequent names are never
   dropped; the cap is per S1.
3. **(c) Address-key blocks, ranked by name similarity** (to catch renamed businesses):
   - `loc_tfidf`: name TF-IDF top-20 within each shared locality;
   - `hs`: house number + street word blocks, top-20 by token-set ratio;
   - `num_loc`: *any* address number + locality, blocks ≤ 200 records, top-20. This catches injected prefix numbers
     like "H.no 777 2759".
4. **(e) `addr_tfidf`:** word TF-IDF on the address text, top-10 within each shared locality. This targets renamed
   businesses at copied addresses ("Shakti Agro Limited" → "Wexveo" at the same "C/o … Beside Zudio …").
5. **(f) `noaddr_tfidf`:** full-vocabulary name TF-IDF over just the pool records **without any address**, top-10.
6. **(d) Aliases:** S2/S3 "X fka/dba Y" names are indexed as extra documents and extra name keys in (a), (b) and
   (f). S1 has no aliases, so there are no alias queries.

## Blocking recall on val (full pool)

| blocker | pairs per S1 | recall (share of all true pairs) | found only by this blocker |
|---|---:|---:|---:|
| certain | 0.67 | 19.4% | 19.4% |
| (a) tfidf | 22.3 | 26.1% | 0.34% |
| (b) name | 7.6 | 39.7% | 1.21% |
| (c) loc_tfidf | 17.1 | 59.6% | 0.74% |
| (c) hs | 6.1 | 41.2% | **2.56%** |
| (c) num_loc | 9.1 | 49.9% | 0.32% |
| (e) addr_tfidf | 9.9 | 58.5% | 1.03% |
| (f) noaddr_tfidf | 10.0 | 4.1% | 0.77% |

| union (incl. certain) | India | US | **all** |
|---|---:|---:|---:|
| pair completeness | 97.72% | 97.53% | **97.60%** |
| candidates per S1: mean / median / p90 | 76.8 / 77 / 99 | 65.1 / 65 / 86 | 69.8 / 69 / 93 |
| entities with all true pairs found | 92.6% | 92.0% | 92.2% |
| reduction ratio (1 − cands / Σ same-country pool) | 0.9999814 | 0.9999895 | 0.9999870 |

**The union is 0.4 points short of the 98% target.** Doubling every cap during development added only 1.2 points at
1.7× the candidates. What remains is mostly:
- candidates with no address and a crowded, common name ("Shri sree technologies pvt ltd |" with nothing after it),
  which are inherently ambiguous;
- renamed businesses whose address was also rewritten.

## Pruner (3b)
- **Model:** LightGBM, 200 trees, trained out-of-fold on dev-train (120k S1, 8.32M candidate pairs, 3.9% positive;
  5 folds by S1 entity) and refitted on all of it.
- **Features:** 13 fast pair features (TF-IDF cosine, token-set and plain ratio of core names, legal-form,
  house-no., street-word, locality and learned-region compatibility, cosine rank and gap, number of blockers,
  candidate source) plus the 6 per-blocker ranks, which blocking provides for free.
- **Quality:** OOF AUC **0.9975**, AP **0.961**. Top gain comes from n_blockers (69%), name ratio (7%), house-no.
  agreement (5%), the address-TF-IDF rank, region compatibility and legal-form compatibility.
- **One-to-one:** each S2/S3 record is kept only for its **top-2** S1 entities.
- **Adaptive cutoff:** p ≥ floor, p ≥ best − margin, at most max_k per S1. Empty lists are allowed. Certain pairs
  are always kept.

### Trade-off (val, full pool) — selected rows
| max_k | floor | margin | avg cands | median | p90 | empty lists | pair completeness | F0.5 upper bound |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.05 | 0.3 | 3.10 | 3 | 5 | 1.05% | 83.0% | 0.9529 |
| 8 | 0.05 | 0.5 | 3.45 | 3 | 6 | 1.05% | 89.5% | 0.9689 |
| 8 | 0.05 | 0.7 | 3.73 | 4 | 6 | 1.05% | 92.4% | 0.9776 |
| 8 | 0.10 | 1.0 | 4.04 | 4 | 6 | 2.06% | 94.4% | 0.9817 |
| 8 | 0.02 | 0.9 | 4.37 | 4 | 7 | 0.29% | 94.95% | 0.9850 |
| **8** | **0.05** | **1.0** | **4.47** | **4** | **7** | **1.05%** | **95.37%** | **0.9852** |
| 8 | 0.02 | 1.0 | 5.13 | 5 | 8 | 0.29% | 96.10% | 0.9878 |
| 8 | 0.01 | 1.0 | 5.64 | 6 | 8 | 0.12% | 96.41% | 0.9888 |

The full 90-row grid is in `artifacts/prune_tradeoff_val.csv`.

**Recommended operating point: `max_k=8, floor=0.05, margin=1.0`.** It is the smallest set that clears 95%
completeness: an average of **4.47** candidates per S1 (median 4, p90 7), **95.37%** completeness, and an F0.5 upper
bound of 0.985 for a perfect matcher. The reduction ratio is 0.9999992. The margin plays little role here; floor and
top-2 per record do the work. Pruned candidates are saved at `artifacts/pruned_val.parquet` and, scored out-of-fold
for Stage 4 training, at `artifacts/pruned_dev_train.parquet`.

## Test-like unknown-token stress slice
- **Setup:** hiding 11.7% of the lexicon's **content-word** mappings raises the unknown-name share of val Indic
  names from 5.1% to **17.0%**, matching test's 17.3%. Legal words such as private and limited stay mapped. All
  752,869 Indic-script India pool records are re-normalised with the reduced lexicon, and the 54,576 val India S1
  entities that own one are re-blocked.

| run | completeness on the slice (110,384 true pairs with an Indic-script candidate) | same S1, all their true pairs | certain on the slice |
|---|---:|---:|---:|
| normal lexicon | 98.64% | 97.17% | 35.8% |
| stressed (test-like unknowns) | **96.90%** | 96.26% | 29.5% |

Test-only vocabulary costs about 1.7 points of blocking completeness on Indic-script records. Address and locality
blockers keep most of them, because renamed words do not change the address.

## Baseline submission (certain + pruner top-1)
- **Rule:** certain pairs, plus each S1's best candidate with p ≥ t, where every record goes to the S1 that scores
  it highest.
- **Threshold on val:** the curve is flat around the optimum: 0.7724 at 0.5, **0.7734 at 0.6**, 0.7732 at 0.7,
  0.7600 at 0.9 and 0.6340 at 0.99. So t = 0.6 is used (certain alone scores 0.3869). A higher "safe" threshold
  only costs recall here, because top-1 plus one-to-one is already very precise.
- **Test run** (`python src/submit_baseline.py test`):
  - input is test S1 plus **S3 only — `dataset/test/test_source2.tsv` is still missing**;
  - 1,732,544 S1 (India 809,986, US 663,106, France 259,452); 627,165 certain pairs;
  - blocking gives 121.7M candidates (France 21.6M, India 63.3M, US 36.7M);
  - pruned to **4,948,621 candidates (2.86 per S1)**, lower than val because the pool is roughly half without S2;
  - 1,929,596 matches (1.11 per S1), 268,952 empty lists;
  - `validate_submission.py` gives **PASS**. Every ID exists in the test files, every match is a candidate, and no
    S2/S3 record is used twice.
- **France:** 180,619 of 259,452 French S1 entities have a certain match, so French S3 records are often exact
  copies. Blocking there took 432 s at 3.62 GB.

## Timings and peak RAM (budget 5 GB, one process at a time)
| step | time | peak RSS |
|---|---:|---:|
| val blocking, India child (176,638 S1 vs 4.04M pool) | 483 s | 3.98 GB |
| val blocking, US child (264,727 S1 vs 5.98M pool) | 629 s | 4.68 GB |
| val blocking report | 28 s | 3.47 GB |
| pruner OOF training (8.32M pairs) | 335 s | 4.28 GB |
| val scoring (30.8M pairs) | 137 s | 2.70 GB |
| trade-off grid (90 points) | 28 s | 3.45 GB |
| stress slice re-blocking | 225 s | 4.60 GB |
| test blocking: France / India / US | 432 / 1,497 / 485 s | 3.62 / 2.83 / 2.59 GB |
| test scoring (121.7M pairs, streamed, p ≥ floor kept) | 461 s | 2.33 GB |
| test end-to-end after blocking (score + prune + write) | 519 s | 3.21 GB |

### What had to change to stay under 5 GB
| problem | fix |
|---|---|
| uncapped (house number, locality) joins exploded (OOM) | size blocks first; cap address blocks at ≤ 200; batch the S1 joins |
| strings kept for every pool column | int64 hashes for address fields; drop the address text after use |
| per-row string keys for `num_loc` | arithmetic int64 keys |
| number tokens as Python strings (~1 GB on US) | int64 |
| one-shot TF-IDF fit | fit vocabulary on a 1.5M sample, then transform in chunks |
| outer merges for the union | union by sorting into a rank matrix |
| 8 threads × pool-wide accumulators in `sparse_dot_topn` | 4 threads for the two whole-pool searches |
| countries sharing one process | one child process per country |
| pruner feature matrix copied per fold | bin the LightGBM dataset once; train folds on subsets |
| test scoring held every score (6.36 GB) | stream row groups and keep only p ≥ floor (proven exact on val) |

## Country hold-out (stand-in for unseen France)
The pruner is trained on ONE country's dev-train and applied to the OTHER country's val candidates, using the same
operating point and baseline threshold (`python src/stage3_run.py holdout`, 626 s, 4.41 GB):

| pruner trained on → evaluated on | avg cands | median | pair completeness | baseline F0.5 |
|---|---:|---:|---:|---:|
| **US → India (hold-out)** | 5.19 | 5 | 91.96% | 0.7090 |
| both → India (in-country) | 4.72 | 5 | 94.72% | 0.7382 |
| **India → US (hold-out)** | 4.64 | 5 | 94.82% | 0.7720 |
| both → US (in-country) | 4.31 | 4 | 95.80% | 0.7969 |

Transferring across countries costs 2.8 points of completeness and 2.9 points of F0.5 into India, and 1.0 and 2.5
points into US. A French pruner will be somewhat weaker than the in-country numbers suggest. Stage 4/5 should
check which features transfer badly (per-country calibration of `n_blockers` and the address features is the
first suspect).
