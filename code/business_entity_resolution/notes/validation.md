# Stage 1 — Evaluation and validation protocol

## Metric (`src/evaluate.py`)
- `f05(pred, truth)` is the reference implementation for one entity. `per_entity_scores` / `macro_f05` are the
  vectorised versions over long pair tables `(s1, m)`.
- Every S1 entity in the evaluation set counts. A singleton scores 1.0 when its prediction is empty and 0.0
  otherwise. A missing prediction row counts as an empty list. Duplicate pairs are ignored.
- `score_report` breaks the score down into overall, singletons, non-singletons and per country.
- `candidate_stats` reports what the Stage 3b pruning rule asks for:
  - pair completeness
  - average, median and p90 candidates per S1 (empty lists count as 0)
  - share of empty lists
  - entity full recall
  - reduction ratio against |S1| × |S2+S3 in the same country|
- CLI: `python src/evaluate.py --pred <matching_results-style.tsv> --s1-subset dev_val` scores a submission-format
  file against the train labels.
- `log_experiment(...)` appends a row to `notes/experiments.md`.

Tests: `python -m pytest tests -q` gives **20 passed**. They cover:
- the official example: [S2-00047, S2-00193, S3-00812] against [S2-00047, S3-00812] gives **0.714285…**
- singleton and empty-prediction edge cases, and missing rows
- the vectorised version against the reference on 400 random entities
- candidate metrics
- the TSV round trip
- the split invariants

## Splits (`src/splits.py` → `artifacts/splits.parquet`)
Everything is split **by S1 entity** with seed 42. Each S2/S3 record belongs to at most one S1, so an entity's true
candidates always stay on its side. S2/S3 are never subsampled: every evaluation searches the full 10.3M-record
pool of its country, as the test set will.

| set | S1 | US | India | singleton share | mean matches | true pairs |
|---|---:|---:|---:|---:|---:|---:|
| all_train | 2,206,821 | 1,323,633 | 883,188 | 5.58% | 3.461 | 7,638,365 |
| train (80%) | 1,765,456 | 1,058,906 | 706,550 | 5.58% | 3.461 | 6,110,831 |
| val (20%) | 441,365 | 264,727 | 176,638 | 5.58% | 3.461 | 1,527,534 |
| dev | 150,000 | 89,969 | 60,031 | 5.55% | 3.460 | 519,062 |
| dev_train | 120,033 | 71,961 | 48,072 | 5.59% | 3.457 | 414,893 |
| **dev_val** | **29,967** | 18,008 | 11,959 | 5.42% | 3.476 | 104,169 |

- The 80/20 split is stratified by country × match count (0–5, 6+), so val keeps the country mix and the singleton
  share.
- `fold` 0–4 inside the train part is for group-aware CV in Stage 4. Folds are by S1 entity, so all of an entity's
  candidate pairs share one fold.
- **Iteration loop:** fit on `dev_train`, score on `dev_val`. Final checks fit on `train` and score on `val`.
- **Country hold-out** (`country_holdout("US")` / `("India")`): fit on the dev entities of one country and score
  the dev entities of the other. The country list comes from the data and is never hard-coded.
- Caveat: records owned by S1 entities outside the evaluated set stay in the pool as extra distractors, and the
  one-to-one reassignment cannot see those owners. Val and dev precision are therefore slightly pessimistic
  compared with the test setting, where every S1 competes.

## Baselines (`src/baselines.py`, logged in `experiments.md`)

| rule | eval set | pairs per S1 | pair completeness | macro F0.5 | mean precision | India dev | US dev |
|---|---|---:|---:|---:|---:|---:|---:|
| empty | dev_val / val | 0 | 0 | 0.0542 / 0.0558 | – | 0.0561 | 0.0551 |
| exact name | dev_val / val | 11.2 / 11.0 | 25.8% / 25.6% | 0.3552 / 0.3541 | 0.556 | 0.2922 | 0.3957 |
| exact name + house-no.&street-word | dev_val / val | 0.49 / 0.48 | 14.1% / 13.8% | 0.3047 / 0.3034 | **0.9998** | 0.2204 | 0.3571 |

What the baselines show:
- **dev_val tracks val to within 0.0015 F0.5** on every rule, so iterating on the dev subset is safe.
- **Exact name gives 11 pairs per S1 but finds only 26% of true pairs.** It still wrongly matches **40% of
  singletons**, because a same-name business exists elsewhere.
- **Exact name + the same "house number + street word" is almost never wrong: 211,422 of 211,460 pairs are correct
  on val.** It makes a near-free high-precision anchor for pruning and features. The low score comes entirely from
  recall (14%), which is the job of the blockers and the model.
- India is harder than the US on every rule (0.29 vs 0.40), consistent with the transliteration and noise found in
  Stage 0.

## Resources
`splits.py` 11 s / 0.63 GB · `baselines.py --full` 134 s / 3.08 GB · tests 11 s.
