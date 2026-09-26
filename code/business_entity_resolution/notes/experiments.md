# Experiment log

One row per run. F0.5 is macro F0.5 over every S1 entity in the eval set (singletons included). Hold-out columns: train on one country, validate on the other.

| date | run | change | eval set | blocking recall | avg cands | val F0.5 | hold-out US→India | hold-out India→US | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-25 | B-empty | rule baseline: empty | dev_val [full pool] | 0.0000 | 0.0000 | 0.0542 | 0.0561 | 0.0551 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-empty | rule baseline: empty | val [full pool] | 0.0000 | 0.0000 | 0.0558 | 0.0561 | 0.0551 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-exact_name | rule baseline: exact_name | dev_val [full pool] | 0.2580 | 11.1730 | 0.3552 | 0.2922 | 0.3957 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-exact_name | rule baseline: exact_name | val [full pool] | 0.2560 | 11.0090 | 0.3541 | 0.2922 | 0.3957 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-name+addr_prefix | rule baseline: name+addr_prefix | dev_val [full pool] | 0.1407 | 0.4892 | 0.3047 | 0.2204 | 0.3571 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-name+addr_prefix | rule baseline: name+addr_prefix | val [full pool] | 0.1384 | 0.4791 | 0.3034 | 0.2204 | 0.3571 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-empty | rule baseline: empty | dev_val [full pool] | 0.0000 | 0.0000 | 0.0542 | 0.0561 | 0.0551 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-empty | rule baseline: empty | val [full pool] | 0.0000 | 0.0000 | 0.0558 | 0.0561 | 0.0551 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-exact_name | rule baseline: exact_name | dev_val [full pool] | 0.2580 | 11.1730 | 0.3552 | 0.2922 | 0.3957 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-exact_name | rule baseline: exact_name | val [full pool] | 0.2560 | 11.0090 | 0.3541 | 0.2922 | 0.3957 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-name+addr_prefix | rule baseline: name+addr_prefix | dev_val [full pool] | 0.1407 | 0.4892 | 0.3047 | 0.2204 | 0.3571 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-name+addr_prefix | rule baseline: name+addr_prefix | val [full pool] | 0.1384 | 0.4791 | 0.3034 | 0.2204 | 0.3571 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-empty | rule baseline: empty | dev_val [testlike pool] | 0.0000 | 0.0000 | 0.0542 | 0.0561 | 0.0551 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-empty | rule baseline: empty | val [testlike pool] | 0.0000 | 0.0000 | 0.0558 | 0.0561 | 0.0551 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-exact_name | rule baseline: exact_name | dev_val [testlike pool] | 0.2580 | 1.8496 | 0.4248 | 0.3337 | 0.4634 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-exact_name | rule baseline: exact_name | val [testlike pool] | 0.2560 | 3.5429 | 0.3957 | 0.3337 | 0.4634 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-name+addr_prefix | rule baseline: name+addr_prefix | dev_val [testlike pool] | 0.1407 | 0.4891 | 0.3047 | 0.2204 | 0.3571 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-name+addr_prefix | rule baseline: name+addr_prefix | val [testlike pool] | 0.1384 | 0.4791 | 0.3034 | 0.2204 | 0.3571 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-certain | rule baseline: certain | dev_val [full pool] | 0.1964 | 0.6833 | 0.3875 | 0.3168 | 0.4332 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-certain | rule baseline: certain | val [full pool] | 0.1945 | 0.6735 | 0.3869 | 0.3168 | 0.4332 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-certain | rule baseline: certain | dev_val [testlike pool] | 0.1964 | 0.6829 | 0.3875 | 0.3169 | 0.4332 | no training: hold-out = score on that country's dev S1 |
| 2026-09-25 | B-certain | rule baseline: certain | val [testlike pool] | 0.1945 | 0.6733 | 0.3869 | 0.3169 | 0.4332 | no training: hold-out = score on that country's dev S1 |
| 2026-09-26 | S3-block | blocking union: certain + 7 blockers (tfidf, name, loc_tfidf, hs, num_loc, addr_tfidf, noaddr_tfidf) | val [full pool] | 0.9760 | 69.8100 | – | – | – | India 0.9772 / US 0.9753; median 69, p90 93; RR 0.9999870; 0.4 pt short of 98% target |
| 2026-09-26 | S3b-prune | OOF LightGBM pruner + top-2 per record + cutoff (floor 0.05, margin 1.0, max_k 8) | val [full pool] | 0.9537 | 4.4700 | – | – | – | median 4, p90 7, F0.5 upper bound 0.9852; hold-out completeness US→India 0.9196, India→US 0.9482 |
| 2026-09-26 | S3-baseline | certain + pruner top-1 (p >= 0.6, one S1 per record) | val [full pool] | 0.9537 | 4.4700 | 0.7734 | 0.7090 | 0.7720 | test run: 2.86 cands/S1, 1.11 matches/S1, validator PASS (test S2 missing) |
| 2026-09-26 | S3-stress | stress slice: lexicon content words hidden -> 17.0% unknown Indic names | val India, Indic-script true pairs | 0.9690 | 75.1000 | – | – | – | normal lexicon 0.9864 on the same slice (-1.7 pt) |
