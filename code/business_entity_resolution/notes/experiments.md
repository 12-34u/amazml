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
