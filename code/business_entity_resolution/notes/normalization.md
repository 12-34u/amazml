# Stage 2 — Normalisation

Code: `src/normalize.py` (pure, vectorised transforms), `src/normalize_build.py` (learning plus the run over all
files), `src/norm_report.py` (before/after evaluation, which writes [normalization_report.md](normalization_report.md)).
Tests: `tests/test_normalize.py`, built on real EDA examples (32 tests; 52 in the whole suite).

Run: `python src/normalize_build.py` → `artifacts/norm_{split}_s{n}.parquet`, one row per record in the same order
as `{split}_s{n}.parquet`.

## Output fields

| field | example (raw → normalised) |
|---|---|
| `name_script` | latin / latin_accented / devanagari / bengali / gurmukhi / gujarati / … / other |
| `name_latin` | "राम मार्केटिंग प्राइवेट लिमिटेड" → "ram marketing praivet limited" (rule transliteration) |
| `name_norm` | full normalised name: accents folded, `&`/`+`→and, punctuation and junk prefixes (`<<`, `...`, `M/s`) removed, L.L.C.→llc, domains stripped of the TLD, lexicon applied to Indic-origin tokens |
| `name_main` / `name_alias` | "Quoviolyra **fka** Harbor Douglas LLC" → main "quoviolyra", alias "harbor douglas llc" (fka / aka / dba / formerly / doing business as / trading as) |
| `name_core` / `alias_core` | legal forms and professional titles removed: "Harbor LLC Douglas" → "harbor douglas" |
| `legal_form` | canonical set: `llc`, `inc`, `corp`, `co`, `lp`, `llp`, `pc`, `pllc`, `ltd`, `pvt`, `pvt_ltd`, `sa`, `sas`, `sarl`, `snc`, `eurl`… ("Shree Limited Private Information" → `pvt_ltd`) |
| `titles` | DO / MD / DDS / DMD / PhD… only after the first word ("Hannah K. Diaz, DO" → `do`; "Do It Best" keeps "do") |
| `is_domain` | "héartassociation.com" → core "heartassociation", flag True |
| `addr_norm` | folded, transliterated, abbreviations expanded (rd→road, st/saint→street, ave/av→avenue, bd→boulevard, nr→near, opp→opposite, rue, chemin…), `, ` kept between components |
| `house_no` | canonical: first house number, range start, zero-padding stripped ("3908-3910"→3908, "0022632"→22632, "006"→6, "A-303"→303); ordinals, PO boxes and unit/suite numbers ignored |
| `street` / `street_word` | "216 HAYES SAINT" and "216 Hayes St" → "hayes street" / "hayes" (falls back to the first component with a street-type word; landmark components skipped) |
| `hs_key` | house_no + street_word, e.g. "216 hayes" (the certain-match key) |
| `postcode` | last 5–6 digit code that is not the first token and is followed by at most 3 words ("Tyler, TX 75701", "75002 Paris") |
| `city` / `region` / `localities` | last place-like component / last region-level component mapped to the S1 spelling / every place-like component |
| `landmark` | near / nr / opposite / opp / behind / beside / next to / en face / près de |
| `core_h`, `hs_h` | int64 hashes of `name_core` and `hs_key` for key joins |

## Country-agnostic by construction
- No list of countries, states or cities is hard-coded. All regexes are format-level (numbers, separators, word
  classes).
- **Regions are learned per country label** from how components are positioned. A component is region-level if
  it makes up ≥ 0.05% of the country's components and is the *last* component ≥ 60% of the times it appears. The
  split is bimodal:
  - states and departments are last 87–90% of the time;
  - cities are last ≤ 31% of the time.

  This finds 82 US, 47 India and 7 France regions (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire, Gironde,
  Loire-Atlantique, Nord, Pas-de-Calais), while Lille, Bordeaux and Nantes stay cities, all without labels for
  France.
- **Region spellings** are mapped to the S1 spelling using train-split true pairs (71 entries: mh→maharashtra,
  texas→tx, maharastr→maharashtra…).
- **Transliteration lexicon** (585 entries) is learned from 419k train-split true pairs whose candidate name is in
  an Indic script and has the same token count as the S1 name: praivet / praibhet / bhiraivedh→private,
  limidhedh→limited, and pra / li→pvt / ltd (from the Hindi abbreviation प्रा. लि.). French and any future
  country simply don't trigger it.

## Transliteration details
indic-transliteration (MIT) converts each **run** of Indic characters to IAST, so Latin words in mixed strings are
left untouched ("Ahmedabad, ગુજરાત" keeps "Ahmedabad"). Then:
1. candra vowels (ॉ, ॅ) are mapped first;
2. anusvara becomes n/m;
3. Hindi schwa deletion runs on the IAST text, word-final and V C a C V, where the long ā is distinct and never
   dropped;
4. accents are folded and ph→f.

Every unique string is transliterated once (1.85M strings, 65 s on 8 processes) and cached in
`artifacts/translit_cache.parquet`. This is the only per-string Python step. Everything else is Arrow compute.

## Before/after on val (out of sample: learning used the train split only)

Name similarity (token-set ratio), true pairs vs hard negatives:

| country | candidate script | true pairs: mean before → after | true pairs ≥ 80: before → after | same-building negatives ≥ 80: before → after |
|---|---|---|---|---|
| India | Devanagari | 13.9 → **99.5** | 2.2% → **99.9%** | 0.0% → 1.1% |
| India | Bengali | 13.5 → **99.4** | 1.8% → **99.9%** | 0.0% → 0.6% |
| India | Gujarati | 14.3 → **99.5** | 2.0% → **99.8%** | 0.0% → 0.5% |
| India | Gurmukhi | 14.0 → **99.5** | 2.0% → **99.9%** | 0.0% → 1.9% |
| India | Latin | 90.7 → 92.1 | 87.3% → 85.5% | 1.6% → 1.3% |
| US | Latin | 93.0 → 93.5 | 89.1% → 88.9% | 1.3% → 1.3% |
| **India, all** | | 48.4 → **96.2** | 40.4% → **93.5%** | 1.3% → 1.2% |

- **Separation (ROC AUC, true pairs vs same-building negatives): India 0.525 → 0.985, US 0.981 → 0.982.**
- **Same-name negatives stay at 100 by definition.** Normalisation cannot separate identical names; the address has
  to. Their address fields rarely agree: house number 1%, street word 0.5%, shared locality 6% (India) and 0.2% (US).
- **A few Latin true pairs score lower.** Before, a shared "private limited" inflated the score; after, only the core
  counts. Examples: "Universal Healthcare Pvt Ltd" vs "Universal Private Limited Partners" 85→68, "XS Party" vs
  "X5 … Center". The same strictness drops Latin same-building negatives by 10 points on average. Legal-form
  agreement returns as its own feature in Stage 4.

Address fields on true pairs (share equal when present on both sides):

| | house no. | street word | city | region | any shared locality | postcode present on both |
|---|---:|---:|---:|---:|---:|---:|
| India | 77.6% | 69.6% | 64.6% | 99.3% | **99.1%** | 0.0% |
| US | 89.8% | 93.0% | 82.5% | 99.9% | 82.8% | 0.1% |

- **India `city` is only 65% consistent**, because S1 addresses are shuffled and name different levels of the
  hierarchy (Kolkata / Howrah, Pune / Pune City, Noida / Gautam Buddha Nagar). The locality set is the reliable
  signal (99.1% for true pairs vs 36% for same-building and 6% for same-name negatives). Stage 4 will use both.
- **Postcode is effectively unusable as a match signal**, because S1 almost never has one. Postcode blocking
  would contribute nothing.

## Certain-match rule
Evaluated on val with the full pool. Test-like results are identical at this precision.

| rule | precision | recall of val pairs | wrong pairs | macro F0.5 val | F0.5 India / US (dev) |
|---|---:|---:|---:|---:|---:|
| Stage 1: basic name + basic address prefix | 99.98% | 13.8% | 38 | 0.303 | 0.220 / 0.357 |
| core name + hs_key | 99.31% | 34.0% | 3,618 | – | – |
| core name + hs_key + compatible legal form | 99.79% | 32.6% | 1,023 | – | – |
| **name_norm + hs_key (kept as "certain")** | **99.95%** | **19.5%** | 144 | **0.387** | **0.317 / 0.433** |

The kept rule is `name_norm` (legal form included) plus `hs_key`. It has the same precision class as Stage 1
(144 errors across 441k val entities) with 41% more recall. Dropping the legal form triples recall but lets planted
decoys through (LLC vs LTD), so the core-name variants are features, not certain matches.

## Resources
- The full build (`--from-step 1`) takes about 8 minutes (65 s + 9 s + 205 s + 79 s + 96 s for steps 1–5). Combined PSS across workers peaked at **4.67 GB** with 3
  parallel file workers. `JOBS` is now 2 to keep margin under the 5 GB budget.
- The report takes 107 s at 3.9 GB. Tests take 7 s.
- Artifacts add 2.2 GB of norm files. **Only about 4 GB of disk is free**, so rewrites stream through one temp file
  at a time.

## Final-pipeline resources (`python src/normalize_build.py --final`)
- **Validation mode** (the default) learns the lexicon and region map from **train-split** labels, and every
  validation number uses these.
- **Final mode** re-learns both from **all** train labels (`translit_lexicon_final.parquet`,
  `region_map_final.parquet`) and re-normalises the test files with them. It uses 524k aligned Indic pairs, up
  from 419k.
- The region list and the region hierarchy are label-free and shared by both modes.

### Unknown Indic tokens
An unknown token is one in an Indic-script name that, after transliteration and the lexicon, is not a word of any
S1 name.

| set | lexicon | Indic-script names | unknown tokens | names with ≥ 1 unknown |
|---|---|---:|---:|---:|
| val-owned train S2/S3 | train split | 110,384 | 1.5% | 5.1% |
| test S3 | train split | 320,639 | 4.6% | 17.3% |
| test S3 | **all train (final)** | 320,639 | **4.6%** | **17.3%** |

- The final lexicon doesn't change the test rate. The unknown tokens are **123 word types of test-only business
  vocabulary**: laksmi, stors, tredars, motars, medikals, bekri, janral, jvelars, otomobails, ilektroniks,
  restorent… These English words simply don't occur in any training name, so no train-learned lexicon can cover
  them.
- **Tried and rejected, a consonant-skeleton rewrite.** It matches an unknown token to a same-country S1 word
  with the same skeleton ('stors'/'stores' → `strs`). It halved the unknown rate (4.6% → 2.4%), but about 27%
  of its mappings were wrong (myanej→manoj, farmesi→frames, entar→nature, ments→moments). A wrong English
  word creates false name overlaps, which costs precision; an unknown token only costs some recall, and
  character n-grams still partly match "laksmi stors" to "lakshmi stores". `consonant_skeleton` is kept for a
  Stage 4 **feature**, where the model decides how much to trust it.
- **Tried and rejected, an address-anchored lexicon on test data** (unlabelled): Indic S3 names aligned with the
  single S1 at the same house number + street word. It gave only 5 entries, because too few anchored pairs
  contain these words.

## France check (test, never seen in training) — see [france_check.md](france_check.md)
Field coverage on test (share of rows):

| source / country | legal form | house no. | street | city | region | postcode |
|---|---:|---:|---:|---:|---:|---:|
| S1 France | 67% | 99.5% | 99.8% | 100% | 100% | 0.1% |
| S3 France | 60% | 92.9% | 95.6% | 97.1% | 65.9% | 0.2% |
| S1 US / India | 56% / 84% | 98.7% / 88.5% | 99.4% / 85.3% | 99% / 99.9% | 99.7% / 86.3% | ≈0 |

- **Legal forms:** SARL, SAS, SASU, SA, EURL and SCI are extracted anywhere in the name ("SARL Spartiate
  Jeunes", "Bordeaux Sport France SAS").
- **Names:** accents fold ("Mérignac Collège" → "merignac college"); `@handles` lose the `@`.
- **Addresses:** "N°56" → 56 and "011" → 11. Three fixes came out of this check (re-tested; 58 tests pass):
  - a lone "R"/"R." after a house number becomes **rue** ("24 R Jean Jaurès" → "rue jean jaures");
  - "bis"/"ter" after a house number are dropped ("23 Bis R Ledru Rollin" → 23, "rue ledru rollin");
  - av/ave/bd/blvd/rte/imp are expanded at the start of a component ("Av Willy Brandt" → "avenue willy brandt").
    The preceding-word rule still protects state codes such as MT/CT, and "S R Layout" is not turned into "rue".
- **Regions: S1 France always gives the region; 31% of S3 France rows give the *department* instead**
  (Gironde, Nord, Loire-Atlantique, Pas-de-Calais), and 34% give none. A plain "region mismatch" feature
  learned on US/India, where true pairs agree 99% of the time, would punish French true pairs.

### Region hierarchy (`artifacts/region_compat.parquet`, label-free)
Region A is compatible with B when ≥ 80% of A's records are in cities that also occur with B at least 20 times,
counted over all sources and both splits. It learns exactly the right links and nothing else:
- **France:** Gironde ⊂ Nouvelle-Aquitaine, Loire-Atlantique ⊂ Pays de la Loire, Nord ⊂ Hauts-de-France,
  Pas-de-Calais ⊂ Hauts-de-France;
- **India:** dl ↔ dilli (Delhi's code and its Hindi transliteration), Telangana ↔ Andhra Pradesh (the 2014
  split, visible in the training pairs);
- **US:** none. Shared city names like Springfield don't link states, because containment is directional and
  needs ≥ 80%.

Stage 4 will use *region compatible* (equal, or linked, or one side missing) instead of *region equal*.
