# Stage 0 — EDA findings

Source: `src/eda.py` (report notebook `notebooks/01_eda.ipynb`). The examples are in [eda_examples.md](eda_examples.md).
Everything reads the compact parquet written by `src/prepare.py`. Peak memory per step is in [resource_log.tsv](resource_log.tsv).

## 1. Row counts

| split | source | US | India | France | total |
|---|---|---:|---:|---:|---:|
| train | S1 | 1,323,633 | 883,188 | – | 2,206,821 |
| train | S2 | 3,016,817 | 2,017,799 | – | 5,034,616 |
| train | S3 | 3,170,056 | 2,115,547 | – | 5,285,603 |
| test | S1 | 663,106 | 809,986 | 259,452 | 1,732,544 |
| test | S2 | **file missing** | | | 0 |
| test | S3 | 1,945,701 | 2,405,000 | 731,615 | 5,082,316 |

**`dataset/test/test_source2.tsv` is not in the provided folder.** The README lists it, and in train about 48% of true
pairs come from S2. Without it, test matches can only come from S3. The pipeline treats sources as an open set
(`data.available_sources`), so it will pick the file up automatically once it is added. **Action: re-download the
test data from the portal.**

About 80% of test S1 entities are US/India, which matches train, and 15% are France (unseen).

## 2. Labels

| | S1 entities | singleton share | mean matches | mean (non-singleton) | max | has both S2 and S3 |
|---|---:|---:|---:|---:|---:|---:|
| US | 1,323,633 | 5.58% | 3.46 | 3.66 | 11 | 80.5% |
| India | 883,188 | 5.59% | 3.46 | 3.67 | 11 | 80.5% |
| all | 2,206,821 | **5.58%** | 3.46 | 3.67 | 11 | 80.5% |

Distribution of true matches per S1 entity (the two countries are identical):
0: 5.6% · 1: 5.4% · 2: 17.0% · 3: 24.1% · 4: 21.9% · 5: 14.6% · 6: 7.5% · 7: 2.9% · 8: 0.8% · 9+: 0.2%.
Per source, an S1 entity typically has 1–3 S2 records and 1–3 S3 records.

Checks on 7,638,365 true pairs:
- **No S2/S3 record is matched to more than one S1 entity.** The one-to-one constraint holds exactly.
- **There are no cross-country true pairs.** Blocking within a country loses nothing.
- Every ID in the ground truth exists in the source files.
- 73.4% of S2 records and 74.6% of S3 records match an S1 entity. **About 26% of the pool matches nothing**, so those
  records are pure distractors.

**Implication.** Only about 1 in 18 S1 entities is a singleton. An all-empty submission scores 0.056, and an entity
where we predict nothing scores 0 in 94% of cases. So "when unsure, don't merge" should apply to each **candidate**,
not to whole entities. A global singleton gate can only help on about 5.6% of entities. The score will be driven by
per-candidate precision *and* recall. For example, with 3 of 4 matches found and no false ones, F0.5 = 0.94. With
all 4 found plus 1 false one, F0.5 = 0.83.

## 3. Noise patterns

Shares of records (train, similar in test):

| | S1 US | S1 India | S2 US | S2 India | S3 US | S3 India | test S1 France | test S3 France |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| non-ASCII name | 0 | 0 | 6.7% | 27.9% | 6.8% | 18.5% | 15.7% | 23.9% |
| Devanagari name | 0 | 0 | 0 | 13.4% | 0 | 7.5% | 0 | 0 |
| junk prefix (`...`, `>>`, `--`) | 0.1% | 0.1% | 2.6% | 11.6% | 2.5% | 7.2% | 3.8% | 3.9% |
| domain-like name (`x.com`) | 0 | 0 | 3.5% | 2.7% | 3.4% | 2.9% | 0 | 2.7% |
| legal-suffix token | 48% | 84% | 43% | 56% | 42% | 63% | 60% | 45% |
| landmark address (near/opp) | 0 | 13.3% | 0 | 11.3% | 0 | 8.6% | 0 | 0 |
| empty address | 0 | 0 | 3.7% | 2.9% | 3.5% | 3.1% | 0 | 2.9% |
| ALL-CAPS address | 0 | 0 | 93% | 25% | 0 | 0 | 0 | 0 |
| 5-digit code | 10.8% | 0.3% | 10.1% | 1.1% | 10.2% | 1.0% | 0.4% | 0.5% |
| 6-digit PIN | 0.1% | 0 | 1.4% | 0 | 1.3% | 0 | 0 | 0 |

What the examples show (see eda_examples.md):
- **Names:** legal suffixes dropped, changed or moved inside the name ("Harbor LLC Douglas"), with punctuation noise
  ("Heart [Association-LP]", "Delta World-L.L.C."). Accents are injected into English ("Harbor Dóuglas", "Prívate
  Límited"). There are typos and digit/letter swaps ("Pi1ates", "Cosnlutancy"). Filler words are added ("Center",
  "Services", "Partners", "Enterprises", "Board", doubled words like "CAMPBELL CAMPBELL"). Words are reordered.
  "&" becomes "+" or "and".
- **Aliases and renames:** "Quoviolyra **fka** Harbor Douglas LLC", "Vantageonyx **Formerly** …", "Novitavonovi
  **doing business as** …", "Evoyuma **D.B.A.** Timber", "Calokorflux **aka** …". Some true matches carry a
  **completely different invented name** ("Tavoquovantage", "Gildvio") and match on address alone. Domain names
  squash the words together, sometimes shuffled ("foodprivategreat.com", "héartassociation.com").
- **Scripts:** India names are fully transliterated into local scripts, not only Devanagari: Gurmukhi "ਈਸਟ
  ਇਨਵੈਸਟਮੈਂਟਸ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ", Gujarati "માય પ્રોડ્યુસર", Bengali "গ্রেট ফুড". State names in addresses also
  appear in native script ("ગુજરાત", "ಕರ್ನಾಟಕ", "महाराष्ट्र").
- **Addresses:** components are reordered ("IA, Iowa City, 1064 Newton Rd"). Abbreviations vary, including a
  *wrong* expansion ("216 HAYES **SAINT**" for "St"). House numbers are noisy: truncated or extended "716" vs "7166",
  ranges "3908-3910", zero-padded "0022632"/"006". There are city typos ("Mniot", "Waukkegan") and "City"/"CITY"
  suffixes. States appear as a code or a full name. Addresses can be partial (only "303, Mumbai, MH"), empty, or
  "PO BOX". Postcodes are rare everywhere (≤ 11% of US records, about 0% for India/France), so **postcode blocking
  will contribute little**.

True-pair similarity (100k sampled pairs, `basic_key` text):

| country | cand. source | exact name | exact address | median name token-set | name token-set < 60 | median address token-set |
|---|---|---:|---:|---:|---:|---:|
| India | S2 | 17.1% | 11.6% | 93 | **27.6%** | 99 |
| India | S3 | 19.8% | 4.2% | 97 | 19.1% | 93 |
| US | S2 | 30.3% | 13.5% | 100 | 4.9% | 95 |
| US | S3 | 30.4% | 4.4% | 100 | 4.9% | 87 |

In India, 19–28% of true pairs have low name similarity. That comes from the script transliteration above, plus
renames. Name-only blocking will miss these. They need an address-driven blocker, a multilingual embedding blocker,
or both.

## 4. Name frequency (this drives the blocking design)

Share of records whose normalised name (within a country) is shared by a group of the given size:

| pool | country | 1 | 2 | 3–5 | 6–10 | 11–50 | 51–200 | >200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train S1 | US | 64.2% | 10.2% | 9.2% | 3.2% | 5.8% | 7.3% | 0.2% |
| train S1 | India | 55.8% | 7.6% | 8.1% | 7.6% | **20.7%** | 0.3% | 0 |
| train S2+S3 | US | 62.0% | 12.6% | 11.1% | 4.1% | 5.5% | 3.7% | **1.1%** |
| train S2+S3 | India | 62.3% | 11.5% | 9.9% | 6.1% | 9.4% | 0.9% | 0 |
| test S1 | US | 70.9% | 9.2% | 6.2% | 1.8% | 10.6% | 1.3% | 0 |
| test S1 | India | 56.5% | 7.5% | 8.4% | 7.2% | 20.4% | 0.1% | 0 |
| test S1 | France | 65.6% | 9.5% | 8.9% | 5.6% | 8.4% | 1.8% | 0.1% |
| test S2+S3 | US | 76.7% | 9.8% | 6.0% | 2.5% | 4.0% | 0.9% | 0.1% |
| test S2+S3 | India | 74.3% | 8.8% | 7.3% | 4.6% | 4.7% | 0.3% | 0 |
| test S2+S3 | France | 64.7% | 13.3% | 9.9% | 3.9% | 5.6% | 2.4% | 0.3% |

- The most frequent names are generic. In US train S2+S3: "physical therapy" (1,071 records), "primary care"
  (1,070), "pediatric dental" (1,004), "urgent care", "internal medicine". In India: "new delhi", "india", "shree",
  "services", plus two-letter strings ("sc", "si", "ss", "st"). In France test: "cc" (351), "pc", "ac", "lc", and
  "bordeaux club"/"nantes club". The **two-letter names are acronym-style names** ("S.C.", "C.C."), so we need an
  acronym feature and must never block on them alone.
- **Even Source 1 repeats names.** In India, 20% of S1 entities share their normalised name with 11–50 other S1
  entities ("Green Agro Private Limited" in Rohtak and in Mumbai are different businesses). Name alone does not
  identify a business.
- **Same-name pressure.** 36% of US S1 entities and 42% of India S1 entities have at least one S2/S3 record with the
  **identical** normalised name that is *not* their match. 4.6% of US S1 entities have over 100 such records.

  | same-name non-matches per S1 | 0 | 1 | 2–5 | 6–20 | 21–100 | >100 |
  |---|---:|---:|---:|---:|---:|---:|
  | US | 63.9% | 7.0% | 10.4% | 5.6% | 8.5% | 4.6% |
  | India | 58.1% | 6.0% | 11.6% | 22.5% | 1.7% | 0 |

**Implication for blocking.** An exact or near-exact name key must not be used on its own, and frequent names must
not be dropped. Inside a name group, rank the members by address similarity (street number + street word, city)
and keep the top-K per S1 record. The largest groups (over 1,000 records) are exactly where that cap matters.

## 5. Hard negatives

50k S1 entities were sampled (seed 11), and key groups larger than 50 were skipped (exploration only).
`true share` is the precision of the key used alone as a match rule.

| type | definition | joined pairs | true share | negatives | S1 with ≥ 1 negative | negatives owned by no S1 |
|---|---|---:|---:|---:|---:|---:|
| same name | identical normalised name | 165,869 | **24.3%** | 125,562 | 32.1% | 15% |
| same address | identical normalised address | 17,194 | 82.3% | 3,050 | 2.8% | 6% |
| look-alike | same house number + street word, name token-set ≥ 80, names differ | 49,991 | 96.4% | 1,788 | 3.3% | **87%** |

- **Same name, different place.** These negatives are mostly (85%) *another S1 entity's* records, for example
  "Shiv Projects Private Limited" in Nagpur against the one in Bareilly. Address disambiguates them.
- **Same address, different business.** These are shared buildings (medical offices, business centres), for
  example "Dermatology Group" against "Internal Medicine Corp" at 8 Mountain Avenue. Name disambiguates them.
- **Look-alikes are planted decoys.** 87% match no S1 at all. They copy the name and address and change one
  thing:
  - an extra distinguishing word: "Subham Trading **Overseas** Private Limited", "Standard Automation Networks
    **Northside**", "Rhombus Traders **Infratech**"
  - a different legal form: "Lower Procap **LLC**" vs "LOWER PROCAP **LTD.**", "Vision Engineering **(L.L.P.)**"
    vs "Private Limited"
  - a different professional title: "Hannah K. Diaz, **DO**" vs "**GO**"
  - a nearby house or unit number: 7083 vs 7104, "G-8" vs "G-10", 1440 vs 1443, "D-192 To D-195" vs "D-192. TO D-216"

  True matches also add filler words ("Center", "Services") and have noisy house numbers. The model therefore needs
  **token-level difference features**: which tokens are unmatched, how rare they are, and whether they are known
  filler words, plus legal-suffix agreement, numeric-token agreement, and the context features (how many near-copies
  sit in the same shortlist). Candidate-level similarity alone will not separate them. The one-to-one constraint
  does not help with the unowned decoys, because nobody else claims them.

## 5b. Candidate budget (for the Stage 3b pruning rule)

The reduction ratio is `1 − candidates / (|S1| × |S2+S3| in the same country)`. Its all-pairs denominators are:

| split | country | S1 | S2+S3 | all pairs | true pairs | true per S1 | min average candidates for 95% completeness |
|---|---|---:|---:|---:|---:|---:|---:|
| train | US | 1,323,633 | 6,186,873 | 8.19e12 | 4,578,522 | 3.459 | 3.29 |
| train | India | 883,188 | 4,133,346 | 3.65e12 | 3,059,843 | 3.465 | 3.29 |
| test | US | 663,106 | 1,945,701* | 1.29e12 | – | – | – |
| test | India | 809,986 | 2,405,000* | 1.95e12 | – | – | – |
| test | France | 259,452 | 731,615* | 1.90e11 | – | – | – |

\* Test counts are S3 only, because `test_source2.tsv` is missing.

Best completeness possible with a fixed per-S1 cap, **assuming a perfect ranker** (train):

| cap k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| completeness ceiling | 27.3% | 53.0% | 73.8% | 87.7% | **95.2%** | 98.5% | 99.6% | 99.9% |
| average candidates | 0.94 | 1.83 | 2.55 | 3.03 | 3.29 | 3.41 | 3.45 | 3.46 |

- **The 2–5 target has a floor of about 3.3.** Every kept true pair counts as a candidate, so completeness ≥ 95%
  needs at least 0.95 × 3.46 ≈ **3.29 candidates per S1 on average**, even with a perfect pruner. The realistic
  window is about 3.5–5.
- **A fixed top-k cannot work.** Reaching 95% needs k ≥ 5 even with a perfect ranker, because 27% of S1 entities
  have 5–11 matches. The pruner needs the adaptive margin/floor rule: many slots for entities with many
  near-identical matches, and zero for clear singletons.
- **The reduction ratio saturates.** At 4 candidates per S1 it is already 0.999999 (train) and 0.999998 (test), so
  it cannot separate operating points. Report it with 7 decimal places, but compare operating points on average
  and median candidates, completeness and final F0.5.

## 6. Dev subset

`python src/splits.py` → `artifacts/dev_s1.parquet` and `artifacts/dev_truth.parquet`
- 150,000 S1 entities, stratified by country with seed 42 (US 89,969, India 60,031), and 519,062 true pairs.
- S2/S3 are **not** subsampled. All 10.3M train S2/S3 records stay in the candidate pool, so blocking faces the real
  number of distractors. Records belonging to S1 entities outside the subset act as extra distractors. This makes dev
  precision slightly *pessimistic*, and the one-to-one reassignment cannot see those owners.

## 7. Resources (peak RSS per step; budget 5 GB)

| step | time | peak RSS |
|---|---:|---:|
| `prepare.py` (all TSV → parquet, truth → long) | 151 s | 2.45 GB |
| `splits.py` (dev subset) | 1 s | 1.13 GB |
| EDA counts and labels | 12 s | 1.26 GB |
| EDA noise and similarity | 31 s | 2.47 GB |
| EDA name frequency | 48 s | 2.47 GB |
| EDA examples and hard negatives | 55 s | 3.62 GB |
| EDA candidate budget | 3 s | 0.73 GB |
| `01_eda.ipynb`, full "Restart & Run All" (one kernel) | – | 4.40 GB |

## 8. Decisions this feeds into later stages

1. **Blocking (Stage 3)** will be within a country and a union of these blockers:
   - name TF-IDF top-K, re-ranked by address inside large name groups (a cap per record, not dropping names)
   - an address key (house number + street word, city) re-ranked by name, to catch renames and invented names
   - an alias split on fka/aka/dba/formerly/"doing business as" and domain tokenisation
   - a multilingual embedding (or transliteration) blocker for the 13–28% of India pairs in native scripts

   Postcode blocking is low priority because postcodes are rare.
2. **Normalisation (Stage 2)** must strip junk prefixes, handle alias markers, segment domains, and avoid wrong
   St→Saint expansions (map "saint" and "st" to the same token). House numbers need range and zero-padding
   handling. State names and codes should map to one form, including native-script state names.
3. **Features (Stage 4)** should add, beyond the planned ones: unmatched-token rarity, filler-word vs distinguishing-
   word flags, legal-form agreement, numeric/unit-token agreement, a "near-copy count in shortlist" context feature,
   and script flags.
4. **Pruning (Stage 3b)** will use an adaptive cutoff (a margin to the best score plus a floor, with empty lists
   allowed), aiming for about 3.5–5 candidates per S1 at ≥ 95% completeness. The one-to-one constraint can already
   be applied at this stage: drop a candidate from an S1 when another S1 claims it with a much higher prune score.
5. **Decision rule (Stage 5)** should always apply the one-to-one assignment (it holds exactly in the labels). Tune
   the thresholds for each candidate. A singleton gate can help only on about 5.6% of entities.
6. **France:** 16–24% of French names are non-ASCII (accents are handled by NFKD), there are no landmarks and almost
   no postcodes, and acronym-style names ("cc", "pc") are common. The country hold-out (US↔India) will be the proxy.
