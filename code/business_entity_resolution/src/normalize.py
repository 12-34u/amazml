"""Text normalisation for names and addresses (Stage 2).

Everything runs as vectorised Arrow compute (RE2 regex, list kernels) over whole
columns. The one exception is transliteration: indic-transliteration (MIT) is a Python
library, so it is called once per *unique* Indic-script string and cached in
artifacts/translit_cache.parquet.

No country is assumed anywhere. The regexes are script- and format-generic. The two
data-driven resources are learned from the provided files only, never external data:
  * translit lexicon: transliterated token -> English token, from train-split true pairs
                      (e.g. "praivet" -> "private")
  * region resources: which address components are region-level (states, departments)
                      per country label, plus a map of region spellings to the S1 spelling
                      learned from train-split true pairs (e.g. "gj" -> "gujarat")

Name outputs:    name_script, name_latin, name_norm, name_main, name_core, legal_form,
                 name_alias, alias_core, titles, is_domain
Address outputs: addr_script, addr_norm, house_no, street, street_word, postcode, landmark,
                 hs_key (house number + first street word); city and region come from
                 `city_region` once the region resources are learned
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

# ============================================================================ basics
# Combining diacritics produced by NFKD (é -> e + U+0301). Only this block is removed,
# because Indic vowel signs are also "marks" (\p{M}) and must be kept.
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
    """Hash of the first number+word in a basic_key address (0 when there is none).
    Survives component reordering ("Crossville, 994 Miller Ave" vs "994 MILLER AVENUE")."""
    pref = addr_key.str.extract(ADDR_PREFIX_RE, expand=False).fillna("")
    return hash_key(pa.array(pref, type=pa.string()))


def _chunked(arr) -> pa.Array:
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr


def _np(arr) -> np.ndarray:
    return _chunked(arr).to_numpy(zero_copy_only=False)


def _sub(x, pattern: str, repl: str):
    return pc.replace_substring_regex(x, pattern, repl)


def _squash_spaces(x):
    return pc.utf8_trim_whitespace(_sub(x, r"\s+", " "))


def _fold(x):
    """NFKD + drop accents + lowercase."""
    return pc.utf8_lower(_sub(pc.utf8_normalize(x, form="NFKD"), _DIACRITICS, ""))


def _rebuild(values: pa.Array, parents: np.ndarray, n: int, sep: str = " ") -> pa.Array:
    """Join token `values` (sorted by parent row) back into one string per row."""
    counts = np.bincount(parents, minlength=n) if len(parents) else np.zeros(n, dtype=np.int64)
    offsets = pa.array(np.concatenate([[0], np.cumsum(counts)]).astype(np.int32))
    return pc.binary_join(pa.ListArray.from_arrays(offsets, _chunked(values)), sep)


# ============================================================================ scripts
# label -> (Unicode block, sanscript scheme name)
INDIC_SCRIPTS = {
    "devanagari": (r"[\x{0900}-\x{097F}]", "devanagari"),
    "bengali": (r"[\x{0980}-\x{09FF}]", "bengali"),
    "gurmukhi": (r"[\x{0A00}-\x{0A7F}]", "gurmukhi"),
    "gujarati": (r"[\x{0A80}-\x{0AFF}]", "gujarati"),
    "oriya": (r"[\x{0B00}-\x{0B7F}]", "oriya"),
    "tamil": (r"[\x{0B80}-\x{0BFF}]", "tamil"),
    "telugu": (r"[\x{0C00}-\x{0C7F}]", "telugu"),
    "kannada": (r"[\x{0C80}-\x{0CFF}]", "kannada"),
    "malayalam": (r"[\x{0D00}-\x{0D7F}]", "malayalam"),
}
SCRIPT_LABELS = ["latin", "latin_accented", *INDIC_SCRIPTS, "other"]


def detect_script(arr) -> np.ndarray:
    """int8 code per row into SCRIPT_LABELS. The first Indic block found wins. Otherwise
    the row is 'latin_accented' if it has Latin letters with diacritics, 'other' for any
    other non-ASCII letter, and plain 'latin' if it is ASCII."""
    arr = _chunked(arr)
    codes = np.zeros(len(arr), dtype=np.int8)
    for i, (pat, _) in enumerate(INDIC_SCRIPTS.values()):
        m = _np(pc.match_substring_regex(arr, pat))
        codes[(codes == 0) & m] = SCRIPT_LABELS.index(list(INDIC_SCRIPTS)[i])
    non_ascii = _np(pc.match_substring_regex(arr, r"[^\x00-\x7f]")) & (codes == 0)
    accented = _np(pc.match_substring_regex(arr, r"[\x{00C0}-\x{024F}\x{1E00}-\x{1EFF}]"))
    codes[non_ascii & accented] = SCRIPT_LABELS.index("latin_accented")
    codes[non_ascii & ~accented] = SCRIPT_LABELS.index("other")
    return codes


def script_array(codes: np.ndarray) -> pa.DictionaryArray:
    return pa.DictionaryArray.from_arrays(pa.array(codes), pa.array(SCRIPT_LABELS))


def is_indic(codes: np.ndarray) -> np.ndarray:
    lo, hi = SCRIPT_LABELS.index("devanagari"), SCRIPT_LABELS.index("malayalam")
    return (codes >= lo) & (codes <= hi)


# ---------------------------------------------------------------- transliteration
# Devanagari/Gujarati "candra" vowels used for English sounds (ऑफिस = office) are
# not in the classical tables: map them to plain o/e signs first.
_PRE_MAP = str.maketrans({"ॉ": "ो", "ऑ": "ओ", "ॅ": "े", "ऍ": "ए", "ૉ": "ો", "ઑ": "ઓ", "ૅ": "ે", "ઍ": "એ"})
# Schwa rules run on IAST, where the inherent short 'a' is distinct from long 'ā'.
_C = "bcdfghjklmnpqrstvwxyzṭḍṅñṇśṣḥ"
_V = "aāiīuūṛṝeo"
_RE_ANUSVARA_LABIAL = re.compile(r"ṃ(?=[pbm])")
_RE_FINAL_SCHWA = re.compile(rf"(?<=\w[{_C}])a\b")                        # rāma -> rām
_RE_MEDIAL_SCHWA = re.compile(rf"(?<=[{_V}][{_C}])a(?=[{_C}][{_V}])")      # inavaisa -> invaisa


# A run of Indic characters (any block, incl. vowel signs and ZWJ/ZWNJ), with inner spaces.
_INDIC_RUN = re.compile(r"[\u0900-\u0DFF](?:[\u0900-\u0DFF\u200c\u200d\s]*[\u0900-\u0DFF\u200c\u200d])?")
_BLOCKS = [(*(int(h, 16) for h in re.findall(r"\{([0-9A-Fa-f]{4})\}", p)), name) for name, (p, _) in INDIC_SCRIPTS.items()]


def _script_of(ch: str) -> str | None:
    o = ord(ch)
    return next((name for lo, hi, name in _BLOCKS if lo <= o <= hi), None)


def _translit_run(run: str, script: str) -> str:
    from indic_transliteration import sanscript
    s = sanscript.transliterate(run.translate(_PRE_MAP), INDIC_SCRIPTS[script][1], sanscript.IAST)
    s = _RE_ANUSVARA_LABIAL.sub("m", s).replace("ṃ", "n").replace("m̐", "n")
    s = _RE_MEDIAL_SCHWA.sub("", _RE_FINAL_SCHWA.sub("", s))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return s.replace("ph", "f")


def transliterate_one(text: str, script: str | None = None) -> str:
    """Indic-script runs -> lowercase ASCII approximating English spelling; everything
    else (Latin words, digits, punctuation) is left untouched, so 'Ahmedabad, ગુજરાત'
    keeps 'Ahmedabad'. Per run: IAST via indic-transliteration, anusvara -> n/m, drop
    accents, ph -> f, Hindi-style schwa deletion (word-final and V C a C V)."""
    return _INDIC_RUN.sub(lambda m: _translit_run(m.group(0), _script_of(m.group(0)[0]) or script), text)


def transliterate_unique(values: list[str], scripts: list[str]) -> list[str]:
    return [transliterate_one(v, s) for v, s in zip(values, scripts)]


def apply_translit_cache(arr, codes: np.ndarray, cache: pa.Table | None):
    """Replace Indic-script rows by their cached transliteration (other rows unchanged)."""
    arr = _chunked(arr)
    if cache is None or not is_indic(codes).any():
        return arr
    idx = pc.index_in(arr, value_set=cache["raw"])
    mapped = pc.take(cache["latin"], idx)
    return pc.if_else(pc.and_(pa.array(is_indic(codes)), pc.is_valid(idx)), mapped, arr)


# ============================================================================ names
LEGAL_TOKENS = {
    "pvtltd": "pvt_ltd", "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "pvte": "pvt",
    "llp": "llp", "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp",
    "corporation": "corp", "co": "co", "company": "co", "lp": "lp", "pllc": "pllc", "pc": "pc",
    "plc": "plc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "snc": "snc",
    "eurl": "eurl", "sci": "sci", "scop": "scop", "selarl": "selarl", "gie": "gie", "gmbh": "gmbh",
}
# Multi-word legal forms, collapsed to one placeholder token before token matching.
LEGAL_PHRASES = [
    (r"\b(?:private|pvt|pvte)\s+(?:limited|ltd)\b", "pvtltd"),
    (r"\blimited\s+liability\s+company\b", "llc"),
    (r"\blimited\s+liability\s+partnership\b", "llp"),
    (r"\bprofessional\s+(?:corporation|corp)\b", "pc"),
    (r"\bsociete\s+a\s+responsabilite\s+limitee\b", "sarl"),
    (r"\bsociete\s+par\s+actions\s+simplifiee\b", "sas"),
    (r"\bsociete\s+anonyme\b", "sa"),
]
TITLE_TOKENS = ["md", "do", "dds", "dmd", "phd", "dr", "mr", "mrs", "ms", "esq", "cpa", "rn", "np",
                "od", "dpm", "dvm", "facs", "facp", "mbbs", "ca", "adv"]
ALIAS_MARKERS = r"fka|aka|dba|formerly known as|formerly|also known as|doing business as|trading as|previously known as"
TLDS = r"co\s*\.\s*in|com|net|org|in|co|fr|biz|info|io|us"

_LEGAL_KEYS = pa.array(list(LEGAL_TOKENS))
_LEGAL_VALS = pa.array(list(LEGAL_TOKENS.values()))
_TITLE_KEYS = pa.array(TITLE_TOKENS)


def collapse_single_letters(x):
    """Join runs of single-letter tokens: 'l l c' -> 'llc', 's a s' -> 'sas', 'f k a' -> 'fka'.
    Two passes mark every single letter as <x>, then adjacent marks are fused."""
    for _ in range(2):
        x = _sub(x, r"(^|\s)([a-z])(\s|$)", r"\1<\2>\3")
    x = _sub(x, r">\s+<", "")
    return _sub(x, r"[<>]", "")


def _clean_name_text(x):
    """Folded text -> tokens separated by single spaces."""
    x = _sub(x, r"['’`´]", "")                                  # orelee's -> orelees
    x = _sub(x, r"\s\+\s", " and ")                             # tele + sons
    x = pc.replace_substring(x, "&", " and ")
    x = _sub(x, r"^\W*m\s*/\s*s\b\.?", " ")                     # "M/s Great Food" (Indian 'Messrs')
    x = _sub(x, _NON_WORD, " ")
    return collapse_single_letters(_squash_spaces(x))


def split_legal(x, n: int) -> tuple[pa.Array, pa.Array, pa.Array]:
    """Normalised name -> (core, legal_form, titles). Legal tokens may sit anywhere
    ("Harbor LLC Douglas"). legal_form is the sorted set of canonical forms joined by '|',
    with pvt+ltd merged into pvt_ltd. If nothing but legal/title tokens remain, the core
    keeps the full name (e.g. a business literally called 'Co')."""
    for pat, rep in LEGAL_PHRASES:
        x = _sub(x, pat, rep)
    toks = pc.split_pattern(x, " ")
    flat, parents = pc.list_flatten(toks), _np(pc.list_parent_indices(toks))
    lens = _np(pc.list_value_length(toks)).astype(np.int64)
    pos = np.arange(len(parents)) - np.repeat(np.cumsum(lens) - lens, lens)
    li = pc.index_in(flat, value_set=_LEGAL_KEYS)
    is_legal = _np(pc.is_valid(li))
    # professional titles follow a person's name ("Hannah K. Diaz, DO"), never lead it
    is_title = _np(pc.is_in(flat, value_set=_TITLE_KEYS)) & (pos > 0) & ~is_legal
    keep = ~is_legal & ~is_title
    core = _rebuild(pc.filter(flat, pa.array(keep)), parents[keep], n)
    core = _squash_spaces(_sub(core, r"^(?:and|the)\s+|\s+and$", ""))
    core = pc.if_else(pc.equal(core, ""), _squash_spaces(_sub(x, r"\bpvtltd\b", "private limited")), core)

    forms = pd.DataFrame({"p": parents[is_legal], "f": _np(pc.take(_LEGAL_VALS, pc.filter(li, pa.array(is_legal))))})
    # 'Private' and 'Limited' anywhere in the name (even reordered) form one pvt_ltd.
    has = lambda f: forms.p[forms.f == f].unique()
    merged = np.union1d(has("pvt_ltd"), np.intersect1d(has("pvt"), has("ltd")))
    forms = forms[~(forms.p.isin(merged) & forms.f.isin(["pvt", "ltd", "pvt_ltd"]))]
    forms = pd.concat([forms, pd.DataFrame({"p": merged, "f": "pvt_ltd"})])
    forms = forms.drop_duplicates().sort_values(["p", "f"], kind="stable")
    legal = _rebuild(pa.array(forms.f.to_numpy(), pa.string()), forms.p.to_numpy(), n, "|")
    tt = pd.DataFrame({"p": parents[is_title], "t": _np(pc.filter(flat, pa.array(is_title)))}).drop_duplicates()
    titles = _rebuild(pa.array(tt.t.to_numpy(), pa.string()), tt.p.to_numpy(), n, "|")
    return core, legal, titles


_SKELETON_RULES = [(r"x", "ks"), (r"ph", "f"), (r"sh", "s"), (r"ch", "c"), (r"c([eiy])", r"s\1"), (r"c", "k"),
                   (r"q", "k"), (r"g([eiy])", r"j\1"), (r"w", "v"), (r"z", "j"), (r"[aeiouy]", "")]


def consonant_skeleton(arr) -> pa.Array:
    """Spelling-robust key for transliterated words: soft c/g, x->ks, ph->f, sh->s, w->v,
    z->j, vowels dropped, doubled consonants collapsed. 'stors'/'stores' -> 'strs',
    'jvelars'/'jewellers' -> 'jvlrs', 'medikals'/'medicals' -> 'mdkls'.
    Used as a matching FEATURE, not to rewrite names: as a rewrite rule it mapped ~27% of
    test-only words to the wrong English word (e.g. 'myanej' -> 'manoj'), creating false overlaps."""
    x = pc.utf8_lower(_chunked(pa.array(arr, pa.string()) if not isinstance(arr, (pa.Array, pa.ChunkedArray)) else arr))
    for pat, rep in _SKELETON_RULES:
        x = _sub(x, pat, rep)
    for ch in "bcdfghjklmnpqrstvwxz":
        x = _sub(x, f"{ch}{ch}+", ch)
    return x


def apply_lexicon(x, rows: np.ndarray, lexicon: pa.Table | None):
    """Replace tokens by their learned English spelling on the selected rows (Indic-origin names)."""
    if lexicon is None or not rows.any():
        return x
    x = _chunked(x)
    src, dst = (pc.cast(_chunked(lexicon[c]), pa.string()) for c in ("src", "dst"))   # tables built from pandas are large_string
    n = len(x)
    toks = pc.split_pattern(x, " ")
    flat, parents = pc.list_flatten(toks), _np(pc.list_parent_indices(toks))
    idx = pc.index_in(flat, value_set=src)
    use = pa.array(rows[parents]) if len(parents) else pa.array([], pa.bool_())
    new = pc.if_else(pc.and_(use, pc.is_valid(idx)), pc.take(dst, idx), flat)
    return _rebuild(new, parents, n)


def normalize_names(raw, cache: pa.Table | None = None, lexicon: pa.Table | None = None) -> dict:
    raw = _chunked(raw)
    n = len(raw)
    codes = detect_script(raw)
    latin = apply_translit_cache(raw, codes, cache)
    x = _fold(latin)
    dom_pat = rf"\b(?:www\s*\.\s*)?([a-z0-9][a-z0-9-]*)\s*\.\s*(?:{TLDS})\b"
    is_domain = pc.or_(pc.match_substring_regex(x, dom_pat), pc.match_substring_regex(x, r"[a-z]\s+com$"))
    x = _sub(_sub(x, dom_pat, r"\1"), r"([a-z])\s+com$", r"\1")
    x = _clean_name_text(x)
    x = apply_lexicon(x, is_indic(codes), lexicon)
    # "Quoviolyra fka Harbor Douglas LLC" -> main 'quoviolyra', alias 'harbor douglas llc'
    parts = pc.extract_regex(x, rf"^(?P<main>.*?)\s+(?:{ALIAS_MARKERS})\s+(?P<alias>.+)$")
    has_alias = pc.is_valid(parts)
    main = pc.if_else(has_alias, pc.struct_field(parts, "main"), x)
    alias = pc.if_else(has_alias, pc.struct_field(parts, "alias"), pa.scalar(""))
    main = pc.fill_null(main, "")
    alias = pc.fill_null(alias, "")
    core, legal, titles = split_legal(main, n)
    a_core, a_legal, _ = split_legal(alias, n)
    legal = pc.if_else(pc.equal(legal, ""), a_legal, legal)
    return {
        "name_script": script_array(codes), "name_latin": latin, "name_norm": x,
        "name_main": main, "name_core": core, "legal_form": legal,
        "name_alias": alias, "alias_core": pc.if_else(pc.equal(alias, ""), pa.scalar(""), a_core),
        "titles": titles, "is_domain": is_domain,
    }


# ============================================================================ addresses
# Canonical spelled-out forms. English and French street types, unit words, landmarks.
# "saint" -> "street" deliberately: the data contains wrong expansions like
# "216 HAYES SAINT" for "St". Mapping both to one token keeps them comparable, and
# genuine "Saint X" place names map the same way on every side.
ADDR_ABBREV = [
    ("road", r"rd|rood"), ("street", r"st|str|saint"), ("avenue", r"ave|av|avn|aven"),
    ("boulevard", r"blvd|bd|bld|boul"), ("drive", r"dr|drv"), ("lane", r"ln"), ("court", r"ct|crt"),
    ("place", r"pl"), ("circle", r"cir|circ"), ("highway", r"hwy|hiway"), ("parkway", r"pkwy|pky"),
    ("cove", r"cv"), ("terrace", r"ter|terr"), ("trail", r"trl"), ("square", r"sq"),
    ("expressway", r"expy"), ("crossing", r"xing"), ("point", r"pt"), ("mount", r"mt"),
    ("suite", r"ste"), ("apartment", r"apt"), ("floor", r"fl|flr"), ("building", r"bldg"),
    ("near", r"nr"), ("opposite", r"opp|opos"), ("number", r"no|num"), ("sector", r"sec"),
    ("route", r"rte"), ("chemin", r"ch|chem"), ("impasse", r"imp"), ("allee", r"all"),
    ("faubourg", r"fbg|fg"), ("residence", r"res"), ("quartier", r"qu"),
]
LANDMARK_RE = r"\b(?:near|nr|opposite|opp|behind|beside|next to|adjacent to|in front of|pres de|en face)\b"
UNIT_WORDS = r"unit|apartment|suite|room|floor|flat|building|block|plot|number|box|po box"


STREET_TYPES = ("road|street|avenue|boulevard|drive|lane|court|place|circle|highway|parkway|cove|terrace|trail|"
                "square|expressway|way|marg|rue|chemin|impasse|allee|route|quai|cours|faubourg")


# Unambiguous street-type abbreviations that may also START a component ("Av Willy Brandt").
LEADING_ABBREV = [("avenue", r"av|ave"), ("boulevard", r"bd|blvd"), ("route", r"rte"), ("impasse", r"imp")]


def expand_address_abbreviations(x):
    """Expand an abbreviation only when a word precedes it in the same component
    ("hayes st" -> "hayes street"). A component that is just "MT" or "CT" is a state
    code, not "mount"/"court", and stays as it is. French specifics: 'bis'/'ter' after a
    house number are dropped ("23 bis r ledru rollin"), a lone 'r' right after a house
    number is 'rue' ("24 r jean jaures"), and a few unambiguous abbreviations are also
    expanded at the start of a component."""
    x = _sub(x, r"\b(\d+)\s+(?:bis|ter)\s+", r"\1 ")
    x = _sub(x, r"\b(\d+[a-z]?\s+)r\b", r"\1rue")
    for full, abbr in LEADING_ABBREV:
        x = _sub(x, rf"(^|,\s)(?:{abbr})\s+(\p{{L}})", rf"\1{full} \2")
    for full, abbr in ADDR_ABBREV:
        x = _sub(x, rf"([\p{{L}}\p{{N}}]\s+)(?:{abbr})\b", rf"\1{full}")
    return x


def _clean_address_text(x):
    """Folded address -> tokens with ', ' kept between components."""
    x = pc.replace_substring(x, "&", " and ")
    x = _sub(x, r"[^\p{L}\p{M}\p{N},]+", " ")                  # '-' and '/' split ranges: 3908-3910
    x = _sub(x, r"\s*,[\s,]*", ", ")
    x = _sub(_squash_spaces(x), r"^,\s*|,\s*$", "")
    return _squash_spaces(expand_address_abbreviations(x))


PC_EXTRACT = r"^.*[\s,](?P<pc>\d{5,6})(?:[\s,]+[^\d\s,]+){0,3}[\s,]*$"   # last 5-6 digit code, not first token
PC_STRIP = r"[\s,]\d{5,6}((?:[\s,]+[^\d\s,]+){0,3}[\s,]*)$"


def parse_address_parts(x) -> dict:
    """Postcode, canonical house number, street and components from a cleaned address."""
    pcode = pc.fill_null(pc.struct_field(pc.extract_regex(x, PC_EXTRACT), "pc"), "")
    body = _sub(x, PC_STRIP, r"\1")
    tmp = _sub(body, r"\b\d+(?:st|nd|rd|th)\b", " ")                     # ordinals: 73rd street
    tmp = _sub(tmp, r"\b(?:po\s+)?box\s+\d+\b", " ")                     # PO BOX 9820
    tmp = _sub(tmp, r"\b(?:unit|apartment|suite)\s+(?:apartment\s+)?\d+[a-z]?\b", " ")
    tmp = _squash_spaces(tmp)
    hn = pc.struct_field(pc.extract_regex(tmp, r"(?:^|[\s,])(?P<h>\d+)[a-z]?(?:[\s,]|$)"), "h")
    house = pc.fill_null(_sub(hn, r"^0+(\d)", r"\1"), "")              # 0022632 -> 22632, 006 -> 6
    st1 = pc.struct_field(pc.extract_regex(tmp, r"(?:^|[\s,])\d+[a-z]?\s+(?P<s>[^,\d][^,]*)"), "s")
    st2 = pc.struct_field(pc.extract_regex(tmp, r"(?:^|[\s,])\d+[a-z]?\s*,\s*(?P<s>[^,\d][^,]*)"), "s")
    # fallback: first component containing a street-type word ("73rd street, minot")
    st3 = pc.struct_field(pc.extract_regex(body, rf"(?:^|,\s)(?P<s>[^,]*\b(?:{STREET_TYPES})\b[^,]*)"), "s")
    street = pc.fill_null(pc.coalesce(st1, st2), "")
    street = _squash_spaces(_sub(street, rf"\b(?:{UNIT_WORDS})\b.*$", ""))
    street = pc.if_else(pc.match_substring_regex(street, rf"^(?:{LANDMARK_RE[2:-2]})"), pa.scalar(""), street)
    st3 = _squash_spaces(_sub(pc.fill_null(st3, ""), r"\b\d+[a-z]*\b", ""))
    street = pc.if_else(pc.equal(street, ""), st3, street)
    return {"postcode": pcode, "house_no": house, "street": street, "body": body}


def address_components(body, n: int):
    """Flatten ', '-separated components: (component text, parent row, position)."""
    comps = pc.split_pattern(body, ", ")
    flat = _squash_spaces(pc.list_flatten(comps))
    return flat, _np(pc.list_parent_indices(comps))


def normalize_addresses(raw, country=None, cache: pa.Table | None = None) -> dict:
    """Everything except city/region, which need the learned region resources and are
    added by `city_region` from `addr_norm` (see normalize_build.py)."""
    raw = _chunked(raw)
    codes = detect_script(raw)
    x = _clean_address_text(_fold(apply_translit_cache(raw, codes, cache)))
    landmark = pc.match_substring_regex(x, LANDMARK_RE)
    parts = parse_address_parts(x)
    parts.pop("body")
    street = parts["street"]
    word = pc.fill_null(pc.struct_field(pc.extract_regex(street, r"^(?P<w>\S+)"), "w"), "")
    hs = pc.if_else(pc.and_(pc.not_equal(parts["house_no"], ""), pc.not_equal(word, "")),
                    pc.binary_join_element_wise(parts["house_no"], word, " "), pa.scalar(""))
    return {"addr_script": script_array(codes), "addr_norm": x, **parts, "street_word": word,
            "landmark": landmark, "hs_key": hs}


JUNK_COMPONENTS = r"^(?:null|none|nan|n a|na|nil|not available|unknown)$"
# A component is region-level (state, province, department...) when it is frequent in its
# country and usually the LAST component: states end ~87-90% of the addresses they appear
# in, cities <= ~31% (measured on all sources, every country incl. France).
REGION_MIN_SHARE, REGION_MIN_LAST_RATIO = 0.0005, 0.6


def component_table(addr_norm, n: int):
    """Components of postcode-stripped addresses with flags: (flat, parents, usable, is_last)."""
    body = _sub(_chunked(addr_norm), PC_STRIP, r"\1")
    flat, parents = address_components(body, n)
    usable = (~_np(pc.match_substring_regex(flat, r"\d")) & _np(pc.not_equal(flat, ""))
              & ~_np(pc.match_substring_regex(flat, JUNK_COMPONENTS)))
    idx = np.nonzero(usable)[0]
    is_last = np.zeros(len(parents), dtype=bool)
    if len(idx):
        rev = idx[::-1]
        _, first = np.unique(parents[rev], return_index=True)
        is_last[rev[first]] = True
    return flat, parents, usable, is_last


def city_region(addr_norm, country, regions: dict | None) -> dict:
    """City = last usable component that is not region-level, a street, a unit or a
    landmark; region = last region-level component, mapped to the S1 spelling when the
    learned map knows it. `regions`: {"region_keys": 'country|comp' strings,
    "region_map": table(src='country|spelling', dst=canonical)}."""
    addr_norm = _chunked(addr_norm)
    n = len(addr_norm)
    flat, parents, usable, _ = component_table(addr_norm, n)
    ctry = pc.take(pc.utf8_lower(pc.cast(_chunked(country), pa.string())), pa.array(parents))
    ckey = pc.binary_join_element_wise(ctry, flat, "|")
    if regions is not None:
        mi = pc.index_in(ckey, value_set=regions["region_map"]["src"])
        canon = pc.if_else(pc.is_valid(mi), pc.take(regions["region_map"]["dst"], mi), flat)
        is_region = _np(pc.or_(pc.is_in(ckey, value_set=regions["region_keys"]), pc.is_valid(mi)))
    else:
        canon, is_region = flat, np.zeros(len(parents), dtype=bool)
    other = _np(pc.match_substring_regex(flat, rf"^(?:{UNIT_WORDS})\b|{LANDMARK_RE}|\b(?:{STREET_TYPES})$"))
    city_ok = usable & ~is_region & ~other

    def last_where(mask, values):
        out = np.full(n, None, dtype=object)
        idx = np.nonzero(mask)[0]
        if len(idx):
            rev = idx[::-1]
            _, first = np.unique(parents[rev], return_index=True)
            out[parents[rev[first]]] = _np(pc.take(values, pa.array(rev[first])))
        return pc.fill_null(pa.array(out, pa.string()), "")

    clean = lambda x: _squash_spaces(_sub(_sub(x, r"^city\s+of\s+", ""), r"\s+city$", ""))
    city = clean(last_where(city_ok, flat))
    # every place-like component (deduplicated, sorted): Indian addresses name different
    # levels of the hierarchy (kolkata / howrah, mumbai / mumbai suburban), so features compare sets
    loc = pd.DataFrame({"p": parents[city_ok], "c": _np(clean(pc.filter(flat, pa.array(city_ok))))})
    loc = loc[loc.c != ""].drop_duplicates().sort_values(["p", "c"], kind="stable")
    localities = _rebuild(pa.array(loc.c.to_numpy(), pa.string()), loc.p.to_numpy(), n, "|")
    return {"city": city, "region": last_where(usable & is_region, canon), "localities": localities}
