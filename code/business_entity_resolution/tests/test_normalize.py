"""Unit tests for src/normalize.py on real noise patterns seen in EDA.  Run: python -m pytest tests -q"""
import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normalize import (SCRIPT_LABELS, detect_script, is_indic, normalize_addresses,  # noqa: E402
                       normalize_names, transliterate_one)


def names(values, lexicon=None):
    raw = pa.array(values)
    codes = detect_script(raw)
    ind = is_indic(codes)
    vals = [v for v, i in zip(values, ind) if i]
    cache = pa.table({"raw": pa.array(vals, pa.string()),
                      "latin": pa.array([transliterate_one(v, SCRIPT_LABELS[c]) for v, c, i in zip(values, codes, ind) if i], pa.string())})
    out = normalize_names(raw, cache, lexicon)
    return [{k: v[i].as_py() for k, v in out.items()} for i in range(len(values))]


def addresses(values, country="US"):
    out = normalize_addresses(pa.array(values))
    return [{k: v[i].as_py() for k, v in out.items()} for i in range(len(values))]


@pytest.mark.parametrize("raw,core,legal", [
    ("Harbor Dóuglas (LLC)", "harbor douglas", "llc"),
    ("Harbor LLC Douglas", "harbor douglas", "llc"),
    ("Delta World-L.L.C.", "delta world", "llc"),
    ("Shree Limited Private  Information", "shree information", "pvt_ltd"),
    ("M/s Great Food Limited Private", "great food", "pvt_ltd"),
    ("Tele + Sons Pvt.  Ltd", "tele and sons", "pvt_ltd"),
    ("Jkm & Co Limited", "jkm", "co|ltd"),
    ("Fractales Amis Groupe S.A.S", "fractales amis groupe", "sas"),
    ("<< Team Ecole", "team ecole", ""),
    ("Orelee's Barbershop", "orelees barbershop", ""),
    ("Co", "co", "co"),                                        # nothing but a legal token: keep it
])
def test_core_and_legal_form(raw, core, legal):
    r = names([raw])[0]
    assert r["name_core"] == core
    assert r["legal_form"] == legal


def test_alias_split():
    r = names(["Quoviolyra fka Harbor Douglas LLC", "Evoyuma D.B.A. Timber",
               "Novitavonovi doing business as Universal Industries Private Limited"])
    assert (r[0]["name_core"], r[0]["alias_core"], r[0]["legal_form"]) == ("quoviolyra", "harbor douglas", "llc")
    assert (r[1]["name_core"], r[1]["alias_core"]) == ("evoyuma", "timber")
    assert (r[2]["alias_core"], r[2]["legal_form"]) == ("universal industries", "pvt_ltd")


def test_domains_and_titles():
    r = names(["héartassociation.com", "earnosethroat com", "Hannah K. Diaz, DO", "Hannah K. Diaz, GO", "Do It Best"])
    assert (r[0]["name_core"], r[0]["is_domain"]) == ("heartassociation", True)
    assert (r[1]["name_core"], r[1]["is_domain"]) == ("earnosethroat", True)
    assert (r[2]["name_core"], r[2]["titles"]) == ("hannah k diaz", "do")
    assert (r[3]["name_core"], r[3]["titles"]) == ("hannah k diaz go", "")     # GO is not a title: a real difference
    assert r[4]["name_core"] == "do it best"                                    # a leading 'do' is not a title


def test_transliteration_scripts():
    r = names(["राम मार्केटिंग प्राइवेट लिमिटेड", "ਈਸਟ ਇਨਵੈਸਟਮੈਂਟਸ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ",
               "માય પ્રોડ્યુસર પ્રાઇવેટ લિમિટેડ", "গ্রেট ফুড প্রাইভেট লিমিটেড", "मॉडर्न फाइनेंस"])
    assert [x["name_script"] for x in r] == ["devanagari", "gurmukhi", "gujarati", "bengali", "devanagari"]
    assert r[0]["name_norm"].startswith("ram marketing")
    assert "limited" in r[0]["name_norm"] and "limited" in r[3]["name_norm"]
    assert r[4]["name_norm"].startswith("modarn")                          # candra-o handled (no raw 'ॉ')
    assert all(x["name_norm"].isascii() for x in r)


def test_lexicon_maps_transliterated_tokens():
    lex = pa.table({"src": ["praivet", "limtid"], "dst": ["private", "limited"]})
    r = names(["राम मार्केटिंग प्राइवेट लिमिटेड", "Praivet Cafe"], lexicon=lex)
    assert r[0]["name_core"] == "ram marketing" and r[0]["legal_form"] == "pvt_ltd"
    assert r[1]["name_core"] == "praivet cafe"                             # Latin rows are never rewritten


@pytest.mark.parametrize("raw,house,street,postcode", [
    ("3908-3910 URANUS AVE, VANDENBERG VILLAGE, CA", "3908", "uranus avenue", ""),
    ("0022632 OBSERVATION DRIVE, CLARKSBURG, MD", "22632", "observation drive", ""),
    ("006, OM SAI NAGAR, BAMROLI ROAD, SURAT, Gujarat", "6", "om sai nagar", ""),
    ("216 HAYES SAINT, BOZEMAN, MT", "216", "hayes street", ""),
    ("216 Hayes St, Bozeman, Montana", "216", "hayes street", ""),
    ("IA, Iowa City, 1064 Newton Rd, Unit 11", "1064", "newton road", ""),
    ("Tyler, TX 75701", "", "", "75701"),
    ("12 Rue de la Paix, 75002 Paris", "12", "rue de la paix", "75002"),
    ("PO BOX 9820, MEMPHIS, TN, 716 COFFEEVILLE CV", "716", "coffeeville cove", ""),
    ("4Th Floor, No 97, Sanjayanagar Main Road, Bangalore, Karnataka", "97", "sanjayanagar main road", ""),
    ("A-303, MARATHON CHAMBERS, MUMBAI, Maharashtra", "303", "marathon chambers", ""),
    ("73rd Street, Minot, North Dakota", "", "street", ""),
    ("Prashil Park, Block No. 92, Near Saurashtra University, Kalawad Road, Rajkot", "92", "kalawad road", ""),
])
def test_address_parts(raw, house, street, postcode):
    r = addresses([raw])[0]
    assert (r["house_no"], r["street"], r["postcode"]) == (house, street, postcode)


def test_state_codes_not_expanded_and_hs_key():
    a, b, c = addresses(["216 HAYES SAINT, BOZEMAN, MT", "216 Hayes St, Bozeman, Montana", "1 Main St, Hartford, CT"])
    assert a["hs_key"] == b["hs_key"] == "216 hayes"
    assert a["addr_norm"].endswith(", mt") and c["addr_norm"].endswith(", ct")


def test_landmark_flag():
    r = addresses(["Nr SBI ATM, Sector 5, Noida", "Near Saurashtra University, Rajkot", "105 Elm St, Morganton, NC"])
    assert [x["landmark"] for x in r] == [True, True, False]


def test_mixed_script_keeps_latin_words():
    from normalize import transliterate_one
    out = transliterate_one("Prashil Park, Ahmedabad, ગુજરાત")
    assert out.startswith("Prashil Park, Ahmedabad, ") and out.endswith("gujrat")   # Latin words untouched
    assert transliterate_one("महाराष्ट्र") == "maharastr"                               # long ā is never dropped
    assert transliterate_one("KOLKATA, পশ্চিমবঙ্গ").startswith("KOLKATA, ")


def test_city_region_localities():
    import pyarrow as pa
    from normalize import city_region, normalize_addresses
    a = normalize_addresses(pa.array(["13/12 Chinar Park, Kolkata, Howrah, West Bengal", "1621 Hudson Street, Columbus, OH"]))
    regions = {"region_keys": pa.array(["india|west bengal", "us|oh"]),
               "region_map": pa.table({"src": pa.array([], pa.string()), "dst": pa.array([], pa.string())})}
    g = city_region(a["addr_norm"], pa.array(["India", "US"]), regions)
    assert g["city"].to_pylist() == ["howrah", "columbus"]
    assert g["region"].to_pylist() == ["west bengal", "oh"]
    assert g["localities"].to_pylist() == ["howrah|kolkata", "columbus"]   # house-number component excluded
