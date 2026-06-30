"""Tests for ade_detection.labeling.ade_mappings."""

from ade_detection.labeling.ade_mappings import get_all_ade_drugs, get_patterns

REQUIRED_KEYS = {"name", "drugs", "icd9_prefixes", "rationale"}


def test_pattern_count():
    assert len(get_patterns()) == 9


def test_each_pattern_has_required_keys():
    for pattern in get_patterns():
        assert (
            REQUIRED_KEYS <= pattern.keys()
        ), f"Pattern '{pattern.get('name')}' missing keys: {REQUIRED_KEYS - pattern.keys()}"


def test_drugs_are_lowercase():
    for pattern in get_patterns():
        for drug in pattern["drugs"]:
            assert (
                drug == drug.lower()
            ), f"Drug '{drug}' in pattern '{pattern['name']}' is not lowercase"


def test_icd9_prefixes_non_empty():
    for pattern in get_patterns():
        assert (
            len(pattern["icd9_prefixes"]) >= 1
        ), f"Pattern '{pattern['name']}' has empty icd9_prefixes"


def test_icd9_prefixes_are_strings():
    for pattern in get_patterns():
        for code in pattern["icd9_prefixes"]:
            assert isinstance(
                code, str
            ), f"ICD-9 prefix {code!r} in '{pattern['name']}' is not a string"


def test_get_all_ade_drugs_non_empty():
    drugs = get_all_ade_drugs()
    assert isinstance(drugs, set)
    assert len(drugs) > 0


def test_get_all_ade_drugs_are_lowercase():
    for drug in get_all_ade_drugs():
        assert drug == drug.lower(), f"Drug '{drug}' in get_all_ade_drugs() is not lowercase"


def test_known_drugs_present():
    drugs = get_all_ade_drugs()
    assert "vancomycin" in drugs
    assert "warfarin" in drugs
    assert "insulin" in drugs


def test_pattern_names_unique():
    names = [p["name"] for p in get_patterns()]
    assert len(names) == len(set(names)), "Duplicate pattern names found"
