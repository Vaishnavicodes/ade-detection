"""Tests for ade_detection.labeling.drug_normalization.

Uses realistic MIMIC PRESCRIPTIONS-style drug strings.
"""

from ade_detection.labeling.ade_mappings import get_all_ade_drugs
from ade_detection.labeling.drug_normalization import match_to_ade_drug, normalize_drug_name

# ---------------------------------------------------------------------------
# normalize_drug_name
# ---------------------------------------------------------------------------


def test_strips_dosage_and_form():
    assert normalize_drug_name("Acetaminophen 325mg Tab") == "acetaminophen"


def test_strips_iv_route():
    result = normalize_drug_name("Morphine Sulfate IV")
    assert "morphine" in result


def test_brand_coumadin_to_warfarin():
    assert normalize_drug_name("Coumadin 5mg") == "warfarin"


def test_brand_lovenox_with_fraction_dose():
    assert normalize_drug_name("Lovenox 40mg/0.4mL Syringe") == "enoxaparin"


def test_brand_in_parens_lantus():
    # Brand name inside parentheses should be resolved via BRAND_TO_GENERIC.
    assert normalize_drug_name("Insulin Glargine (Lantus)") == "insulin"


def test_hyphenated_combo_drug():
    # Combo drug — no brand match, but the primary ingredient is preserved.
    result = normalize_drug_name("Acetaminophen-Caffeine")
    assert "acetaminophen" in result


def test_brand_cipro():
    assert normalize_drug_name("Cipro 500mg Tab") == "ciprofloxacin"


def test_brand_dilaudid_ivpb():
    assert normalize_drug_name("Dilaudid 0.5mg IVPB") == "hydromorphone"


def test_brand_depakote():
    assert normalize_drug_name("Depakote 500mg ER Tab") == "valproate"


def test_humulin_insulin():
    assert normalize_drug_name("Humulin R 100 units/mL Injection") == "insulin"


def test_novolog_insulin():
    assert normalize_drug_name("NovoLog 10mL Vial") == "insulin"


def test_vancomycin_already_generic():
    result = normalize_drug_name("Vancomycin HCl 1250mg IVPB")
    assert result == "vancomycin"


def test_lowercase_output():
    result = normalize_drug_name("WARFARIN SODIUM 5MG TABLET")
    assert result == result.lower()


# ---------------------------------------------------------------------------
# match_to_ade_drug
# ---------------------------------------------------------------------------


def test_match_returns_none_for_saline():
    ade_drugs = get_all_ade_drugs()
    assert match_to_ade_drug("Sodium Chloride 0.9%", ade_drugs) is None


def test_match_morphine():
    ade_drugs = get_all_ade_drugs()
    assert match_to_ade_drug("Morphine Sulfate 4mg IV", ade_drugs) == "morphine"


def test_match_brand_resolves_to_ade_drug():
    ade_drugs = get_all_ade_drugs()
    assert match_to_ade_drug("Coumadin 5mg", ade_drugs) == "warfarin"


def test_match_unrelated_drug_returns_none():
    ade_drugs = get_all_ade_drugs()
    assert match_to_ade_drug("Pantoprazole 40mg Tab", ade_drugs) is None


def test_match_vancomycin():
    ade_drugs = get_all_ade_drugs()
    assert match_to_ade_drug("Vancomycin HCl 1g IVPB", ade_drugs) == "vancomycin"
