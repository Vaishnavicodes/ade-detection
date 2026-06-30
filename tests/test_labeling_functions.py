"""Tests for ade_detection.labeling.labeling_functions.

Rows are synthetic pd.Series that mimic a pre-joined cohort DataFrame where each
row is one visit_occurrence with aggregated icd9_codes and drug_names lists.

ICD-9 codes are in MIMIC dotless format (e.g. "E9354" not "E935.4", "5849" not
"584.9", "99529" not "995.29") — matching what build_labeling_frame produces from
condition_source_value. The normalize_icd9() helper inside each LF strips dots from
pattern prefixes before comparison, so ade_mappings stays human-readable.

Discharge values use real MIMIC ADMISSIONS.DISCHARGE_LOCATION strings
(e.g. "HOME", "HOME HEALTH CARE", "DEAD/EXPIRED", "SNF").
"""

from __future__ import annotations

import pandas as pd

from ade_detection.labeling.labeling_functions import (
    ABSTAIN,
    ADE,
    NOT_ADE,
    get_local_lfs,
    lf_ade_diagnosis,
    lf_ecode,
    lf_elective_admission,
    lf_routine_discharge,
    lf_sider_curated,
    normalize_icd9,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def row(icd9_codes, drug_names, los, discharge, admission_type="EMERGENCY"):
    return pd.Series(
        {
            "icd9_codes": icd9_codes,
            "drug_names": drug_names,
            "length_of_stay_days": float(los),
            "discharge_disposition": discharge,
            "admission_type": admission_type,
        }
    )


# ---------------------------------------------------------------------------
# normalize_icd9
# ---------------------------------------------------------------------------


def test_normalize_icd9_strips_dot():
    assert normalize_icd9("584.9") == "5849"
    assert normalize_icd9("995.29") == "99529"
    assert normalize_icd9("E935.4") == "E9354"


def test_normalize_icd9_already_dotless():
    assert normalize_icd9("5849") == "5849"
    assert normalize_icd9("E9354") == "E9354"


def test_normalize_icd9_uppercases():
    assert normalize_icd9("e9354") == "E9354"


# ---------------------------------------------------------------------------
# lf_ecode  — dotless MIMIC E-codes
# ---------------------------------------------------------------------------


def test_lf_ecode_votes_ade_for_e935():
    # E9354 = drug-induced ADE (E935.4 in dotted notation)
    r = row(["E9354", "4280"], ["Morphine Sulfate 4mg IV"], 6.0, "DEAD/EXPIRED")
    assert lf_ecode(r) == ADE


def test_lf_ecode_votes_ade_for_e930():
    # Lower boundary of drug-ADE E-code range
    r = row(["E9300"], ["Vancomycin 1g IVPB"], 8.0, "HOME HEALTH CARE")
    assert lf_ecode(r) == ADE


def test_lf_ecode_abstains_for_non_ecode():
    r = row(["4280", "25000"], ["Furosemide 40mg Tab"], 3.0, "HOME")
    assert lf_ecode(r) == ABSTAIN


def test_lf_ecode_abstains_for_e950():
    # E9500 = suicide attempt — outside drug-ADE range
    r = row(["E9500"], ["Acetaminophen 1g Tab"], 2.0, "HOME")
    assert lf_ecode(r) == ABSTAIN


# ---------------------------------------------------------------------------
# lf_ade_diagnosis  — dotless MIMIC diagnosis codes
# ---------------------------------------------------------------------------


def test_lf_ade_diagnosis_votes_ade_for_995_2x():
    # 99529 = 995.29 adverse effect of other correct medicinal substance
    r = row(["99529", "4280"], ["Vancomycin HCl 1g IVPB"], 7.0, "HOME HEALTH CARE")
    assert lf_ade_diagnosis(r) == ADE


def test_lf_ade_diagnosis_votes_ade_for_poisoning_range():
    # 9650 = poisoning by opiates (965.0 in dotted notation) — in 960–979
    r = row(["9650", "51881"], ["Fentanyl 25mcg IV"], 8.0, "DEAD/EXPIRED")
    assert lf_ade_diagnosis(r) == ADE


def test_lf_ade_diagnosis_abstains_for_unrelated_code():
    r = row(["4280", "25000"], ["Insulin 10 units SC"], 4.0, "HOME")
    assert lf_ade_diagnosis(r) == ABSTAIN


# ---------------------------------------------------------------------------
# lf_routine_discharge  — uses real MIMIC DISCHARGE_LOCATION values
# ---------------------------------------------------------------------------


def test_lf_routine_discharge_votes_not_ade_short_home_stay():
    # "HOME" is a routine discharge in MIMIC; LOS<=4 and no ADE codes → NOT_ADE
    r = row(["4280"], ["Sodium Chloride 0.9%", "D5W"], 2.0, "HOME")
    assert lf_routine_discharge(r) == NOT_ADE


def test_lf_routine_discharge_votes_not_ade_home_health_care():
    # "HOME HEALTH CARE" also counts as home discharge
    r = row(["41071"], ["Aspirin 81mg Tab"], 3.0, "HOME HEALTH CARE")
    assert lf_routine_discharge(r) == NOT_ADE


def test_lf_routine_discharge_abstains_when_los_too_long():
    r = row(["4280"], ["Sodium Chloride 0.9%"], 8.0, "HOME")
    assert lf_routine_discharge(r) == ABSTAIN


def test_lf_routine_discharge_abstains_when_ecode_present():
    # ADE code present — cannot be a clean negative even with short LOS
    r = row(["E9354", "4280"], ["Morphine 4mg IV"], 3.0, "HOME")
    assert lf_routine_discharge(r) == ABSTAIN


def test_lf_routine_discharge_abstains_for_dead_expired():
    # DEAD/EXPIRED is not a routine discharge
    r = row(["4280"], ["Furosemide 40mg Tab"], 2.0, "DEAD/EXPIRED")
    assert lf_routine_discharge(r) == ABSTAIN


def test_lf_routine_discharge_abstains_for_snf():
    # SNF (skilled nursing facility) transfer is not routine home discharge
    r = row(["4280"], ["Furosemide 40mg Tab"], 2.0, "SNF")
    assert lf_routine_discharge(r) == ABSTAIN


# ---------------------------------------------------------------------------
# lf_sider_curated  — dotless MIMIC ICD-9, dotted prefixes in ade_mappings
# ---------------------------------------------------------------------------


def test_lf_sider_curated_votes_ade_warfarin_hemorrhage():
    # anticoagulant_hemorrhage: warfarin (via "Coumadin") + 4590 (459.0 dotless)
    r = row(["4590", "4280"], ["Coumadin 5mg"], 5.0, "HOME HEALTH CARE")
    assert lf_sider_curated(r) == ADE


def test_lf_sider_curated_votes_ade_vancomycin_aki():
    # drug_induced_aki: vancomycin + 5849 (584.9 dotless)
    r = row(["5849"], ["Vancomycin HCl 1250mg IVPB"], 9.0, "HOME HEALTH CARE")
    assert lf_sider_curated(r) == ADE


def test_lf_sider_curated_abstains_drug_without_matching_icd9():
    # Warfarin present but ICD-9 is heart failure (4280), not hemorrhage
    r = row(["4280"], ["Coumadin 5mg"], 6.0, "HOME HEALTH CARE")
    assert lf_sider_curated(r) == ABSTAIN


def test_lf_sider_curated_abstains_matching_icd9_without_ade_drug():
    # Hemorrhage code present but no anticoagulant prescribed
    r = row(["4590"], ["Furosemide 40mg Tab"], 5.0, "HOME HEALTH CARE")
    assert lf_sider_curated(r) == ABSTAIN


# ---------------------------------------------------------------------------
# lf_elective_admission
# ---------------------------------------------------------------------------


def test_lf_elective_admission_votes_not_ade_clean_elective():
    # Pre-scheduled procedure, no ADE codes → clean NOT_ADE negative
    r = row(["4280", "41401"], ["Aspirin 81mg Tab"], 2.0, "HOME", admission_type="ELECTIVE")
    assert lf_elective_admission(r) == NOT_ADE


def test_lf_elective_admission_abstains_when_ecode_present():
    # Elective admission but E-code present — cannot be a clean negative
    r = row(["E9354", "4280"], ["Morphine 4mg IV"], 3.0, "HOME", admission_type="ELECTIVE")
    assert lf_elective_admission(r) == ABSTAIN


def test_lf_elective_admission_abstains_when_ade_diagnosis_present():
    # Elective but 995.2x code present
    r = row(
        ["99529", "4280"], ["Vancomycin HCl 1g"], 5.0, "HOME HEALTH CARE", admission_type="ELECTIVE"
    )
    assert lf_elective_admission(r) == ABSTAIN


def test_lf_elective_admission_abstains_for_emergency():
    # Emergency admission — even with no ADE codes, wrong type → ABSTAIN
    r = row(["4280"], ["Furosemide 40mg Tab"], 2.0, "HOME", admission_type="EMERGENCY")
    assert lf_elective_admission(r) == ABSTAIN


def test_lf_elective_admission_abstains_for_urgent():
    r = row(["4280"], ["Metoprolol 25mg Tab"], 1.0, "HOME", admission_type="URGENT")
    assert lf_elective_admission(r) == ABSTAIN


# ---------------------------------------------------------------------------
# Cross-LF: saline-only short stay — only lf_routine_discharge fires
# ---------------------------------------------------------------------------


def test_saline_only_short_stay_only_routine_fires():
    r = row(["4280"], ["Sodium Chloride 0.9%", "Dextrose 5%"], 3.0, "HOME")
    assert lf_ecode(r) == ABSTAIN
    assert lf_ade_diagnosis(r) == ABSTAIN
    assert lf_sider_curated(r) == ABSTAIN
    assert lf_routine_discharge(r) == NOT_ADE
    assert lf_elective_admission(r) == ABSTAIN  # type is EMERGENCY (default)


def test_elective_clean_admission_two_negatives_fire():
    # Both negative LFs should fire on a clean short elective admission
    r = row(["4280"], ["Aspirin 81mg Tab"], 3.0, "HOME", admission_type="ELECTIVE")
    assert lf_ecode(r) == ABSTAIN
    assert lf_ade_diagnosis(r) == ABSTAIN
    assert lf_sider_curated(r) == ABSTAIN
    assert lf_routine_discharge(r) == NOT_ADE
    assert lf_elective_admission(r) == NOT_ADE


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_get_local_lfs_returns_five():
    lfs = get_local_lfs()
    assert len(lfs) == 5


def test_get_local_lfs_names():
    names = [lf.name for lf in get_local_lfs()]
    assert "lf_ecode" in names
    assert "lf_ade_diagnosis" in names
    assert "lf_routine_discharge" in names
    assert "lf_elective_admission" in names
    assert "lf_sider_curated" in names
