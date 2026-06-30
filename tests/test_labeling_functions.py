"""Tests for ade_detection.labeling.labeling_functions.

Rows are synthetic pd.Series that mimic a pre-joined cohort DataFrame where each
row is one visit_occurrence with aggregated icd9_codes and drug_names lists.
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
    lf_routine_discharge,
    lf_sider_curated,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def row(icd9_codes, drug_names, los, discharge):
    return pd.Series(
        {
            "icd9_codes": icd9_codes,
            "drug_names": drug_names,
            "length_of_stay_days": float(los),
            "discharge_disposition": discharge,
        }
    )


# ---------------------------------------------------------------------------
# lf_ecode
# ---------------------------------------------------------------------------


def test_lf_ecode_votes_ade_for_e935():
    # E935.4 = analgesic-induced ADE, integer part 935 ∈ [930, 949]
    r = row(["E935.4", "428.0"], ["Morphine Sulfate 4mg IV"], 6.0, "EXPIRED")
    assert lf_ecode(r) == ADE


def test_lf_ecode_votes_ade_for_e930():
    # Lower boundary of range
    r = row(["E930.0"], ["Vancomycin 1g IVPB"], 8.0, "HOME HEALTH CARE")
    assert lf_ecode(r) == ADE


def test_lf_ecode_abstains_for_non_ecode():
    r = row(["428.0", "250.00"], ["Furosemide 40mg Tab"], 3.0, "ROUTINE")
    assert lf_ecode(r) == ABSTAIN


def test_lf_ecode_abstains_for_e950():
    # E950 = suicide attempt — outside drug-ADE range
    r = row(["E950.0"], ["Acetaminophen 1g Tab"], 2.0, "ROUTINE")
    assert lf_ecode(r) == ABSTAIN


# ---------------------------------------------------------------------------
# lf_ade_diagnosis
# ---------------------------------------------------------------------------


def test_lf_ade_diagnosis_votes_ade_for_995_2x():
    r = row(["995.29", "428.0"], ["Vancomycin HCl 1g IVPB"], 7.0, "HOME HEALTH CARE")
    assert lf_ade_diagnosis(r) == ADE


def test_lf_ade_diagnosis_votes_ade_for_poisoning_range():
    # 965.0 = poisoning by opiates — in 960–979
    r = row(["965.0", "518.81"], ["Fentanyl 25mcg IV"], 8.0, "EXPIRED")
    assert lf_ade_diagnosis(r) == ADE


def test_lf_ade_diagnosis_abstains_for_unrelated_code():
    r = row(["428.0", "250.00"], ["Insulin 10 units SC"], 4.0, "ROUTINE")
    assert lf_ade_diagnosis(r) == ABSTAIN


# ---------------------------------------------------------------------------
# lf_routine_discharge
# ---------------------------------------------------------------------------


def test_lf_routine_discharge_votes_not_ade_short_routine_stay():
    r = row(["428.0"], ["Sodium Chloride 0.9%", "D5W"], 2.0, "ROUTINE")
    assert lf_routine_discharge(r) == NOT_ADE


def test_lf_routine_discharge_votes_not_ade_home_discharge():
    r = row(["410.71"], ["Aspirin 81mg Tab"], 3.0, "HOME")
    assert lf_routine_discharge(r) == NOT_ADE


def test_lf_routine_discharge_abstains_when_los_too_long():
    r = row(["428.0"], ["Sodium Chloride 0.9%"], 8.0, "ROUTINE")
    assert lf_routine_discharge(r) == ABSTAIN


def test_lf_routine_discharge_abstains_when_ecode_present():
    # ADE code present — cannot be called a clean negative
    r = row(["E935.4", "428.0"], ["Morphine 4mg IV"], 3.0, "ROUTINE")
    assert lf_routine_discharge(r) == ABSTAIN


def test_lf_routine_discharge_abstains_non_routine_discharge():
    r = row(["428.0"], ["Furosemide 40mg Tab"], 2.0, "EXPIRED")
    assert lf_routine_discharge(r) == ABSTAIN


# ---------------------------------------------------------------------------
# lf_sider_curated
# ---------------------------------------------------------------------------


def test_lf_sider_curated_votes_ade_warfarin_hemorrhage():
    # anticoagulant_hemorrhage: warfarin (via "Coumadin") + 459.0
    r = row(["459.0", "428.0"], ["Coumadin 5mg"], 5.0, "HOME HEALTH CARE")
    assert lf_sider_curated(r) == ADE


def test_lf_sider_curated_votes_ade_vancomycin_aki():
    # drug_induced_aki: vancomycin + 584.9
    r = row(["584.9"], ["Vancomycin HCl 1250mg IVPB"], 9.0, "HOME HEALTH CARE")
    assert lf_sider_curated(r) == ADE


def test_lf_sider_curated_abstains_drug_without_matching_icd9():
    # Warfarin present but ICD-9 is heart failure (428.0), not hemorrhage
    r = row(["428.0"], ["Coumadin 5mg"], 6.0, "HOME HEALTH CARE")
    assert lf_sider_curated(r) == ABSTAIN


def test_lf_sider_curated_abstains_matching_icd9_without_ade_drug():
    # Hemorrhage code present but no anticoagulant prescribed
    r = row(["459.0"], ["Furosemide 40mg Tab"], 5.0, "HOME HEALTH CARE")
    assert lf_sider_curated(r) == ABSTAIN


# ---------------------------------------------------------------------------
# Cross-LF: saline-only short stay — only lf_routine_discharge fires
# ---------------------------------------------------------------------------


def test_saline_only_short_stay_only_routine_fires():
    r = row(["428.0"], ["Sodium Chloride 0.9%", "Dextrose 5%"], 3.0, "ROUTINE")
    assert lf_ecode(r) == ABSTAIN
    assert lf_ade_diagnosis(r) == ABSTAIN
    assert lf_sider_curated(r) == ABSTAIN
    assert lf_routine_discharge(r) == NOT_ADE


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_get_local_lfs_returns_four():
    lfs = get_local_lfs()
    assert len(lfs) == 4
