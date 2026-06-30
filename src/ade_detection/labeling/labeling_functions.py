"""Snorkel labeling functions for per-admission ADE detection.

Each LF operates on a single admission row (pd.Series) with pre-joined fields:

  icd9_codes              list[str]   condition_source_value codes for the admission
  drug_names              list[str]   raw drug strings from drug_exposure
  length_of_stay_days     float       (discharge_datetime - admit_datetime).days
  discharge_disposition   str         MIMIC ADMISSIONS.DISCHARGE_LOCATION

Apply all four LFs via snorkel.labeling.PandasLFApplier on the cohort DataFrame:

    applier = PandasLFApplier(lfs=get_local_lfs())
    L = applier.apply(cohort_df)          # L shape: (n_admissions, 4)
    label_model = LabelModel(cardinality=2)
    label_model.fit(L)
    probs = label_model.predict_proba(L)  # ade_prob per admission
"""

from __future__ import annotations

try:
    from snorkel.labeling import labeling_function
except ImportError:  # pragma: no cover — snorkel on CI/Colab; shim for bare local env

    def labeling_function(func=None, **kwargs):
        """Lightweight shim: makes module importable without snorkel installed."""

        def _wrap(f):
            f.f = f
            return f

        return _wrap(func) if func is not None else _wrap


from ade_detection.labeling.ade_mappings import get_patterns
from ade_detection.labeling.drug_normalization import match_to_ade_drug

# ---------------------------------------------------------------------------
# Label constants (mirrors snorkel.labeling.ABSTAIN / cardinality-2 convention)
# ---------------------------------------------------------------------------

ABSTAIN = -1
NOT_ADE = 0
ADE = 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_ecode_ade(code: str) -> bool:
    """True if *code* is an ICD-9 E-code in the drug-ADE range E930–E949."""
    code = code.strip().upper()
    if not code.startswith("E"):
        return False
    try:
        return 930 <= int(code[1:].split(".")[0]) <= 949
    except ValueError:
        return False


def _is_ade_diagnosis_code(code: str) -> bool:
    """True if *code* is 995.2x (adverse drug effect) or 960–979 (drug poisoning)."""
    code = code.strip()
    if code.startswith("995.2"):
        return True
    try:
        return 960 <= int(code.split(".")[0]) <= 979
    except ValueError:
        return False


def _has_ade_codes(icd9_codes: list) -> bool:
    """True if any code would fire lf_ecode or lf_ade_diagnosis."""
    for code in icd9_codes:
        s = str(code).strip()
        if not s or s.lower() in ("nan", "none"):
            continue
        if _is_ecode_ade(s) or _is_ade_diagnosis_code(s):
            return True
    return False


# ---------------------------------------------------------------------------
# Labeling functions
# ---------------------------------------------------------------------------


@labeling_function()
def lf_ecode(x) -> int:
    """ADE if any ICD-9 code is a drug-induced adverse-event E-code (E930–E949).

    High-precision, moderate-recall signal from structured diagnosis coding.
    """
    for code in x.icd9_codes:
        if _is_ecode_ade(str(code)):
            return ADE
    return ABSTAIN


@labeling_function()
def lf_ade_diagnosis(x) -> int:
    """ADE if any ICD-9 code is 995.2x (adverse drug effect) or 960–979 (drug poisoning).

    Complementary to lf_ecode: catches explicitly coded ADEs not using E-code convention.
    """
    for code in x.icd9_codes:
        if _is_ade_diagnosis_code(str(code)):
            return ADE
    return ABSTAIN


@labeling_function()
def lf_routine_discharge(x) -> int:
    """NOT_ADE negative anchor: short stay + routine/home discharge + no ADE codes.

    Conditions (all must hold):
      - length_of_stay_days <= 4
      - discharge_disposition contains "routine" or "home" (case-insensitive)
      - no ICD-9 code fires lf_ecode or lf_ade_diagnosis

    Provides strong negative supervision to balance the label distribution.
    """
    discharge = str(x.discharge_disposition).lower()
    is_routine = "routine" in discharge or "home" in discharge
    if x.length_of_stay_days <= 4 and is_routine and not _has_ade_codes(x.icd9_codes):
        return NOT_ADE
    return ABSTAIN


@labeling_function()
def lf_sider_curated(x) -> int:
    """ADE if the admission has BOTH a drug AND a co-occurring ICD-9 from the same SIDER pattern.

    For each of the 9 curated ade_mappings patterns, checks:
      (a) any drug_name normalizes to a drug in that pattern, AND
      (b) any icd9_code matches at least one of that pattern's icd9_prefixes.

    The conjunction requirement makes this a high-precision, causal signal:
    a known nephrotoxic drug PLUS an AKI code is far more likely to be a true ADE
    than either signal alone.
    """
    for pattern in get_patterns():
        drug_set = set(pattern["drugs"])
        drug_matched = any(
            match_to_ade_drug(raw_drug, drug_set) is not None for raw_drug in x.drug_names
        )
        if not drug_matched:
            continue
        for code in x.icd9_codes:
            for prefix in pattern["icd9_prefixes"]:
                if str(code).strip().startswith(prefix):
                    return ADE
    return ABSTAIN


# ---------------------------------------------------------------------------
# LF registry
# ---------------------------------------------------------------------------


def get_local_lfs() -> list:
    """Return all local (non-note) labeling functions for PandasLFApplier.

    Note-based LFs (lf_note_mention) run on Colab and are not included here.
    """
    return [lf_ecode, lf_ade_diagnosis, lf_routine_discharge, lf_sider_curated]
