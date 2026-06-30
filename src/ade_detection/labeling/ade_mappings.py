"""Curated drug→injury ADE patterns for labeling function LF#4.

These pairs are clinically curated and grounded in the SIDER side-effect
resource (v4.1); ICD-9 codes are verified against ICD-9-CM references.
Scope is high-precision v1 — only well-established mechanism-confirmed
drug–event pairs are included.  A full UMLS→ICD-9 crosswalk to broaden
recall is documented as future work.

ICD-9 matching is prefix-based: a condition_source_value starting with a
listed prefix is considered a match (e.g. "584.5" matches "584.5", "584.51").
"""

from __future__ import annotations

_ADE_PATTERNS: list[dict] = [
    {
        "name": "drug_induced_aki",
        "drugs": [
            "vancomycin",
            "gentamicin",
            "tobramycin",
            "ibuprofen",
            "ketorolac",
            "lisinopril",
            "enalapril",
        ],
        "icd9_prefixes": ["584.5", "584.6", "584.7", "584.8", "584.9"],
        "rationale": ("Nephrotoxic agents or renal-perfusion reducers cause acute tubular injury."),
    },
    {
        "name": "drug_induced_liver_injury",
        "drugs": ["acetaminophen", "isoniazid", "amiodarone", "valproate"],
        "icd9_prefixes": ["570", "573.3"],
        "rationale": "Hepatocellular toxicity / acute hepatic necrosis.",
    },
    {
        "name": "anticoagulant_hemorrhage",
        "drugs": ["warfarin", "heparin", "enoxaparin"],
        "icd9_prefixes": ["459.0", "578", "998.1"],
        "rationale": "Impaired clotting causing major bleeding.",
    },
    {
        "name": "opioid_respiratory_depression",
        "drugs": ["morphine", "fentanyl", "hydromorphone", "oxycodone"],
        "icd9_prefixes": ["518.81"],
        "rationale": "Opioid suppression of brainstem respiratory drive.",
    },
    {
        "name": "drug_induced_hypoglycemia",
        "drugs": ["insulin", "glipizide", "glyburide"],
        "icd9_prefixes": ["251.0", "251.2"],
        "rationale": "Excess insulin/sulfonylurea drives glucose too low.",
    },
    {
        "name": "antibiotic_cdiff",
        "drugs": ["clindamycin", "ciprofloxacin", "ceftriaxone", "levofloxacin"],
        "icd9_prefixes": ["008.45"],
        "rationale": ("Antibiotics disrupt gut flora enabling C. difficile overgrowth."),
    },
    {
        "name": "drug_induced_thrombocytopenia",
        "drugs": ["heparin", "valproate", "linezolid"],
        "icd9_prefixes": ["287.49", "289.84"],
        "rationale": ("Drug-triggered immune platelet destruction (incl. HIT)."),
    },
    {
        "name": "drug_induced_hyperkalemia",
        "drugs": ["lisinopril", "enalapril", "spironolactone"],
        "icd9_prefixes": ["276.7"],
        "rationale": "Reduced potassium excretion raising serum potassium.",
    },
    {
        "name": "drug_anaphylaxis",
        "drugs": ["penicillin", "amoxicillin", "sulfamethoxazole", "vancomycin"],
        "icd9_prefixes": ["995.0"],
        "rationale": "Immune-mediated anaphylactic reaction to a drug.",
    },
]


def get_patterns() -> list[dict]:
    """Return the full list of ADE pattern dicts."""
    return _ADE_PATTERNS


def get_all_ade_drugs() -> set[str]:
    """Return every drug name appearing in any ADE pattern (lowercase)."""
    return {drug for pattern in _ADE_PATTERNS for drug in pattern["drugs"]}
