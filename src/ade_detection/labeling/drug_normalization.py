"""Drug name normalization for matching MIMIC PRESCRIPTIONS to ADE mappings.

MIMIC drug strings are free-text and include brand names, salt forms, dosage
strengths, and route/formulation suffixes (e.g. "Vancomycin HCl 1250mg IVPB").
This module cleans those strings to lowercase generic names so they can be
compared against the curated drug lists in ade_mappings.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Brand → generic mapping
# ---------------------------------------------------------------------------

BRAND_TO_GENERIC: dict[str, str] = {
    # Analgesics / antipyretics
    "tylenol": "acetaminophen",
    "ofirmev": "acetaminophen",
    # NSAIDs
    "motrin": "ibuprofen",
    "advil": "ibuprofen",
    "toradol": "ketorolac",
    # Anticoagulants
    "coumadin": "warfarin",
    "jantoven": "warfarin",
    "lovenox": "enoxaparin",
    # Opioids
    "dilaudid": "hydromorphone",
    "exalgo": "hydromorphone",
    "duragesic": "fentanyl",
    "oxycontin": "oxycodone",
    "percocet": "oxycodone",
    "roxicodone": "oxycodone",
    "ms contin": "morphine",
    "kadian": "morphine",
    # Insulins — multiple brand/subtype names all map to generic "insulin"
    "lantus": "insulin",
    "novolog": "insulin",
    "humulin": "insulin",
    "humalog": "insulin",
    "levemir": "insulin",
    "toujeo": "insulin",
    "tresiba": "insulin",
    "apidra": "insulin",
    "novolin": "insulin",
    "glargine": "insulin",
    "aspart": "insulin",
    "lispro": "insulin",
    "detemir": "insulin",
    # Sulfonylureas
    "glucotrol": "glipizide",
    "diabeta": "glyburide",
    "micronase": "glyburide",
    "glynase": "glyburide",
    # ACE inhibitors
    "prinivil": "lisinopril",
    "zestril": "lisinopril",
    "vasotec": "enalapril",
    # Diuretics
    "lasix": "furosemide",
    "aldactone": "spironolactone",
    "carospir": "spironolactone",
    # Antibiotics
    "cipro": "ciprofloxacin",
    "levaquin": "levofloxacin",
    "rocephin": "ceftriaxone",
    "cleocin": "clindamycin",
    "zyvox": "linezolid",
    "bactrim": "sulfamethoxazole",
    "septra": "sulfamethoxazole",
    "amoxil": "amoxicillin",
    "trimox": "amoxicillin",
    "augmentin": "amoxicillin",
    "ancef": "cefazolin",
    "kefzol": "cefazolin",
    "vancocin": "vancomycin",
    "firvanq": "vancomycin",
    "garamycin": "gentamicin",
    "nebcin": "tobramycin",
    "tobi": "tobramycin",
    # Anticonvulsants / mood stabilizers
    "depakote": "valproate",
    "depakene": "valproate",
    "depacon": "valproate",
    # Benzodiazepines
    "ativan": "lorazepam",
    "valium": "diazepam",
    "versed": "midazolam",
    # Penicillins
    "pen vk": "penicillin",
    "bicillin": "penicillin",
    "pfizerpen": "penicillin",
}

# ---------------------------------------------------------------------------
# Cleaning patterns
# ---------------------------------------------------------------------------

# Dosage/strength tokens: "325mg", "40mg/0.4mL", "0.9%", "1000 units", etc.
_DOSAGE_RE = re.compile(
    r"\d+(?:\.\d+)?"
    r"\s*(?:mg|mcg|ml|g|units?|%|meq|mmol|miu|iu)"
    r"(?:\s*/\s*\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|%|meq))?",
    re.IGNORECASE,
)

# Route and formulation tokens stripped as whole words.
# Excludes electrolyte terms (sodium, chloride) that are standalone drug names.
_FORM_TOKENS: frozenset[str] = frozenset(
    {
        # Routes
        "iv",
        "ivpb",
        "po",
        "sc",
        "sq",
        "im",
        "sl",
        "pr",
        "ng",
        "intravenous",
        "subcutaneous",
        "intramuscular",
        # Solid forms
        "tab",
        "tabs",
        "tablet",
        "tablets",
        "cap",
        "caps",
        "capsule",
        "capsules",
        # Liquid/injectable forms
        "injection",
        "injectable",
        "infusion",
        "solution",
        "soln",
        "sol",
        "syringe",
        "syringes",
        "suspension",
        "susp",
        "vial",
        "ampule",
        "bag",
        # Topical forms
        "patch",
        "cream",
        "ointment",
        "gel",
        "oral",
        "topical",
        # Release modifiers
        "extended",
        "immediate",
        "release",
        "er",
        "xr",
        "sr",
        "cr",
        # Salt suffixes — never a standalone drug name
        "hcl",
        "hydrochloride",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_drug_name(raw: str) -> str:
    """Normalize a raw MIMIC drug string to a clean lowercase generic name.

    Steps:
      1. Lowercase and strip.
      2. Extract parenthetical content (may hold a brand name like "(Lantus)").
      3. Remove dosage/strength tokens (e.g. "325mg", "40mg/0.4mL", "0.9%").
      4. Normalize punctuation (hyphens, slashes, commas → spaces).
      5. Remove route/formulation tokens word by word.
      6. Look up the result (and parenthetical words) in BRAND_TO_GENERIC.
      7. Return the generic name, or the cleaned string if no brand match.

    Parameters
    ----------
    raw:
        Free-text drug name as it appears in MIMIC PRESCRIPTIONS.drug.

    Returns
    -------
    str
        Best-effort lowercase generic name.
    """
    s = raw.lower().strip()

    # Pull parenthetical content before stripping parens; may reveal brand names.
    parens_content = re.findall(r"\(([^)]+)\)", s)
    s = re.sub(r"\([^)]+\)", " ", s)

    # Remove dosage/strength tokens.
    s = _DOSAGE_RE.sub(" ", s)

    # Hyphens, slashes, commas → spaces (handles "Acetaminophen-Caffeine", "40mg/0.4mL").
    s = re.sub(r"[-/,]", " ", s)

    # Remove any bare numbers left after dosage stripping.
    s = re.sub(r"\b\d+(?:\.\d+)?\b", " ", s)

    # Strip route/form tokens.
    words = [w for w in s.split() if w not in _FORM_TOKENS]
    s = " ".join(words).strip()

    # Build brand-lookup candidates: full cleaned string, each word, then parens.
    candidates: list[str] = [s] + s.split()
    for p in parens_content:
        p = p.strip()
        candidates.append(p)
        candidates.extend(p.split())

    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate in BRAND_TO_GENERIC:
            return BRAND_TO_GENERIC[candidate]

    return s


def match_to_ade_drug(raw: str, ade_drugs: set[str]) -> str | None:
    """Return the matching ADE drug name if *raw* normalizes to a known ADE drug.

    Uses substring containment after normalization so that "morphine sulfate"
    still matches "morphine".  Returns ``None`` (abstain) when no match is
    found — this is the intended behavior for LF#4.

    Parameters
    ----------
    raw:
        Raw drug string from MIMIC PRESCRIPTIONS.
    ade_drugs:
        Set of lowercase generic drug names to match against (e.g. from
        ``ade_mappings.get_all_ade_drugs()``).

    Returns
    -------
    str | None
        The matched ADE drug name, or ``None`` if no match.
    """
    normalized = normalize_drug_name(raw)
    for drug in ade_drugs:
        if drug in normalized:
            return drug
    return None
