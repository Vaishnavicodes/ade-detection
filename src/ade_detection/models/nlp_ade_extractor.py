"""ClinicalBERT-based NLP ADE extractor — STUB.

ENVIRONMENT: Colab (GPU strongly recommended; T4 minimum).

PURPOSE
-------
Fine-tune a ClinicalBERT model (emily-alsentzer/Bio_ClinicalBERT or
emilyalsentzer/Bio_ClinicalBERT from HuggingFace) on MIMIC-III discharge
summaries to extract ADE mentions at the sentence/span level.

This module produces span-level ADE predictions that feed into the weak-
supervision label pipeline as a high-quality but expensive LF, or directly
as a feature (note_ade_score) for the downstream risk classifier.

INTENDED INPUTS
---------------
  note_df         : OMOP note table (discharge summaries from NOTEEVENTS)
  model_name      : HuggingFace model ID string
  checkpoint_dir  : path to save/load fine-tuned weights (Google Drive on Colab)

INTENDED OUTPUT
---------------
  pd.DataFrame with:
    note_id, visit_occurrence_id, person_id,
    ade_span_text   (extracted drug–ADE mention),
    ade_score       (model confidence 0–1),
    drug_mention    (drug string from span),
    event_mention   (ADE event string from span)

COLAB SETUP
-----------
  !pip install transformers>=4.40 torch>=2.0

TODO
----
  1. Tokenise note_text with ClinicalBERT tokenizer (handle 512-token limit
     via sliding window or truncation).
  2. Fine-tune as NER or sequence classification on an ADE-annotated corpus
     (e.g. ADE-corpus-V2 from HuggingFace datasets, or n2c2 2018 shared task).
  3. Run inference on discharge summaries; aggregate scores to visit level.
  4. Expose predict() returning the output schema above.
"""

from __future__ import annotations


def load_model(
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT", checkpoint_dir: str | None = None
):
    """Load ClinicalBERT (from HuggingFace or fine-tuned checkpoint).

    TODO: implement with transformers.AutoModelForTokenClassification.
    Requires: pip install transformers torch (Colab only).
    """
    raise NotImplementedError("TODO: implement load_model — Colab only")


def predict(note_df, model, tokenizer, batch_size: int = 16):
    """Run ADE span extraction on note_df['note_text'].

    TODO: implement batched inference with sliding-window tokenization.
    """
    raise NotImplementedError("TODO: implement predict — Colab only")


def fine_tune(note_df, labels_df, model_name: str, output_dir: str, epochs: int = 3):
    """Fine-tune ClinicalBERT on ADE-annotated examples.

    TODO: implement using HuggingFace Trainer API.
    labels_df should contain token-level BIO annotations.
    """
    raise NotImplementedError("TODO: implement fine_tune — Colab GPU only")
