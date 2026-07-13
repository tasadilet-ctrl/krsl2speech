"""
Evaluation metrics for KRSL → Kazakh Speech.
"""
import re
from jiwer import wer, cer


def normalize_kazakh(text):
    """Normalize Kazakh text for WER/CER computation."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)      # collapse whitespace
    return text


def compute_wer(reference, hypothesis):
    """Compute Word Error Rate for Kazakh text."""
    ref_norm = normalize_kazakh(reference)
    hyp_norm = normalize_kazakh(hypothesis)
    return wer(ref_norm, hyp_norm)


def compute_cer(reference, hypothesis):
    """Compute Character Error Rate for Kazakh text."""
    ref_norm = normalize_kazakh(reference)
    hyp_norm = normalize_kazakh(hypothesis)
    return cer(ref_norm, hyp_norm)


def compute_batch_wer_cer(references, hypotheses):
    """Compute WER and CER over a batch."""
    wers = [compute_wer(r, h) for r, h in zip(references, hypotheses)]
    cers = [compute_cer(r, h) for r, h in zip(references, hypotheses)]
    return sum(wers) / len(wers), sum(cers) / len(cers)
