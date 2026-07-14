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


def compute_bleu(references, hypotheses):
    """
    Corpus-level BLEU (sacrebleu -- the reproducible, citable implementation;
    NLTK's sentence-level BLEU varies with smoothing-function choice and
    isn't directly comparable across papers/runs).

    Normalized the same way as WER/CER above: BLEU's n-gram matching is
    edit-distance-adjacent, so punctuation noise hurts it the same way it
    hurts WER, unlike BERTScore below.

    Returns: 0-100 scale (sacrebleu convention), 0.0 if refs/hyps are empty.
    """
    if not references or not hypotheses:
        return 0.0
    import sacrebleu
    refs_norm = [normalize_kazakh(r) for r in references]
    hyps_norm = [normalize_kazakh(h) for h in hypotheses]
    return sacrebleu.corpus_bleu(hyps_norm, [refs_norm]).score


class _WhitespaceTokenizer:
    """
    rouge_score's own DefaultTokenizer strips anything not matching
    [a-z0-9] -- confirmed empirically to reduce Kazakh (Cyrillic) sentences
    to just their embedded digits, e.g. "жыл сайын жүрекке 500-ге тарта"
    tokenizes to ['500'], silently discarding every Cyrillic word. This
    plain split() tokenizer (same approach already used for WER above)
    keeps the actual words.
    """
    def tokenize(self, text):
        return text.split()


_ROUGE_TOKENIZER = _WhitespaceTokenizer()


def compute_rouge(references, hypotheses):
    """
    Average ROUGE-1/2/L F-measure (rouge_score, Google's reference
    implementation), using a whitespace tokenizer instead of the package
    default (see _WhitespaceTokenizer above -- the default silently mangles
    Cyrillic text). use_stemmer=False -- rouge_score's stemmer is the
    English Porter stemmer, which does nothing useful (and could silently
    mis-stem) on Kazakh tokens, so stemming is disabled rather than left on
    by default and quietly wrong.

    Returns: (rouge1, rouge2, rougeL) each in [0, 1], zeros if empty input.
    """
    if not references or not hypotheses:
        return 0.0, 0.0, 0.0
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                      use_stemmer=False,
                                      tokenizer=_ROUGE_TOKENIZER)
    r1, r2, rl = [], [], []
    for r, h in zip(references, hypotheses):
        scores = scorer.score(normalize_kazakh(r), normalize_kazakh(h))
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rl.append(scores['rougeL'].fmeasure)
    n = len(r1)
    return sum(r1) / n, sum(r2) / n, sum(rl) / n


def compute_bertscore(references, hypotheses, device='cpu',
                      model_type='bert-base-multilingual-cased'):
    """
    BERTScore F1 via a MULTILINGUAL encoder. The package's own English
    default (roberta-large) has essentially no Kazakh subword coverage and
    would silently produce meaningless scores -- multilingual BERT covers
    100+ languages including Kazakh. bert-base-multilingual-cased (not the
    heavier xlm-roberta-large) is the default here since this runs every
    validation epoch alongside WER's beam-search generation; pass a
    stronger model_type explicitly if quality matters more than epoch time.

    Deliberately NOT text-normalized, unlike WER/BLEU/ROUGE above --
    BERTScore's contextual embeddings benefit from natural casing and
    punctuation; stripping punctuation the way normalize_kazakh() does for
    edit-distance-style metrics would actively remove signal it uses.

    Returns: mean F1 in [0, 1] (typically much higher baseline than
    BLEU/ROUGE since it's semantic-similarity-based, not exact n-gram
    overlap -- compare against a same-language random-pairs baseline, not
    against 0), 0.0 if refs/hyps are empty.
    """
    if not references or not hypotheses:
        return 0.0
    from bert_score import score as bert_score_fn
    _, _, f1 = bert_score_fn(hypotheses, references, model_type=model_type,
                             lang=None, device=device, verbose=False)
    return f1.mean().item()
