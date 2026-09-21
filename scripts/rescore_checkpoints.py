#!/usr/bin/env python3
"""
E1: re-score existing checkpoints on the canonical selection set, and sweep
decoding settings -- no training.

Every number this project recorded before E0 came from the trainer's first 25
unshuffled dev batches (200 informburo clips) with a raw-WER / normalized-BLEU
mix. This re-scores checkpoints on the frozen, source-stratified selection
manifest through the ONE shared metric implementation, and saves every
prediction so any number can be re-derived later.

Contamination is reported, not hidden. Each checkpoint is declared with the
annotation roots whose TRAIN split it saw; clips from videos in those splits
are flagged and metrics are reported twice:
  * all      -- the full selection set
  * clean    -- only clips from videos that checkpoint never trained on
Compare checkpoints on `clean`, and on the common clean subset in summary.json.

Every declared root must exist and carry a train split for every source, or
the script exits before loading anything. A checkpoint that trained on no
KRSL data declares the root `none`, which is the only way to report zero
contamination.

Usage (one or more --ckpt, each  path=label:root[,root...]):
  PYTHONPATH=. python scripts/rescore_checkpoints.py \
    --root ~/asan_canonical --use-enriched --out output/e1 \
    --ckpt output/ours_enriched_friend_mt5.pth=init:/data/archive/asan-dataset \
    --ckpt output/clean_treat/phase1_mt5_epoch10.pth=treat_e10:/data/archive/asan-dataset,~/asan_clean \
    --sweep-on auto
"""
import argparse
import collections
import contextlib
import io
import itertools
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from data.asan_dataset import AsanDataset
from data.utils import ENRICHED_DIM, KEYPOINT_DIM, UNISIGN_DIM
from models.unisign_encoder import KeypointEncoder
from train.train_encoder_mt5 import UniSignMT5, SimpleCollator
from utils.metrics import compute_bleu, compute_corpus_wer, normalize_kazakh

SOURCES = ('informburo', 'khabar', 'qazaqstantv')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def video_id(entry):
    if entry.get('video_id'):
        return str(entry['video_id'])
    m = re.match(r'[^_]+__(.+?)__seg', entry.get('clip_id', ''))
    return m.group(1) if m else None


# Declares a checkpoint that trained on no KRSL data at all (e.g. a base
# Uni-Sign + base mT5 init). The only way to get an empty contamination set.
NO_TRAIN_ROOTS = 'none'


def train_split_paths(root):
    """{source: train.json path} for a declared training root, or exit.

    Every declared root must exist and carry a train split for every source.
    These used to be skipped with `if os.path.exists(p)`, so a mistyped root
    produced an empty set and every clip was reported as uncontaminated: the
    check could not detect its own misconfiguration. A root missing a single
    source was worse, because it under-reported only that source and still
    looked plausible.
    """
    path = os.path.expanduser(root)
    if not os.path.isdir(path):
        sys.exit(f"[fatal] training root {root!r} does not exist ({path}). "
                 f"A checkpoint trained on no KRSL data declares "
                 f"{NO_TRAIN_ROOTS!r} instead.")
    splits = {src: os.path.join(path, src, 'annotations', 'kz', 'train.json')
              for src in SOURCES}
    missing = [src for src, p in splits.items() if not os.path.exists(p)]
    if missing:
        sys.exit(f"[fatal] training root {root!r} has no train split for "
                 f"{missing}; contamination for those sources would be "
                 f"silently under-reported")
    return splits


def validate_roots(roots):
    """Fail fast on a bad root list, before any checkpoint is loaded."""
    if NO_TRAIN_ROOTS in roots and roots != [NO_TRAIN_ROOTS]:
        sys.exit(f"[fatal] {NO_TRAIN_ROOTS!r} cannot be combined with other "
                 f"training roots: {roots}")
    if roots != [NO_TRAIN_ROOTS]:
        for root in roots:
            train_split_paths(root)


def train_videos(roots):
    """(source, video_id) pairs a checkpoint trained on, from its roots' TRAIN splits."""
    validate_roots(roots)
    if roots == [NO_TRAIN_ROOTS]:
        return set()
    seen = set()
    for root in roots:
        from_root = set()
        for src, p in train_split_paths(root).items():
            from_root |= {(src, video_id(e)) for e in json.load(open(p))}
        if not from_root:
            sys.exit(f"[fatal] training root {root!r} has train splits but "
                     f"they list no videos")
        seen |= from_root
    return seen


def content_word_recall(refs, hyps, min_len=4):
    """Fraction of reference content words (len >= min_len) present in the
    hypothesis. Separates 'fluent but wrong story' from partial translation."""
    got = total = 0
    for r, h in zip(refs, hyps):
        hw = set(normalize_kazakh(h).split())
        rw = [w for w in normalize_kazakh(r).split() if len(w) >= min_len]
        total += len(rw)
        got += sum(w in hw for w in rw)
    return got / max(total, 1)


def score(rows):
    if not rows:
        return {'n': 0}
    refs = [r['ref'] for r in rows]
    hyps = [r['hyp'] for r in rows]
    out = {
        'n': len(rows),
        'wer': compute_corpus_wer(refs, hyps),
        'wer_raw': compute_corpus_wer(refs, hyps, normalize=False),
        'bleu': compute_bleu(refs, hyps),
        'content_recall': content_word_recall(refs, hyps),
        'hyp_len_ratio': (sum(len(h.split()) for h in hyps)
                          / max(sum(len(r.split()) for r in refs), 1)),
        'empty_hyps': sum(1 for h in hyps if not h.strip()),
    }
    try:
        import sacrebleu
        out['bleu_raw'] = sacrebleu.corpus_bleu(hyps, [refs]).score
        out['chrf'] = sacrebleu.corpus_chrf(hyps, [refs]).score
    except ImportError:
        pass
    return out


def breakdown(rows):
    res = {'all': score(rows)}
    for src in SOURCES:
        res[src] = score([r for r in rows if r['source'] == src])
    return res


def build_loader(root, manifest_path, use_enriched, signspace, downsample,
                 tokenizer, batch_size, real_wrists=False, score_quantiles=None,
                 unisign_preprocess=False, unisign_hand_scale=None):
    with open(manifest_path) as fh:
        manifest = {m['clip_id']: m for m in json.load(fh)}
    parts, meta = [], []
    for src in SOURCES:
        with contextlib.redirect_stdout(io.StringIO()):
            ds = AsanDataset(root=root, sources=[src], split='val', lang='kz',
                             use_enriched=use_enriched, signspace=signspace,
                             real_wrists=real_wrists, score_quantiles=score_quantiles,
                             unisign_preprocess=unisign_preprocess,
                             unisign_hand_scale=unisign_hand_scale,
                             downsample_every=downsample)
        parts.append(ds)
    indices, offset = [], 0
    for ds in parts:
        for i, c in enumerate(ds.clips):
            if c.get('clip_id') in manifest:
                indices.append(offset + i)
                meta.append({'clip_id': c['clip_id'], 'source': c['_source'],
                             'video_id': video_id(c)})
        offset += len(ds)
    if len(indices) != len(manifest):
        sys.exit(f"[fatal] manifest has {len(manifest)} clips, only "
                 f"{len(indices)} resolve under {root} -- refusing to score a "
                 f"different subset than the one frozen")
    concat = torch.utils.data.ConcatDataset(parts)
    loader = DataLoader(Subset(concat, indices), batch_size=batch_size,
                        shuffle=False, num_workers=4,
                        collate_fn=SimpleCollator(mt5_tokenizer=tokenizer,
                                                  max_text_tokens=128))
    return loader, meta


def load_weights(model, path, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.encoder.load_state_dict(ckpt['encoder'])
    if 'pose_norm' in ckpt:
        model.pose_norm.load_state_dict(ckpt['pose_norm'])
    else:
        log("  [warn] checkpoint has no pose_norm -- using default LayerNorm init")
        model.pose_norm.reset_parameters()
    if 'mt5' in ckpt:
        model.mt5.load_state_dict(ckpt['mt5'])
    elif 'mt5_lora' in ckpt:
        sys.exit("[fatal] LoRA checkpoints need their LoRA config from "
                 "checkpoint metadata; not supported here yet")
    else:
        sys.exit(f"[fatal] {path} has no mT5 weights")
    info = {k: ckpt.get(k) for k in ('epoch', 'source', 'use_lora')}
    info['run_args'] = ckpt.get('run_args')
    del ckpt
    return info


# Input-pipeline flags that change what the encoder sees. A checkpoint scored
# with a different setting than it trained with gets silently wrong inputs.
_INPUT_FLAGS = ('use_enriched', 'signspace', 'real_wrists', 'score_quantiles',
                'unisign_preprocess', 'unisign_hand_scale', 'block_padding_mask')


def check_run_args(info, args, path):
    ra = info.get('run_args') or {}
    bad = []
    for k in _INPUT_FLAGS:
        if k not in ra:
            continue                      # older checkpoints predate the flag
        trained, scoring = ra[k], getattr(args, k)
        if k in ('score_quantiles', 'unisign_hand_scale'):
            trained, scoring = bool(trained), bool(scoring)
        if bool(trained) != bool(scoring):
            bad.append(f"{k}: trained={ra[k]!r} scoring={getattr(args, k)!r}")
    if bad:
        sys.exit(f"[fatal] {path} input pipeline mismatch -- " + '; '.join(bad))


def run(model, loader, meta, device, decode):
    # Key on clip_id, never on position: SimpleCollator sorts each batch by
    # length and drops invalid samples, so positional matching silently
    # attaches the wrong source/video/contamination label to a prediction.
    by_id = {m['clip_id']: m for m in meta}
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            hyps = model.generate(batch['keypoints'].to(device),
                                  input_lengths=batch['input_lengths'].to(device),
                                  **decode)
            ids = batch['clip_ids']
            assert len(ids) == len(hyps) == len(batch['texts'])
            for cid, ref, hyp in zip(ids, batch['texts'], hyps):
                rows.append({**by_id[cid], 'ref': ref, 'hyp': hyp})
    rows.sort(key=lambda r: r['clip_id'])
    if len({r['clip_id'] for r in rows}) != len(rows):
        sys.exit('[fatal] duplicate clip_id in predictions')
    if len(rows) != len(meta):
        log(f"  [warn] generated {len(rows)}/{len(meta)} -- coverage incomplete")
    return rows


def evaluate(tag, model, loader, meta, device, decode, seen, out_dir):
    t0 = time.time()
    rows = run(model, loader, meta, device, decode)
    for r in rows:
        r['contaminated'] = (r['source'], r['video_id']) in seen
    with open(os.path.join(out_dir, f'{tag}.predictions.jsonl'), 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    res = {'tag': tag, 'decode': decode, 'coverage': f'{len(rows)}/{len(meta)}',
           'n_contaminated': sum(r['contaminated'] for r in rows),
           'all': breakdown(rows),
           'clean': breakdown([r for r in rows if not r['contaminated']]),
           'seconds': round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, f'{tag}.metrics.json'), 'w') as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    a, c = res['all']['all'], res['clean']['all']
    log(f"  {tag}: clean n={c['n']} WER={c.get('wer', 0):.4f} "
        f"BLEU={c.get('bleu', 0):.2f} (raw {c.get('bleu_raw', 0):.2f}) "
        f"chrF={c.get('chrf', 0):.2f} recall={c.get('content_recall', 0):.3f} "
        f"| contaminated={res['n_contaminated']} | {res['seconds']}s")
    return res, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--root', default='~/asan_canonical')
    ap.add_argument('--manifest', default=None,
                    help='defaults to <root>/selection_manifest.json')
    ap.add_argument('--ckpt', action='append', required=True,
                    help='path=label:trainroot[,trainroot...]')
    ap.add_argument('--use-enriched', action='store_true')
    ap.add_argument('--signspace', action='store_true')
    ap.add_argument('--block-padding-mask', action='store_true')
    ap.add_argument('--real-wrists', action='store_true')
    ap.add_argument('--score-quantiles', default=None)
    ap.add_argument('--unisign-preprocess', action='store_true')
    ap.add_argument('--unisign-hand-scale', default=None)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--out', default='output/e1')
    ap.add_argument('--sweep-on', default=None,
                    help="checkpoint label to sweep decoding on, or 'auto'")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    manifest = os.path.expanduser(args.manifest or
                                  os.path.join(root, 'selection_manifest.json'))
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    downsample = cfg['paths']['asan'].get('downsample_every', 1)

    specs = []
    for spec in args.ckpt:
        path, rest = spec.split('=', 1)
        label, roots = rest.split(':', 1)
        roots = roots.split(',')
        # Checked here, not per checkpoint: a typo in the last --ckpt would
        # otherwise surface only after every earlier checkpoint was scored.
        validate_roots(roots)
        specs.append((os.path.expanduser(path), label, roots))
    if args.sweep_on not in (None, 'auto') and args.sweep_on not in {s[1] for s in specs}:
        sys.exit(f"[fatal] --sweep-on {args.sweep_on!r} is not a checkpoint label")

    input_dim = (UNISIGN_DIM if args.unisign_preprocess
                 else ENRICHED_DIM() if args.use_enriched else KEYPOINT_DIM)
    with contextlib.redirect_stdout(io.StringIO()):
        encoder = KeypointEncoder(hidden_dim=cfg['model']['d_model'], input_dim=input_dim,
                                  block_padding_mask=args.block_padding_mask,
                                  real_wrists=args.real_wrists,
                                  unisign_input=args.unisign_preprocess)
        model = UniSignMT5(encoder=encoder, lang='Kazakh').to(device)
    loader, meta = build_loader(root, manifest, args.use_enriched, args.signspace,
                                downsample, model.mt5_tokenizer, args.batch_size,
                                real_wrists=args.real_wrists,
                                score_quantiles=args.score_quantiles,
                                unisign_preprocess=args.unisign_preprocess,
                                unisign_hand_scale=args.unisign_hand_scale)
    log(f"selection: {len(meta)} clips from {manifest}  "
        f"{dict(collections.Counter(m['source'] for m in meta))}")

    default_decode = {'num_beams': 4, 'repetition_penalty': 1.3,
                      'no_repeat_ngram_size': 3}
    summary = {'root': root, 'manifest': manifest, 'n_selection': len(meta),
               'checkpoints': {}, 'sweep': []}
    contaminated = {}

    for path, label, roots in specs:
        log(f"=== {label}: {path}")
        info = load_weights(model, path, device)
        check_run_args(info, args, path)
        seen = train_videos(roots)
        res, rows = evaluate(label, model, loader, meta, device, default_decode,
                             seen, args.out)
        res['checkpoint'] = {'path': path, 'train_roots': roots, **info}
        summary['checkpoints'][label] = res
        contaminated[label] = {r['clip_id'] for r in rows if r['contaminated']}
        with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=1, ensure_ascii=False)

    # Common clean subset: clips no scored checkpoint trained on, so the
    # checkpoints can be compared against each other on identical data.
    bad = set().union(*contaminated.values()) if contaminated else set()
    common = {}
    for label in summary['checkpoints']:
        rows = [json.loads(l) for l in open(os.path.join(args.out, f'{label}.predictions.jsonl'))]
        common[label] = breakdown([r for r in rows if r['clip_id'] not in bad])
    summary['common_clean'] = {'n': len(meta) - len(bad), 'by_checkpoint': common}
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    log(f"=== common clean subset: {len(meta) - len(bad)}/{len(meta)} clips")
    log(f"{'checkpoint':14s} {'WER':>7s} {'BLEU':>6s} {'rawBLEU':>7s} {'chrF':>6s} {'recall':>7s}")
    for label, res in common.items():
        a = res['all']
        log(f"{label:14s} {a.get('wer', 0):7.4f} {a.get('bleu', 0):6.2f} "
            f"{a.get('bleu_raw', 0):7.2f} {a.get('chrf', 0):6.2f} "
            f"{a.get('content_recall', 0):7.3f}")

    # Decoding sweep. 'auto' picks the checkpoint with the best chrF on the
    # common clean subset -- chrF, not BLEU, because character n-grams are far
    # less brittle for agglutinative Kazakh on a 600-clip set. The rule is
    # fixed here, before any sweep result exists.
    target = args.sweep_on
    if target == 'auto':
        target = max(common, key=lambda k: common[k]['all'].get('chrf', -1))
        log(f"=== sweep target (auto, best common-clean chrF): {target}")
    if target:
        path, _, roots = next(sp for sp in specs if sp[1] == target)
        load_weights(model, path, device)
        seen = train_videos(roots)
        log(f"=== decoding sweep on {target} (dev selection only, never test)")
        for beams, rep, ngram in itertools.product((1, 4), (1.0, 1.1, 1.3), (0, 3)):
            decode = {'num_beams': beams, 'repetition_penalty': rep,
                      'no_repeat_ngram_size': ngram}
            tag = f'{target}.sweep_b{beams}_rp{rep}_ng{ngram}'
            sres, _ = evaluate(tag, model, loader, meta, device, decode, seen, args.out)
            summary['sweep'].append(sres)
            with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
                json.dump(summary, fh, indent=1, ensure_ascii=False)
        log(f"{'sweep config':34s} {'WER':>7s} {'BLEU':>6s} {'chrF':>6s} {'recall':>7s} {'len':>5s}")
        for r in summary['sweep']:
            c = r['clean']['all']
            log(f"{r['tag'].split('.', 1)[1]:34s} {c.get('wer', 0):7.4f} "
                f"{c.get('bleu', 0):6.2f} {c.get('chrf', 0):6.2f} "
                f"{c.get('content_recall', 0):7.3f} {c.get('hyp_len_ratio', 0):5.2f}")
    log("E1 DONE")


if __name__ == '__main__':
    main()
