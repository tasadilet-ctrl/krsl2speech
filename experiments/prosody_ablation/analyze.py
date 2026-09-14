#!/usr/bin/env python3
"""Re-derives the prosody ablation's conclusion from its training logs.

The 28 GB of checkpoints these runs produced were deleted; the logs are the
evidence and they are enough. Run with no arguments:

    python3 experiments/prosody_ablation/analyze.py

Three things are shown, matching the three reasons the direction was closed:
the run-to-run noise floor, the winner flipping between epochs, and the two
headline metrics disagreeing inside a single epoch.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = re.compile(
    r"Epoch (?P<ep>\d+)/\d+ \| Train: (?P<train>[\d.]+) \| Val: (?P<ce>[\d.]+) \| "
    r"WER: (?P<wer>[\d.]+) \| BLEU: (?P<bleu>[\d.]+)")


def parse(path):
    """Epoch -> list of runs. An epoch appears twice where a run was resumed."""
    out = {}
    for m in LINE.finditer(path.read_text()):
        out.setdefault(int(m["ep"]), []).append(
            {k: float(m[k]) for k in ("train", "ce", "wer", "bleu")})
    return out


def main():
    base, treat = parse(HERE / "abl_baseline.log"), parse(HERE / "abl_prosody.log")

    print("Prosody-as-encoder-supervision ablation")
    print("baseline = --prosody-aux-weight 0.0, treatment = 0.5, same seed,")
    print("same data, resumed from one identical checkpoint.\n")

    print("1. Noise floor, from epochs that were accidentally run twice")
    print("   (a resume re-ran them; nothing else differed):")
    found = False
    for name, runs in (("baseline", base), ("treatment", treat)):
        for ep, rs in sorted(runs.items()):
            if len(rs) > 1:
                found = True
                dw = max(r["wer"] for r in rs) - min(r["wer"] for r in rs)
                db = max(r["bleu"] for r in rs) - min(r["bleu"] for r in rs)
                print(f"   {name:9s} epoch {ep}: "
                      f"WER {' vs '.join(f'{r[chr(119)+chr(101)+chr(114)]:.4f}' for r in rs)}"
                      f"  ->  dWER {dw:.4f}   dBLEU {db:.2f}")
    if not found:
        print("   (no duplicated epochs found)")
    print("   Any single-seed gap smaller than this is indistinguishable"
          "\n   from nondeterminism.\n")

    print("2. Which arm is ahead, epoch by epoch (last run of each epoch):")
    print(f"   {'ep':>3}  {'base WER':>9}{'treat WER':>10}  {'WER says':<10}"
          f"{'base BLEU':>10}{'treat BLEU':>11}  {'BLEU says':<10}")
    flips, disagree = 0, 0
    prev = None
    for ep in sorted(set(base) & set(treat)):
        b, t = base[ep][-1], treat[ep][-1]
        wer_win = "baseline" if b["wer"] < t["wer"] else "treatment"
        bleu_win = "baseline" if b["bleu"] > t["bleu"] else "treatment"
        if prev and wer_win != prev:
            flips += 1
        prev = wer_win
        if wer_win != bleu_win:
            disagree += 1
        print(f"   {ep:>3}  {b['wer']:>9.4f}{t['wer']:>10.4f}  {wer_win:<10}"
              f"{b['bleu']:>10.2f}{t['bleu']:>11.2f}  {bleu_win:<10}")
    print(f"\n   WER's winner changed hands {flips} time(s) across "
          f"{len(set(base) & set(treat))} epochs.")
    print(f"   WER and BLEU disagreed with each other in {disagree} epoch(s).\n")

    print("3. Validation CE against generation quality (baseline arm):")
    print(f"   {'ep':>3}{'val CE':>9}{'WER':>9}{'BLEU':>7}")
    eps = sorted(base)
    for ep in eps:
        r = base[ep][-1]
        print(f"   {ep:>3}{r['ce']:>9.4f}{r['wer']:>9.4f}{r['bleu']:>7.2f}")
    first, last = base[eps[0]][-1], base[eps[-1]][-1]
    print(f"\n   CE {first['ce']:.4f} -> {last['ce']:.4f} "
          f"({'worse' if last['ce'] > first['ce'] else 'better'}), while "
          f"WER {first['wer']:.4f} -> {last['wer']:.4f} "
          f"({'better' if last['wer'] < first['wer'] else 'worse'}).")
    print("   Selecting the best checkpoint on val CE picks an early, worse")
    print("   model. This is why --select-metric defaults to wer (899cf39).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
