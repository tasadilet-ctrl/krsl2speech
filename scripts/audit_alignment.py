"""
Transcript-alignment audit.

Sanity-checks whether clip text plausibly matches clip duration, using
characters-per-second. Kazakh news speech runs at roughly 12-18 chars/sec;
clips far outside that band are likely mis-segmented, carry the wrong
transcript, or (for the legacy Informburo loader) carry the FULL VIDEO
transcript on a short segment.

Usage:
  # asan-dataset (annotations carry per-clip T)
  python scripts/audit_alignment.py asan \
      --root /data/shared/asan-dataset \
      --sources informburo khabar qazaqstantv --split train --fps 50

  # khabar_kz manifest (jsonl with duration/start/end + norm_text)
  python scripts/audit_alignment.py manifest \
      --manifest /raid/shared/alikhan_datasets/khabar_kz/khabar_kz.jsonl

Output: per-source stats + the worst outliers, optional CSV via --csv.
"""
import os
import sys
import json
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CPS_LOW = 5.0    # < this: text too short for the clip (dropped words?)
CPS_HIGH = 30.0  # > this: text too long for the clip (wrong/overfull transcript)


def _percentile(vals, p):
    if not vals:
        return float('nan')
    s = sorted(vals)
    k = min(int(len(s) * p / 100.0), len(s) - 1)
    return s[k]


def audit_records(records, label, csv_writer=None, show_worst=5):
    """
    records: iterable of dicts {clip_id, duration_sec, text}
    Prints stats and returns the list of flagged records.
    """
    rows = []
    for r in records:
        dur = r['duration_sec']
        text = (r.get('text') or '').strip()
        if dur <= 0:
            continue
        cps = len(text) / dur
        flag = ('TEXT_TOO_SHORT' if cps < CPS_LOW else
                'TEXT_TOO_LONG' if cps > CPS_HIGH else '')
        row = {'source': label, 'clip_id': r.get('clip_id', ''),
               'duration_sec': round(dur, 2), 'chars': len(text),
               'chars_per_sec': round(cps, 2), 'flag': flag}
        rows.append(row)
        if csv_writer:
            csv_writer.writerow(row)

    if not rows:
        print(f"[{label}] no records")
        return []

    cps_vals = [r['chars_per_sec'] for r in rows]
    flagged = [r for r in rows if r['flag']]
    n = len(rows)
    print(f"\n[{label}] {n} clips")
    print(f"  chars/sec: median={_percentile(cps_vals, 50):.1f}  "
          f"p5={_percentile(cps_vals, 5):.1f}  p95={_percentile(cps_vals, 95):.1f}")
    print(f"  flagged: {len(flagged)} ({100.0 * len(flagged) / n:.1f}%)  "
          f"[outside {CPS_LOW}-{CPS_HIGH} cps]")
    worst = sorted(flagged, key=lambda r: min(abs(r['chars_per_sec'] - CPS_LOW),
                                              abs(r['chars_per_sec'] - CPS_HIGH)),
                   reverse=True)[:show_worst]
    for r in worst:
        print(f"    {r['flag']:15s} {r['clip_id'][:60]:60s} "
              f"{r['duration_sec']:7.1f}s {r['chars']:6d} ch "
              f"{r['chars_per_sec']:6.1f} cps")
    return flagged


def iter_asan(root, sources, lang, split, fps):
    split_file = {'train': 'train.json', 'val': 'dev.json',
                  'dev': 'dev.json', 'test': 'test.json'}[split]
    for source in sources:
        path = os.path.join(root, source, 'annotations', lang, split_file)
        if not os.path.exists(path):
            print(f"[warn] missing {path}")
            continue
        with open(path) as f:
            entries = json.load(f)
        yield source, [{'clip_id': e.get('clip_id', ''),
                        'duration_sec': e.get('T', 0) / fps,
                        'text': e.get('text', '')} for e in entries]


def iter_manifest(manifest):
    records = []
    with open(manifest) as f:
        for line in f:
            e = json.loads(line)
            dur = e.get('duration',
                        (e.get('end', 0) or 0) - (e.get('start', 0) or 0))
            records.append({'clip_id': e.get('clip_id', ''),
                            'duration_sec': dur,
                            'text': e.get('norm_text', e.get('text', ''))})
    yield os.path.basename(manifest), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)

    p_asan = sub.add_parser('asan')
    p_asan.add_argument('--root', required=True)
    p_asan.add_argument('--sources', nargs='+',
                        default=['informburo', 'khabar', 'qazaqstantv'])
    p_asan.add_argument('--lang', default='kz')
    p_asan.add_argument('--split', default='train',
                        choices=['train', 'val', 'dev', 'test'])
    p_asan.add_argument('--fps', type=float, default=50.0)

    p_man = sub.add_parser('manifest')
    p_man.add_argument('--manifest', required=True)

    for p in (p_asan, p_man):
        p.add_argument('--csv', default=None, help='write per-clip rows to CSV')

    args = parser.parse_args()

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, 'w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            'source', 'clip_id', 'duration_sec', 'chars',
            'chars_per_sec', 'flag'])
        csv_writer.writeheader()

    if args.mode == 'asan':
        groups = iter_asan(args.root, args.sources, args.lang,
                           args.split, args.fps)
    else:
        groups = iter_manifest(args.manifest)

    total_flagged = 0
    for label, records in groups:
        total_flagged += len(audit_records(records, label, csv_writer))

    if csv_file:
        csv_file.close()
        print(f"\nPer-clip rows written to {args.csv}")

    print(f"\nTotal flagged: {total_flagged}")
    print("Interpretation: TEXT_TOO_LONG at high rates usually means clips "
          "carry a transcript that spans more video than the clip (the "
          "legacy Informburo loader does this by construction); "
          "TEXT_TOO_SHORT suggests dropped/partial transcripts.")


if __name__ == '__main__':
    main()
