#!/usr/bin/env python3
"""
Page of the corpus audit's flagged clips: each clip (with its keypoints drawn)
next to the transcript it carries AND the sentence from the same broadcast that
fits it better, so a person can judge whether the reference is wrong.

    python scripts/build_audit_viewer.py --per-source 20
    cd ~/audit_viewer && python3 -m http.server 8790   # then open in a browser
"""
import argparse
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_keypoint_overlays as rko   # noqa: E402

ROOT = os.path.expanduser('~/asan_canonical')


def video_of(src, split, clip_id):
    for e in json.load(open(os.path.join(ROOT, src, 'annotations', 'kz', f'{split}.json'))):
        if e['clip_id'] == clip_id:
            v = e.get('video')
            return os.path.join(ROOT, v) if v else \
                e['pose'].replace('/keypoints/', '/videos/').replace('.npz', '.mp4')
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flagged', default='output/audit_report/flagged_worst.json')
    ap.add_argument('--per-source', type=int, default=20)
    ap.add_argument('--out', default='~/audit_viewer')
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    for d in ('clips', 'overlays'):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    flagged = json.load(open(a.flagged))            # worst first
    picked, per = [], {}
    for r in flagged:
        if per.get(r['source'], 0) < a.per_source:
            picked.append(r); per[r['source']] = per.get(r['source'], 0) + 1

    cards = []
    for k, r in enumerate(picked, 1):
        vid = video_of(r['source'], r['split'], r['clip_id'])
        if not vid or not os.path.exists(vid):
            continue
        name = r['clip_id'] + '.mp4'
        link = os.path.join(out, 'clips', name)
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(vid), link)
        ov = os.path.join(out, 'overlays', name)
        kp = rko.keypoint_path(os.path.realpath(vid))
        if not os.path.exists(ov) and kp:
            rko.render(os.path.realpath(vid), kp, ov)
        src_v = f'overlays/{name}' if os.path.exists(ov) else f'clips/{name}'
        adj = 'the ADJACENT sentence' if r['best_other_is_adjacent'] else 'a non-adjacent sentence'
        cards.append(
            f'<div class="card" data-s="{r["source"]}">'
            f'<video src="{src_v}" controls loop muted preload="metadata"></video>'
            f'<div class="meta"><b>{html.escape(r["clip_id"])}</b> &middot; {r["source"]} {r["split"]} '
            f'&middot; {r["wps"]:.1f} words/s &middot; own sentence ranks {r["own_rank"]} of {r["pool"]}'
            + (f' &middot; text: {r["text_source"]}' if r['text_source'] else '') +
            f'</div><div class="lab">carries this transcript</div><p>{html.escape(r["text"])}</p>'
            f'<div class="lab">fits better: {adj} &middot; margin {r["row_margin"]:.1f} / '
            f'{r["col_margin"]:.1f}</div><p class="alt">{html.escape(r["best_other_text"])}</p></div>')
        if k % 10 == 0:
            print(f'  {k}/{len(picked)}', flush=True)

    page = f"""<!doctype html><html lang="kk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Corpus audit</title>
<style>
:root {{ --bg:#f6f6f3; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#dcdcd6; --alt:#2b5d8a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --card:#1f1f1d; --ink:#ececea;
  --muted:#9a9a94; --line:#34342f; --alt:#8cb8e0; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui,sans-serif; }}
header {{ position:sticky; top:0; z-index:1; background:var(--card); border-bottom:1px solid var(--line);
  padding:10px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
main {{ padding:16px; display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(560px,1fr)); }}
@media (max-width:600px) {{ main {{ grid-template-columns:1fr; }} }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }}
video {{ width:100%; max-width:640px; background:#000; border-radius:6px; }}
.meta {{ color:var(--muted); font-size:13px; margin-top:6px; }}
.lab {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin-top:10px; }}
p {{ margin:4px 0 0; }} .alt {{ color:var(--alt); }}
details {{ flex-basis:100%; font-size:14px; max-width:780px; }} summary {{ cursor:pointer; color:var(--muted); }}
</style></head><body>
<header><b>Corpus audit: worst-fitting references ({len(cards)} clips)</b>
<select id="s"><option value="">all sources</option><option>informburo</option><option>khabar</option>
<option>qazaqstantv</option></select>
<details><summary>How to read this</summary>
<p>Each clip was scored against every sentence of its own broadcast by a model that never trained on it.
These are the clips where <b>another sentence fits better than the clip's own, from both directions</b>
(the clip prefers another sentence, and its sentence prefers another clip), sorted worst first.</p>
<p>Watch the signing, then compare the two texts. If the blue sentence matches what is signed, the
reference is misaligned. "Adjacent" usually means a boundary shift between neighbouring clips. The model
also makes ordinary errors (it misses on ~17% of clean clips), so some flags will be false alarms --
that is what this page is for.</p></details></header>
<main>{''.join(cards)}</main>
<script>
const s=document.getElementById('s'), cs=[...document.querySelectorAll('.card')];
s.onchange=()=>cs.forEach(c=>c.style.display=(!s.value||c.dataset.s===s.value)?'':'none');
</script></body></html>"""
    open(os.path.join(out, 'index.html'), 'w', encoding='utf-8').write(page)
    print(f'[write] {out}/index.html: {len(cards)} flagged clips')


if __name__ == '__main__':
    main()
