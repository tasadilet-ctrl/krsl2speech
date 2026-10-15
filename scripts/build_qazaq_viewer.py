#!/usr/bin/env python3
"""
Read-only viewer for the clean qazaqstantv recollection: each clip's video next
to the transcript it trains on, with duration and words/second.

Videos are SYMLINKED from the recollection directory (nothing is copied);
point --recollect at it, or set QAZ_RECOLLECT.
Serve the folder and open it in a normal browser (Chrome/Safari play H.264;
VS Code's built-in player usually does not):

    python3 scripts/build_qazaq_viewer.py --split dev --n 80
    cd ~/qazaq_viewer && python3 -m http.server 8765
    # VS Code forwards the port -> open http://localhost:8765

Speech in news runs ~2-3 words/s; values far outside that are flagged, since
the original corruption showed up exactly as impossible words/second.
"""
import argparse
import html
import json
import os
import random

ROOT = os.path.expanduser('~/asan_canonical')
FPS = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='dev', choices=['train', 'dev', 'test'])
    ap.add_argument('--n', type=int, default=80)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='~/qazaq_viewer')
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.join(out, 'clips'), exist_ok=True)

    entries = [e for e in json.load(open(os.path.join(
        ROOT, 'qazaqstantv', 'annotations', 'kz', f'{a.split}.json'))) if e.get('text', '').strip()]
    random.Random(a.seed).shuffle(entries)
    rows = []
    for e in entries:
        if len(rows) >= a.n:
            break
        vid = e['pose'].replace('/keypoints/', '/videos/').replace('.npz', '.mp4')
        if not os.path.exists(vid):
            continue
        link = os.path.join(out, 'clips', os.path.basename(vid))
        if not os.path.lexists(link):
            os.symlink(vid, link)
        secs = e['T'] / FPS
        words = len(e['text'].split())
        rows.append({'id': e['clip_id'], 'file': 'clips/' + os.path.basename(vid),
                     'text': e['text'].strip(), 'secs': secs, 'wps': words / max(secs, 1e-6),
                     'words': words, 'src': vid})

    cards = []
    for r in rows:
        flag = ('too fast' if r['wps'] > 4 else 'too slow' if r['wps'] < 1 else '')
        ov = os.path.join(out, 'overlays', os.path.basename(r['file']))
        vsrc = 'overlays/' + os.path.basename(r['file']) if os.path.exists(ov) else r['file']
        cards.append(
            f'<div class="card" data-flag="{1 if flag else 0}" '
            f'data-k="{html.escape((r["id"] + " " + r["text"]).lower())}">'
            f'<video src="{vsrc}" controls loop muted preload="metadata"></video>'
            + (f'<div class="meta">keypoints exactly as the model receives them &middot; '
               f'<a href="{r["file"]}">plain video</a></div>'
               if vsrc != r['file'] else '')
            +
            f'<div class="meta"><b>{html.escape(r["id"])}</b> &middot; {r["secs"]:.1f} s &middot; '
            f'{r["words"]} words &middot; <span class="{"bad" if flag else ""}">{r["wps"]:.2f} words/s'
            f'{" &mdash; " + flag if flag else ""}</span></div>'
            f'<p>{html.escape(r["text"])}</p>'
            f'<div class="path">{html.escape(r["src"])}</div></div>')
    flagged = sum(1 for r in rows if r['wps'] > 4 or r['wps'] < 1)
    page = f"""<!doctype html><html lang="kk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>qazaqstantv clips</title>
<style>
:root {{ --bg:#f6f6f3; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#dcdcd6; --bad:#b23b3b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --card:#1f1f1d; --ink:#ececea;
  --muted:#9a9a94; --line:#34342f; --bad:#f09a92; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui,sans-serif; }}
header {{ position:sticky; top:0; background:var(--card); border-bottom:1px solid var(--line);
  padding:10px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
input {{ flex:1; min-width:200px; font:inherit; padding:6px 10px; border:1px solid var(--line);
  border-radius:6px; background:var(--bg); color:var(--ink); }}
main {{ padding:16px; display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(680px,1fr)); }}
@media (max-width:720px) {{ main {{ grid-template-columns:1fr; }} }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }}
video {{ width:100%; background:#000; border-radius:6px; }}
a {{ color:inherit; }}
.meta {{ color:var(--muted); font-size:13px; margin-top:6px; }}
.bad {{ color:var(--bad); font-weight:600; }}
.path {{ color:var(--muted); font-size:11px; word-break:break-all; margin-top:6px; }}
details {{ flex-basis:100%; font-size:14px; }} summary {{ cursor:pointer; color:var(--muted); }}
.help {{ max-width:760px; }} .help p {{ margin:6px 0; }}
p {{ margin:8px 0 0; }}
</style></head><body>
<header><b>qazaqstantv &middot; {a.split} &middot; {len(rows)} clips</b>
<input id="q" placeholder="filter by clip id or word">
<label><input type="checkbox" id="fl"> only unusual words/s ({flagged})</label><span id="n"></span>
<details><summary>How to check a flagged clip</summary><div class="help">
<p>News speech runs about <b>2-3 words per second</b>. Each clip's words/s is its transcript's word
count divided by the clip's length, so a value far outside 1-4 means the text and the video do not
cover the same stretch of the broadcast.</p>
<p><b>Too fast</b> (e.g. 30 words in a 5-second clip): the transcript holds more speech than the clip
contains. Watch whether the interpreter is already mid-sign when the clip starts and still signing when
it ends, or whether the clip is very short. Likely cause: the clip was cut shorter than its text.</p>
<p><b>Too slow</b> (e.g. 4 words in 10 seconds): the transcript is a fragment. Watch whether the
interpreter signs steadily for the whole clip -- far more than a few words' worth. Likely cause: the
text was cut, or is the ~10% fallback to speech-recognition text.</p>
<p>No audio is included, so this checks structure (how much signing vs. how much text). Checking
whether the signs match the words needs someone who reads KRSL.</p></div></details></header>
<main>{''.join(cards)}</main>
<script>
const q=document.getElementById('q'), fl=document.getElementById('fl'), n=document.getElementById('n'),
      cs=[...document.querySelectorAll('.card')];
function f(){{ const t=q.value.toLowerCase(); let k=0;
  cs.forEach(c=>{{ const s=c.dataset.k.includes(t) && (!fl.checked || c.dataset.flag==='1');
                  c.style.display=s?'':'none'; k+=s; }}); n.textContent=k+' shown'; }}
q.oninput=f; fl.onchange=f; f();
</script></body></html>"""
    open(os.path.join(out, 'index.html'), 'w', encoding='utf-8').write(page)
    print(f'[write] {out}/index.html: {len(rows)} {a.split} clips ({flagged} with unusual words/s)')


if __name__ == '__main__':
    main()
