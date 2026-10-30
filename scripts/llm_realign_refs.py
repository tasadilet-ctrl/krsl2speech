#!/usr/bin/env python3
"""
E15 step 1: rewrite qazaqstantv references with an LLM so they say exactly
what was spoken in the clip, in the article's clean wording.

Why: E13/E14 showed that DROPPING misaligned qazaqstantv references does not
close its gap to khabar. The two transcripts the recollection stores fail in
opposite ways:
  text_asr    Whisper on the broadcast audio. Boundaries are right (it is what
              the interpreter signs), wording is noisy, no case or punctuation.
  text_human  website article sentences fuzzy-matched to the clip. Wording is
              right, boundaries are shifted -- e.g. a clip's article text ends
              early and its tail is the start of the next clip's text.
The LLM gets the clip's ASR plus the article text of the surrounding clips and
returns the article wording for exactly the ASR's span. It never sees the
current training reference, so it cannot just copy it.

Output is raw: one jsonl row per clip with the prompt inputs and the model's
text. Acceptance (word-count and character-overlap guards against the ASR) is
a separate step, scripts/realign_guard.py, so guards can be tuned without
re-running the model.

Runs in the vLLM env, not the project venv:
    CUDA_VISIBLE_DEVICES=0 ~/skillopt_work/vllm_env/bin/python \
        scripts/llm_realign_refs.py --splits dev --limit 60 --out output/e15/pilot.jsonl
"""
import argparse
import collections
import json
import os

PROMPT = """You are cleaning the reference transcripts of a Kazakh sign-language news corpus.

SPEECH is an automatic speech recognition of exactly what the news anchor said during one short video clip. Its first and last words are the true boundaries of the clip, but words can be misrecognised, and it has no capital letters or punctuation.

ARTICLE is the human-written news text for this broadcast, around the clip. Its wording, spelling, names and numbers are correct, but it is NOT aligned to the clip: it contains sentences said before and after the clip, and it may lack some things that were said.

Write the transcript of exactly what was said in this clip, in clean written Kazakh:
- Cover the same content as SPEECH, from its first word to its last. Do not add anything SPEECH does not contain, and do not drop anything it does.
- Where SPEECH matches the ARTICLE, use the ARTICLE's wording, spelling, capitalisation, numbers and punctuation.
- Where SPEECH has no counterpart in the ARTICLE, keep it, fixing only obvious recognition errors.
- If SPEECH starts or ends in the middle of a sentence, start or end at the same point.
- Speech recognition sometimes repeats a passage by mistake. If SPEECH contains the same passage twice in a row and the ARTICLE has it once, write it once.
- Do not add ellipses or any marks that are not part of the text.

Output only the transcript, nothing else.

ARTICLE:
{article}

SPEECH:
{speech}"""


def article_context(clips, i, k):
    """Article text of clips i-k..i+k of the same video, in time order, with
    consecutive duplicates removed (sibling clips can share a text_human)."""
    parts = []
    for c in clips[max(0, i - k): i + k + 1]:
        h = (c.get('text_human') or '').strip()
        if h and (not parts or parts[-1] != h):
            parts.append(h)
    return '\n'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recollect', default=os.environ.get('QAZ_RECOLLECT'),
                    help='qazaqstantv_kz_fixed.jsonl (default: $QAZ_RECOLLECT)')
    ap.add_argument('--canonical', default='~/asan_canonical')
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    ap.add_argument('--model', default='Qwen/Qwen3.6-27B')
    ap.add_argument('--context', type=int, default=2, help='neighbour clips each side')
    ap.add_argument('--limit', type=int, default=0, help='first N clips per split (pilot)')
    ap.add_argument('--shard', default='0/1', help='i/n: this process takes every n-th clip')
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if not a.recollect:
        ap.error('--recollect is required (or set QAZ_RECOLLECT)')

    by_video, rec = collections.defaultdict(list), {}
    for line in open(os.path.expanduser(a.recollect)):
        d = json.loads(line)
        rec[d['clip_id']] = d
        by_video[d['video_id']].append(d)
    pos = {}
    for v, cs in by_video.items():
        cs.sort(key=lambda c: (c['start'], c['clip_id']))
        for i, c in enumerate(cs):
            pos[c['clip_id']] = (v, i)

    jobs = []
    for split in a.splits:
        rows = json.load(open(os.path.join(os.path.expanduser(a.canonical), 'qazaqstantv',
                                           'annotations', 'kz', f'{split}.json')))
        if a.limit:
            rows = rows[:a.limit]
        for e in rows:
            r = rec[e['clip_id']]
            v, i = pos[e['clip_id']]
            jobs.append({'clip_id': e['clip_id'], 'split': split,
                         'speech': (r.get('text_asr') or '').strip(),
                         'article': article_context(by_video[v], i, a.context),
                         'current': e['text'].strip(),
                         'text_human': (r.get('text_human') or '').strip()})
    si, sn = map(int, a.shard.split('/'))
    jobs = jobs[si::sn]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):                       # resumable
        done = {json.loads(l)['clip_id'] for l in open(a.out)}
    todo = [j for j in jobs if j['clip_id'] not in done and j['speech'] and j['article']]
    skipped = [j for j in jobs if j['clip_id'] not in done and not (j['speech'] and j['article'])]
    print(f'[jobs] {len(jobs)} in shard, {len(done)} already done, {len(todo)} to run, '
          f'{len(skipped)} without speech or article (kept as-is)', flush=True)

    with open(a.out, 'a', encoding='utf-8') as fh:
        for j in skipped:
            fh.write(json.dumps({**j, 'llm': None, 'note': 'no speech or no article'},
                                ensure_ascii=False) + '\n')
    if not todo:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=0.90, max_num_seqs=256,
              limit_mm_per_prompt={'image': 0, 'video': 0})
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    B = 512
    for b in range(0, len(todo), B):
        chunk = todo[b:b + B]
        msgs = [[{'role': 'user', 'content': PROMPT.format(article=j['article'], speech=j['speech'])}]
                for j in chunk]
        outs = llm.chat(msgs, sp, use_tqdm=False,
                        chat_template_kwargs={'enable_thinking': False})
        with open(a.out, 'a', encoding='utf-8') as fh:
            for j, o in zip(chunk, outs):
                fh.write(json.dumps({**j, 'llm': o.outputs[0].text.strip(),
                                     'finish': o.outputs[0].finish_reason},
                                    ensure_ascii=False) + '\n')
        print(f'  {min(b + B, len(todo))}/{len(todo)}', flush=True)


if __name__ == '__main__':
    main()
