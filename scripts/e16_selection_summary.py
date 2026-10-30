import json, sys, numpy as np
from sacrebleu.metrics import CHRF
ch = CHRF()
D, DS = sys.argv[1], sys.argv[2]
def load(p): return {json.loads(l)['clip_id']: json.loads(l) for l in open(p)}
summ = json.load(open(f'{D}/summary.json'))['checkpoints']
for tag in summ:
    a_ = summ[tag]['all']
    a = load(f'{D}/{tag}.predictions.jsonl'); b = load(f'{DS}/{tag}.predictions.jsonl')
    k = list(a); h = [x for x in a[k[0]] if 'hyp' in x][0]
    dist = len(set(a[c][h] for c in k))
    dl = np.array([ch.sentence_score(a[c][h], [a[c]['ref']]).score - ch.sentence_score(b[c][h], [b[c]['ref']]).score for c in k])
    rng = np.random.default_rng(0); bs = [rng.choice(dl, len(dl)).mean() for _ in range(2000)]
    q, kh = a_['qazaqstantv'], a_['khabar']
    take = a_['all']['chrf'] >= 26 and a_['all']['content_recall'] >= 0.08
    print(f"{tag}: chrF {a_['all']['chrf']:.2f} recall {a_['all']['content_recall']:.3f} distinct {dist}/600 "
          f"takeoff {'YES' if take else 'no'} | infor {a_['informburo']['chrf']:.2f} khabar {kh['chrf']:.2f} "
          f"qazaq {q['chrf']:.2f} (recall {q['content_recall']:.3f}, len {q['hyp_len_ratio']:.2f}) "
          f"qz/kh {q['chrf']/kh['chrf']:.3f} | coupling {dl.mean():.2f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]")
