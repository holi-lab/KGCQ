"""Rebuild the exp6 HV training subgraphs on the augmented graph with the exp6 HG's own predictions (variant a)
or gold-anchored (variant b), matching the inference-time subgraph construction.

  python build_hv_v3subgraph_data.py --gpu 0 --hg_ratio 35
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import os, sys, json, argparse, copy

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', default='0')
ap.add_argument('--top_k', type=int, default=2)
ap.add_argument('--tau', type=float, default=0.005)
ap.add_argument('--ratios', default='0,5,10,15,20,25,30,35')
ap.add_argument('--hg_ratio', type=int, default=35)
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

HERE = os.path.dirname(os.path.abspath(__file__))
import _config as C          # noqa  (scripts/oog/_config.py)
import _subgraph as SG       # noqa  (scripts/oog/_subgraph.py)
import pandas as pd          # noqa
from kgcq.graph import DiagnosticKnowledgeGraph      # noqa
from kgcq.models import EmbeddingModel               # noqa
sys.path.insert(0, HERE)
from inference_exp34 import HGAdapterDetector

RATIOS = [int(x) for x in args.ratios.split(',')]

hg_nodes = pd.read_csv(C.KG['augmented']['nodes'])
disease_names_hg = [r['name'] for _, r in hg_nodes.iterrows() if r['label'] == 'Disease']

v3_nodes = pd.read_csv(C.KG['augmented']['nodes'])
v3_edges = pd.read_csv(C.KG['augmented']['edges'])
v3_dis = [r['name'] for _, r in v3_nodes.iterrows() if r['label'] == 'Disease']
assert v3_dis == disease_names_hg, "disease order differs from the HG label order"
print(f"[KG] HG labels {len(disease_names_hg)}  nodes {len(v3_nodes)} edges {len(v3_edges)}", flush=True)

emb = EmbeddingModel(C.EMB_MODEL)
kg = DiagnosticKnowledgeGraph(v3_nodes, v3_edges, embedding_model=emb)
meta = {
    'name2id': {str(r['name']).strip().lower(): r['id'] for _, r in v3_nodes.iterrows() if r['label'] == 'Disease'},
    'id2name': dict(zip(v3_nodes['id'], v3_nodes['name'])),
    'id2label': dict(zip(v3_nodes['id'], v3_nodes['label'])),
    'disease_ids': [r['id'] for _, r in v3_nodes.iterrows() if r['label'] == 'Disease'],
}

hg_dir = f'{P.MODEL_DIR}/exp6_hg_r{args.hg_ratio:02d}_Qwen2.5-7B'
print(f"[HG] {hg_dir}", flush=True)
hg = HGAdapterDetector(C.BASE_MODEL, hg_dir, disease_names_hg)


def hg_predict(dialogue):
    """HG top-k and probabilities for a dialogue (greeting excluded), as at inference."""
    msgs = dialogue[1:] if dialogue and len(dialogue) > 1 else dialogue
    res = hg.rank_diseases(messages=msgs, gold_disease_list=None, top_k=args.top_k)
    res = res if isinstance(res, dict) else res[0]
    seeds = [d for d, _ in res['ranked_list']]
    probs = res.get('all_probs') or {}
    return seeds, probs


def item_dialogue(item):
    for k, v in item.items():
        if k in ('ground_truth', 'subgraph', '_meta'):
            continue
        if isinstance(v, dict) and v.get('dialogue') and len(v['dialogue']) > 1:
            return v['dialogue']
    return None


sup_path = f'{P.TRAIN_DATA_DIR}/data_exp6_hv/train_r{max(RATIOS):02d}_hv_symcentric.json'
superset = json.load(open(sup_path))
print(f"[cache] HG predictions for {len(superset)} dialogues (r{max(RATIOS)} superset)", flush=True)
cache = {}
n_nodlg = 0
for i, (hadm, item) in enumerate(superset.items()):
    dlg = item_dialogue(item)
    if dlg is None:
        cache[hadm] = None
        n_nodlg += 1
        continue
    seeds, probs = hg_predict(dlg)
    cache[hadm] = {'seeds': seeds, 'probs': probs}
    if (i + 1) % 300 == 0:
        print(f"    {i+1}/{len(superset)}", flush=True)
print(f"[cache] done; dialogues without a usable turn: {n_nodlg}", flush=True)


def build_sg(seeds, probs):
    if not seeds:
        return [], set()
    return SG.build_subgraph_3hop(kg, meta, seeds, probs, tau=args.tau, fmt='symptom')


for variant in ('a', 'b'):
    out = f'{P.TRAIN_DATA_DIR}/data_exp6_hv_v3{variant}'
    os.makedirs(out, exist_ok=True)
for R in RATIOS:
    data = json.load(open(f'{P.TRAIN_DATA_DIR}/data_exp6_hv/train_r{R:02d}_hv_symcentric.json'))
    outA, outB = {}, {}
    statA = {'gold_in': 0, 'empty': 0, 'lines': 0, 'n': 0}
    statB = {'gold_in': 0, 'empty': 0, 'lines': 0, 'n': 0}
    for hadm, item in data.items():
        c = cache.get(hadm)
        gt = item.get('ground_truth') or []
        gold = str(gt[0]).strip().lower() if gt else None
        seedsA = c['seeds'] if c else []
        probsA = c['probs'] if c else {}
        # A: HG-pred (realistic)
        linesA, dzA = build_sg(seedsA, probsA)
        seedsB = ([gold] if gold else []) + seedsA
        probsB = dict(probsA)
        if gold:
            probsB[gold] = 1.0
        linesB, dzB = build_sg(seedsB, probsB)

        ia = copy.copy(item); ia['subgraph'] = linesA; outA[hadm] = ia
        ib = copy.copy(item); ib['subgraph'] = linesB; outB[hadm] = ib
        for st, lines, dz in ((statA, linesA, dzA), (statB, linesB, dzB)):
            st['n'] += 1
            st['lines'] += len(lines)
            if not lines:
                st['empty'] += 1
            if gold and gold in dz:
                st['gold_in'] += 1
    json.dump(outA, open(f'{P.TRAIN_DATA_DIR}/data_exp6_hv_v3a/train_r{R:02d}_hv_symcentric.json', 'w'), ensure_ascii=False)
    json.dump(outB, open(f'{P.TRAIN_DATA_DIR}/data_exp6_hv_v3b/train_r{R:02d}_hv_symcentric.json', 'w'), ensure_ascii=False)

    def fmt(s):
        n = max(1, s['n'])
        return f"gold_in {s['gold_in']}/{s['n']} ({100*s['gold_in']/n:.0f}%)  empty {s['empty']}  avg_lines {s['lines']/n:.1f}"
    print(f"  r{R:02d} ({len(data)})  A[{fmt(statA)}]  B[{fmt(statB)}]", flush=True)

print("\ndone -> data_exp6_hv_v3a/ (HG-predicted)  data_exp6_hv_v3b/ (gold-anchored)")
