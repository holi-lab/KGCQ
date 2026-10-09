#!/usr/bin/env python
"""Split a profile JSON into N batches (for parallel API generation) or merge batch dialog.json files.

  python split_merge_batches.py split --input data/profiles/hv_train.json --output_dir tmp/hv_train_batches --n 8
  python split_merge_batches.py merge --inputs tmp/hv_train_batches/*/dialog.json --output data/dialogues/hv_train.json
"""
import argparse
import glob
import json
import os


def split(args):
    data = json.load(open(args.input))
    items = list(data.items())
    bs = (len(items) + args.n - 1) // args.n
    os.makedirs(args.output_dir, exist_ok=True)
    for i in range(args.n):
        batch = dict(items[i * bs:(i + 1) * bs])
        with open(os.path.join(args.output_dir, f"{i}.json"), "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False, indent=4)
        print(f"batch {i}: {len(batch)} profiles")


def merge(args):
    merged = {}
    for pattern in args.inputs:
        for path in sorted(glob.glob(pattern)):
            d = json.load(open(path))
            if isinstance(d, dict):
                merged.update(d)
            else:
                print(f"skip non-dict file {path}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)
    print(f"merged {len(merged)} profiles -> {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("--input", required=True); s.add_argument("--output_dir", required=True); s.add_argument("--n", type=int, default=8)
    m = sub.add_parser("merge"); m.add_argument("--inputs", nargs="+", required=True); m.add_argument("--output", required=True)
    a = ap.parse_args()
    split(a) if a.cmd == "split" else merge(a)
