#!/usr/bin/env python
"""Prepare the SuperNI CIT data used by Instil / SAPT (no extra dependencies).

Why this exists
---------------
The training/eval data lives in ``data/SuperNI/<task>/{train,dev,test}.json``.
Each split file is simply the authoritative Natural-Instructions task file
(Definition + metadata + Instances) with the ``Instances`` list sliced per split.
This script (re)builds those files from ``allenai/natural-instructions`` (or a
local checkout of it) so the pipeline is reproducible on any machine, even if
``data/`` is missing or you want to regenerate different splits/sizes.

Schema produced (identical to the existing files, consumed by
``instil/data_superni.py``)::

    {  "Definition": ["..."], ...all task metadata...,
       "Instances": [ {"id": ..., "input": ..., "output": [...]}, ... ] }

Usage
-----
    # download the 15 standard SuperNI CIT tasks into data/SuperNI
    python scripts/prepare_superni_data.py

    # use a local clone instead of the network
    git clone --depth 1 https://github.com/allenai/natural-instructions
    python scripts/prepare_superni_data.py --source natural-instructions/tasks

    # custom subset / sizes / destination
    python scripts/prepare_superni_data.py --tasks task363_sst2_polarity_classification,task875_emotion_classification \
        --train_size 1000 --dev_size 100 --test_size 100 --output_dir data

Notes
-----
* Splits are disjoint and reproducible (seeded shuffle by default; ``--no_shuffle``
  keeps file order).  They will not be byte-identical to the authors' original
  indices, but they are valid, deterministic, and comparable across runs.
* Existing files are kept unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

RAW_BASE = "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks"

# The 15 SuperNI CIT tasks (both order-1 and order-2 use this same set).
SUPERNI_TASKS = [
    "task1572_samsum_summary",
    "task363_sst2_polarity_classification",
    "task1290_xsum_summarization",
    "task181_outcome_extraction",
    "task002_quoref_answer_generation",
    "task1510_evalution_relation_extraction",
    "task639_multi_woz_user_utterance_generation",
    "task1729_personachat_generate_next",
    "task073_commonsenseqa_answer_generation",
    "task1590_diplomacy_text_generation",
    "task748_glucose_reverse_cause_event_detection",
    "task511_reddit_tifu_long_text_summarization",
    "task591_sciq_answer_generation",
    "task1687_sentiment140_classification",
    "task875_emotion_classification",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=None,
                   help="comma-separated task names (default: the 15 SuperNI tasks)")
    p.add_argument("--source", default="download",
                   help="'download' (default) or a path to a local "
                        "natural-instructions 'tasks' directory")
    p.add_argument("--output_dir", default="data",
                   help="benchmark root; files go to <output_dir>/<benchmark>/<task>/")
    p.add_argument("--benchmark", default="SuperNI")
    p.add_argument("--train_size", type=int, default=1000)
    p.add_argument("--dev_size", type=int, default=100)
    p.add_argument("--test_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_shuffle", action="store_true",
                   help="keep original file order instead of a seeded shuffle")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing train/dev/test.json")
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()


def fetch_task_json(task: str, source: str, retries: int) -> dict:
    """Return the full Natural-Instructions task dict (download or local)."""
    if source != "download":
        path = os.path.join(source, f"{task}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Point --source at natural-instructions/tasks "
                f"or use --source download."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{RAW_BASE}/{task}.json"
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "instil-prep"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, TimeoutError) as e:  # pragma: no cover
            last_err = e
            wait = 2 * attempt
            print(f"    fetch attempt {attempt}/{retries} failed ({e}); "
                  f"retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(
        f"Could not download {url} after {retries} tries: {last_err}\n"
        f"Fallback: git clone --depth 1 "
        f"https://github.com/allenai/natural-instructions and rerun with "
        f"--source natural-instructions/tasks"
    )


def split_instances(instances, sizes, shuffle, seed):
    idx = list(range(len(instances)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    train_n, dev_n, test_n = sizes
    # Scale down proportionally for tasks with few instances.
    total = train_n + dev_n + test_n
    if len(idx) < total:
        scale = len(idx) / total
        train_n = max(1, int(train_n * scale))
        dev_n = max(1, int(dev_n * scale))
        test_n = len(idx) - train_n - dev_n
    take = lambda a, b: [instances[i] for i in idx[a:b]]
    train = take(0, train_n)
    dev = take(train_n, train_n + dev_n)
    test = take(train_n + dev_n, train_n + dev_n + test_n)
    return {"train": train, "dev": dev, "test": test}


def write_split(task_dict: dict, instances, out_path: str) -> None:
    d = {k: v for k, v in task_dict.items() if k != "Instances"}
    d["Instances"] = instances
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    tasks = args.tasks.split(",") if args.tasks else SUPERNI_TASKS
    sizes = (args.train_size, args.dev_size, args.test_size)
    dest_root = os.path.join(args.output_dir, args.benchmark)
    os.makedirs(dest_root, exist_ok=True)

    print(f"Preparing {len(tasks)} tasks -> {dest_root}")
    prepared, skipped, failed = 0, 0, []
    for task in tasks:
        tdir = os.path.join(dest_root, task)
        splits_exist = all(os.path.exists(os.path.join(tdir, f"{s}.json"))
                           for s in ("train", "dev", "test"))
        if splits_exist and not args.force:
            print(f"  [skip] {task} (already present; use --force to rebuild)")
            skipped += 1
            continue
        try:
            print(f"  [get ] {task}")
            task_dict = fetch_task_json(task, args.source, args.retries)
            instances = task_dict.get("Instances", [])
            if not instances:
                raise ValueError("no Instances in task file")
            parts = split_instances(instances, sizes, not args.no_shuffle, args.seed)
            os.makedirs(tdir, exist_ok=True)
            for split, insts in parts.items():
                write_split(task_dict, insts, os.path.join(tdir, f"{split}.json"))
            print(f"         train={len(parts['train'])} dev={len(parts['dev'])} "
                  f"test={len(parts['test'])}  ({task_dict.get('Definition', [''])[0][:50]!r})")
            prepared += 1
        except Exception as e:  # pragma: no cover
            print(f"  [FAIL] {task}: {e}")
            failed.append(task)

    print(f"\nDone. prepared={prepared} skipped={skipped} failed={len(failed)}")
    if failed:
        print("Failed tasks:", ",".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
