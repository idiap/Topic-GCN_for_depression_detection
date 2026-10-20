# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
"""Prints Table 2 (macro F1 and per-class F1) from the results.json files of the runs."""
import os
import json
import argparse

# Baselines are not part of this repository: their rows show the numbers of the paper
PAPER_BASELINES = [("MentalBERT", ["0.62", "0.55", "0.68", "0.68", "0.60", "0.77"]),
                   ("LongBERT", ["0.63", "0.44", "0.81", "0.64", "0.64", "0.64"]),
                   ("GPT-4.1 (zero-shot)", ["0.80", "0.73", "0.88", "0.73", "0.63", "0.83"])]
ROWS = [("gcn", "GCN", "(ohe, tf-idf, -)", "n/a"), ("topic-gcn-ft", "Topic-GCN", "(ft, ft, -)", "n/a"),
        ("topic-gcn-lsa", "Topic-GCN", "(ft, ft, ft)", "LSA"), ("topic-gcn-lda", "Topic-GCN", "(ft, ft, ft)", "LDA"),
        ("topic-gcn-bertopic", "Topic-GCN", "(ft, ft, ft)", "BERTopic")]
DATASETS = ["daic-woz", "e-daic"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--exp", default="exp")
parser.add_argument("--eval-split", default="test")
args = parser.parse_args()


def scores(dataset, system):
    path = os.path.join(args.exp, dataset, system, args.eval_split, "results.json")
    if not os.path.exists(path):
        return ["-"] * 3
    with open(path) as reader:
        report = json.load(reader)["report"]
    return [f"{report[k]['f1-score']:.2f}" for k in ("macro avg", "positive", "negative")]


print(f"Results on the {args.eval_split} split: macro F1 (Avg.) and F1 of the depressed (D) and control (C) classes\n")
print(f"| Model | Init. | Topic model | {' | '.join(f'{d} Avg. | D | C' for d in DATASETS)} |")
print("|---|---|---|" + "---:|" * 6)
for name, cells in PAPER_BASELINES:
    print(f"| {name} (paper) | n/a | n/a | {' | '.join(cells)} |")
for system, name, init, topic in ROWS:
    cells = [c for d in DATASETS for c in scores(d, system)]
    print(f"| {name} | {init} | {topic} | {' | '.join(cells)} |")
