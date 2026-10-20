# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
"""Prints Table 1 (number of topics and C_v coherence) from the trained topic models."""
import os
import json
import argparse

PAPER = {("daic-woz", "lsa"): (7, 0.28), ("daic-woz", "lda"): (13, 0.30), ("daic-woz", "bertopic"): (20, 0.37),
         ("e-daic", "lsa"): (9, 0.31), ("e-daic", "lda"): (13, 0.30), ("e-daic", "bertopic"): (36, 0.39)}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--models-root", default="models")
args = parser.parse_args()

print(f"{'Dataset':10s} {'Model':9s} {'k':>4s} {'C_v':>6s}   {'paper k':>7s} {'paper C_v':>9s}")
for (dataset, model), (paper_k, paper_cv) in PAPER.items():
    path = os.path.join(args.models_root, dataset, model, "coherence.json")
    if os.path.exists(path):
        with open(path) as reader:
            result = json.load(reader)
        k, cv = str(result["k"]), f"{result['coherence']:.2f}"
    else:
        k, cv = "-", "-"
    print(f"{dataset:10s} {model:9s} {k:>4s} {cv:>6s}   {paper_k:>7d} {paper_cv:>9.2f}")
