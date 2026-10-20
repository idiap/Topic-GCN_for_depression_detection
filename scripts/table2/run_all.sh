#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
# Table 2: GCN and Topic-GCN on DAIC-WOZ and E-DAIC.
#
# By default every system is trained once with the hyperparameters of the
# Table 2 runs (conf/hparams.yaml). Set HPO=1 to search them again with 100
# Optuna trials per system instead.
#
#   DATASETS="daic-woz e-daic"   datasets to run (default: both)
#   SYSTEMS="gcn topic-gcn-lda"  systems to run (default: all five)
#   EVAL_SPLIT=test              split used for model selection (paper: test)
#   HPO=1                        search the hyperparameters with Optuna
#   DATA_ROOT, MODELS_ROOT, OUT  default: data/, models/ and exp/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATASETS=${DATASETS:-"daic-woz e-daic"}
SYSTEMS=${SYSTEMS:-"gcn topic-gcn-ft topic-gcn-lsa topic-gcn-lda topic-gcn-bertopic"}
EVAL_SPLIT=${EVAL_SPLIT:-test}
EXTRA=""
[ "${HPO:-0}" = 1 ] && EXTRA="--hpo"

for dataset in $DATASETS; do
    for system in $SYSTEMS; do
        python run_topic_gcn.py --dataset "$dataset" --system "$system" --eval-split "$EVAL_SPLIT" \
            --data-root "${DATA_ROOT:-data}" --models-root "${MODELS_ROOT:-models}" --out "${OUT:-exp}" $EXTRA
    done
done

python scripts/table2/results.py --exp "${OUT:-exp}" --eval-split "$EVAL_SPLIT"
