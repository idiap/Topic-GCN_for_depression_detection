#!/bin/bash
# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
# Table 1: C_v coherence of the topic models, and training of every model the
# Topic-GCN experiments need (FastText embeddings, LSA, LDA and BERTopic).
#
#   DATASETS="daic-woz e-daic"  datasets to run (default: both)
#   SWEEP=1                     also print the coherence for every number of topics (slow)
#   DATA_ROOT, MODELS_ROOT      default: data/ and models/
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATASETS=${DATASETS:-"daic-woz e-daic"}
DATA_ROOT=${DATA_ROOT:-data}
MODELS_ROOT=${MODELS_ROOT:-models}
ARGS="--data-root $DATA_ROOT --models-root $MODELS_ROOT"

for dataset in $DATASETS; do
    python train_fasttext.py --dataset "$dataset" $ARGS
    for model in lsa lda; do
        if [ "${SWEEP:-0}" = 1 ]; then
            python train_topic_model.py --dataset "$dataset" --model $model --sweep $ARGS
        fi
        python train_topic_model.py --dataset "$dataset" --model $model $ARGS
    done
    python train_topic_model.py --dataset "$dataset" --model bertopic $ARGS
done

python scripts/table1/results.py --models-root "$MODELS_ROOT"
