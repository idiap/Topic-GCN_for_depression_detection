# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Trains the FastText embeddings used to initialise the word, document and topic nodes,
on the training interviews of one dataset.

Example:
    python train_fasttext.py --dataset daic-woz
    # -> models/daic-woz/fasttext/fasttext.model
"""
import os
import argparse

from gensim.models import FastText

from topic_gcn.data import DATASETS, dataset_path, read_texts, get_analyzer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--dim", type=int, default=200, help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (the models behind the paper used 30)")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="More than 1 makes training non-deterministic")
    args = parser.parse_args()

    texts, _ = read_texts(dataset_path(args.data_root, args.dataset, "train"))
    tokenize = get_analyzer()
    sentences = [tokenize(text) for text in texts]

    model = FastText(sentences=sentences, vector_size=args.dim, window=args.window, min_count=args.min_count,
                     epochs=args.epochs, seed=args.seed, workers=args.workers)

    out_dir = os.path.join(args.models_root, args.dataset, "fasttext")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "fasttext.model"))
    print(f"FastText model ({len(model.wv)} words, dim {args.dim}, {args.epochs} epochs) saved to {out_dir}/fasttext.model")


if __name__ == "__main__":
    main()
