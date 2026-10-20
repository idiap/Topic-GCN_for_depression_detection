# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
"""Dataset loading, tokenisation and chunking shared by all the scripts."""
import os
import torch

from sklearn.feature_extraction.text import TfidfVectorizer

DATASETS = ("daic-woz", "e-daic")
SPLITS = ("train", "dev", "test")


def dataset_path(data_root, dataset, split):
    return os.path.join(data_root, dataset, f"{split}.tsv")


def read_texts(path):
    """Reads a `<label>\\t<text>` file and returns (texts, labels) as two lists of strings."""
    texts, labels = [], []
    with open(path) as reader:
        for line in reader:
            parts = line.strip().split("\t", 1)
            labels.append(parts[0])
            texts.append(parts[1] if len(parts) > 1 else "")
    return texts, labels


def load_dataset(path, label2ix=None):
    """Reads a `<label>\\t<text>` file and returns (documents, y, ix2label, label2ix)."""
    with open(path) as reader:
        lines = reader.read().split("\n")
    data = [(line.split("\t")[0], line.split("\t")[1]) for line in lines if line]
    document_labels = [label for label, _ in data]
    documents = [doc for _, doc in data]

    ix2label = {}
    if label2ix is None:
        label2ix = {}
        for label in sorted(set(document_labels)):
            label2ix[label] = len(label2ix)
            ix2label[label2ix[label]] = label

    y = torch.tensor([label2ix[label] for label in document_labels], dtype=torch.long)
    return documents, y, ix2label, label2ix


def get_analyzer():
    """The tokeniser used everywhere: scikit-learn's word analyzer with English stop words removed."""
    return TfidfVectorizer(stop_words="english").build_analyzer()


def chunk_tokens(tokens, length):
    """Splits a list of tokens into consecutive chunks of `length` tokens (the last one may be shorter)."""
    return [" ".join(tokens[ix: ix + length]) for ix in range(0, len(tokens), length)]


def chunk_documents(documents, length=300, tokenize=None):
    """Splits every document into chunks of `length` tokens, as done to train the topic models."""
    tokenize = tokenize or get_analyzer()
    chunks = []
    for doc in documents:
        chunks.extend(chunk_tokens(tokenize(doc), length))
    return chunks
