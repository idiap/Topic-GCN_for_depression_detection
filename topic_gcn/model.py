# SPDX-FileCopyrightText: Copyright © 2023-2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Topic-GCN: an inductive text GCN (InducT-GCN) over word, document and topic nodes.

Based on the InducT-GCN implementation by Sergio Burdisso (Idiap, 2023), extended
by Ernesto Reyes-Ramírez (CIMAT) with dense node initialisation and topic nodes.

Graph (training):
    nodes:  [ words (V) | training documents (N) | topics (K) ]
    edges:  word-word      PMI (positive values only), PageRank on the diagonal
            word-document  TF-IDF
            word-topic     Tw (topic model)
            document-topic Td (topic model)
            self-loops     1

The adjacency matrix is normalised as D^-1/2 A D^-1/2 and a two-layer GCN is
trained on the training document nodes. At inference, each test document is
connected to the learned word and topic representations (TF-IDF and Td edges)
without retraining.

Without topic model, this is the GCN baseline (one-hot words and TF-IDF documents,
`word_vectors=None`) or Topic-GCN (ft, ft, -) (FastText words and documents).
"""
import warnings
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import log
from collections import defaultdict
from sklearn.preprocessing import normalize


def page_rank(x, df=0.85, max_iter=100):
    """PageRank values of the nodes of the graph whose adjacency matrix is `x`."""
    A = normalize(sp.csr_matrix(x, dtype=float), axis=0, norm="l1")
    R = np.ones(A.shape[0]).reshape(-1, 1)
    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1, 1)
    for _ in range(max_iter):
        R = df * (A * R) + bias
    return R.T[0]


def average_embeddings(words, word_vectors):
    """Average of the embeddings of `words` (zeros if empty)."""
    average = np.zeros(word_vectors.vector_size)
    for word in words:
        average += word_vectors[word]
    if words:
        average /= len(words)
    return average


class TopicGCN(nn.Module):

    def __init__(self, vectorizer, labels, hidden_dim=64, dropout=.5,
                 word_vectors=None, topic_model=None, device="cpu"):
        """
        vectorizer:   fitted TfidfVectorizer whose vocabulary are the word nodes
        labels:       class names, in label index order
        word_vectors: gensim KeyedVectors to initialise the nodes (None: one-hot words, TF-IDF documents)
        topic_model:  one of the adapters in `topic_gcn.topics` (None: no topic nodes)
        """
        super().__init__()
        self.classes_ = labels
        self.word_vectors = word_vectors
        self.topic_model = topic_model
        self.num_topics = topic_model.num_topics if topic_model is not None else 0
        self.device = device

        self._vectorizer = vectorizer
        self._tokenize = vectorizer.build_analyzer()
        self._vocab = vectorizer.vocabulary_
        self._words = [None] * len(self._vocab)
        for word, ix in self._vocab.items():
            self._words[ix] = word

        input_dim = word_vectors.vector_size if word_vectors is not None else len(self._vocab)
        self.get_H_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=False)
        )
        self.node_emb2out = nn.Linear(hidden_dim, len(labels), bias=False)

        self.A_norm = None
        self.Conv_0 = None
        self.H_1_words = None
        self.H_1_topics = None
        self.n_train_docs = None
        self._test_cache = (None, None, None)  # (documents, A_B, Conv_0_B)

    # ----------------------------------------------------------------- node features
    def word_features(self):
        if self.word_vectors is None:
            return torch.eye(len(self._vocab), dtype=torch.float)
        features = np.zeros((len(self._vocab), self.word_vectors.vector_size))
        for ix, word in enumerate(self._words):
            features[ix] = self.word_vectors[word]
        return torch.tensor(features, dtype=torch.float)

    def doc_features(self, documents):
        if self.word_vectors is None:
            return torch.tensor(self._vectorizer.transform(documents).todense(), dtype=torch.float)
        features = np.empty((len(documents), self.word_vectors.vector_size))
        for i, doc in enumerate(documents):
            features[i] = average_embeddings(self._tokenize(doc), self.word_vectors)
        return torch.tensor(features, dtype=torch.float)

    def topic_features(self):
        if self.topic_model is None:
            return torch.zeros((0, self.get_H_1[0].in_features), dtype=torch.float)
        features = np.zeros((self.num_topics, self.word_vectors.vector_size))
        for i in range(self.num_topics):
            features[i] = average_embeddings(self.topic_model.topic_words(i), self.word_vectors)
        return torch.tensor(features, dtype=torch.float)

    # ----------------------------------------------------------------- graph
    def build_graph(self, train_documents, window_size=3, use_pagerank=True):
        V, N, K = len(self._vocab), len(train_documents), self.num_topics
        nodes_n = V + N + K
        self.n_train_docs = N
        print(f"[Topic-GCN] Building graph: {V} words + {N} documents + {K} topics = {nodes_n} nodes")

        # Co-occurrence counts in sliding windows, for PMI
        total_windows = 0
        tok2win_fr = defaultdict(int)
        cotok2win_fr = defaultdict(lambda: defaultdict(int))
        for doc in [self._tokenize(doc) for doc in train_documents]:
            doc = [w for w in doc if w in self._vocab]
            finish_sliding = False
            ix = 0
            while not finish_sliding:
                window = set(doc[ix: ix + window_size])
                for token_i in window:
                    tok2win_fr[token_i] += 1
                    for token_j in window:
                        if token_i != token_j:
                            cotok2win_fr[token_i][token_j] += 1
                finish_sliding = ix + window_size >= len(doc)
                total_windows += 1
                ix += 1

        A = torch.zeros((nodes_n, nodes_n), dtype=torch.float)

        # A_ij = PMI(i, j) if i and j are words (only positive values)
        for token_i in tok2win_fr:
            i = self._vocab[token_i]
            A[i, i] = tok2win_fr[token_i]
            for token_j in cotok2win_fr[token_i]:
                A[i, self._vocab[token_j]] = cotok2win_fr[token_i][token_j]
        invalid_cooccurrences = A == 0
        torch.log(A, out=A)
        torch.nan_to_num(A, neginf=0, out=A)
        tok_win_fr = A[range(V), range(V)]
        torch.sub(A[:V, :V], tok_win_fr, out=A[:V, :V])
        torch.sub(A[:V, :V], tok_win_fr.reshape(-1, 1), out=A[:V, :V])
        A[:V, :V] += log(total_windows)
        A[invalid_cooccurrences] = 0
        A[A < 0] = 0

        # A_ij = TF-IDF(i, j) if i is a word and j a document
        A[:V, V:V + N] = torch.tensor(self._vectorizer.transform(train_documents).todense(), dtype=torch.float).T

        # A_ii = 1
        A[range(nodes_n), range(nodes_n)] = 1

        if self.topic_model is not None:
            A[:V, V + N:] = torch.tensor(self.topic_model.word_topic(self._words), dtype=torch.float)       # Tw
            A[V:V + N, V + N:] = torch.tensor(self.topic_model.doc_topic(train_documents), dtype=torch.float)  # Td

        # A = D^-1/2 A D^-1/2
        D_inv_sqrt = torch.inverse(torch.eye(nodes_n, dtype=torch.float) * A.sum(dim=1)).sqrt()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = torch.chain_matmul(D_inv_sqrt, A, D_inv_sqrt)

        # A_ii = PageRank(i) if i is a word
        if use_pagerank:
            A[range(V), range(V)] = torch.tensor(page_rank(A[:V, :V]), dtype=torch.float)

        H_0 = torch.cat([self.word_features(), self.doc_features(train_documents), self.topic_features()])

        self.A_norm = A.clone().to(self.device)
        self.Conv_0 = torch.mm(A, H_0).to(self.device)

    # ----------------------------------------------------------------- training
    def loss(self, y, class_weight=None):
        """Cross-entropy on the training document nodes."""
        V, N = len(self._vocab), self.n_train_docs
        H_1 = self.get_H_1(self.Conv_0)
        # Learned word and topic representations, used to classify unseen documents
        self.H_1_words = H_1[:V].detach().cpu().clone()
        self.H_1_topics = H_1[V + N:].detach().cpu().clone()

        z_logits = self.node_emb2out(torch.mm(self.A_norm, H_1))[V:V + N]
        return nn.CrossEntropyLoss(weight=class_weight)(z_logits, y)

    # ----------------------------------------------------------------- inference
    def _test_graph(self, documents):
        """Edges (A_B) and first-layer input (A_B H_0) of the unseen documents, cached."""
        if self._test_cache[0] is documents:
            return self._test_cache[1], self._test_cache[2]
        V, B, K = len(self._vocab), len(documents), self.num_topics
        A_B = torch.zeros((B, V + B + K), dtype=torch.float)
        A_B[:, :V] = torch.tensor(self._vectorizer.transform(documents).todense(), dtype=torch.float)
        A_B[:, V:V + B] = torch.eye(B, dtype=torch.float)
        if self.topic_model is not None:
            A_B[:, V + B:] = torch.tensor(self.topic_model.doc_topic(documents), dtype=torch.float)

        H_0_B = torch.cat([self.word_features(), self.doc_features(documents), self.topic_features()])
        Conv_0_B = torch.mm(A_B, H_0_B)
        self._test_cache = (documents, A_B, Conv_0_B)
        return A_B, Conv_0_B

    def forward(self, documents):
        """Class probabilities of unseen documents (computed on CPU)."""
        self.eval()
        with torch.no_grad():
            A_B, Conv_0_B = self._test_graph(documents)
            W_0 = self.get_H_1[0].weight.cpu()
            W_1 = self.node_emb2out.weight.cpu()
            H_1_B = F.relu(F.linear(Conv_0_B, W_0))
            H_1 = torch.cat([self.H_1_words, H_1_B, self.H_1_topics])
            return torch.softmax(F.linear(torch.mm(A_B, H_1), W_1), dim=-1)

    def predict(self, documents):
        return [self.classes_[ix] for ix in torch.argmax(self.forward(documents), dim=-1).tolist()]

    def node_predictions(self):
        """Class probabilities of every node of the training graph (words, training documents, topics)."""
        self.eval()
        with torch.no_grad():
            z = torch.softmax(self.node_emb2out(torch.mm(self.A_norm, self.get_H_1(self.Conv_0))), dim=-1).cpu()
        V, N = len(self._vocab), self.n_train_docs
        return {"words": list(zip(self._words, z[:V].tolist())),
                "documents": z[V:V + N].tolist(),
                "topics": z[V + N:].tolist()}

    # ----------------------------------------------------------------- checkpoints
    def get_state(self):
        return {"weights": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
                "H_1_words": self.H_1_words.clone(), "H_1_topics": self.H_1_topics.clone()}

    def set_state(self, state):
        self.load_state_dict(state["weights"])
        self.H_1_words = state["H_1_words"]
        self.H_1_topics = state["H_1_topics"]
