# SPDX-FileCopyrightText: Copyright © 2023-2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
"""Training and evaluation of Topic-GCN, as in the experiments of the paper."""
import random
import numpy as np
import torch
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

from .model import TopicGCN

TARGET_METRIC = "f1-score"  # macro-averaged
NO_IMPROVEMENT_LIMIT = 2500


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vectorizer(train_documents, y_train, n_features=250):
    """TF-IDF vectorizer over the `n_features` words with the highest ANOVA F-value with the labels."""
    vectorizer = TfidfVectorizer(stop_words="english")
    if n_features > 0:
        vectorizer.fit(train_documents)
        vocab = vectorizer.get_feature_names_out()
        selector = SelectKBest(f_classif, k=n_features).fit(vectorizer.transform(train_documents), y_train)
        support = selector.get_support()
        vectorizer = TfidfVectorizer(stop_words="english", vocabulary=[vocab[i] for i in range(len(vocab)) if support[i]])
    return vectorizer.fit(train_documents)


def evaluate(model, documents, y):
    with torch.no_grad():
        y_pred = torch.argmax(model(documents), dim=-1).tolist()
    return classification_report(y.tolist(), y_pred, target_names=model.classes_,
                                 digits=4, output_dict=True, zero_division=0)


def train_topic_gcn(X_train, y_train, X_eval, y_eval, labels, learning_rate, num_steps,
                    word_vectors=None, topic_model=None, n_features=250, hidden_dim=64,
                    dropout=.5, window_size=3, use_pagerank=True, seed=17, device="cpu"):
    """
    Trains Topic-GCN with AdamW for `num_steps` full-batch steps, evaluating on
    (X_eval, y_eval) after each step and keeping the best step by macro F1.

    Returns (best macro F1, classification report of the best step, model at the best step).
    """
    set_seed(seed)

    vectorizer = build_vectorizer(X_train, y_train.tolist(), n_features)
    model = TopicGCN(vectorizer, labels, hidden_dim, dropout, word_vectors, topic_model, device)
    model.build_graph(X_train, window_size=window_size, use_pagerank=use_pagerank)

    class_weight = torch.tensor(
        compute_class_weight(class_weight="balanced", classes=np.arange(len(labels)), y=y_train.tolist()),
        dtype=torch.float).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    y_train = y_train.to(device)

    best_value, best_report, best_state = 0, None, None
    no_improvement = 0
    for step in range(num_steps):
        model.train()
        model.zero_grad()
        loss = model.loss(y_train, class_weight=class_weight)
        loss.backward()
        optimizer.step()

        report = evaluate(model, X_eval, y_eval)
        value = report["macro avg"][TARGET_METRIC]
        if value > best_value:
            best_value, best_report, best_state = value, report, model.get_state()
            best_report["step"] = step + 1
            no_improvement = 0
        no_improvement += 1
        if no_improvement > NO_IMPROVEMENT_LIMIT:
            break

    if best_state is not None:
        model.set_state(best_state)
    return best_value, best_report, model
