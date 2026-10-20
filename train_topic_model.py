# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Trains the topic models (LSA, LDA, BERTopic) on the training interviews of one dataset,
split into chunks of 300 tokens, and reports their C_v coherence (Table 1).

    --sweep     train one model per number of topics in --k-range and print their coherence
                (LSA and LDA; BERTopic finds its number of topics by itself)
    otherwise   train the model with --k topics (default: the paper's) and save it under
                <models-root>/<dataset>/<model>/

Examples:
    python train_topic_model.py --dataset daic-woz --model lda --sweep
    python train_topic_model.py --dataset daic-woz --model lda          # k=13, as in the paper
    python train_topic_model.py --dataset e-daic   --model bertopic
"""
import os
import json
import pickle
import argparse

from gensim import corpora
from gensim.models import CoherenceModel

from topic_gcn.data import DATASETS, dataset_path, read_texts, get_analyzer, chunk_documents, chunk_tokens

# Number of topics of the models behind the paper (Table 1)
PAPER_K = {"lsa": {"daic-woz": 7, "e-daic": 9}, "lda": {"daic-woz": 13, "e-daic": 13}}
# LDA passes of the models behind the paper
PAPER_LDA_PASSES = {"daic-woz": 50, "e-daic": 10}
DEFAULT_K_RANGE = list(range(2, 21)) + [30, 40]
CHUNK_LENGTH = 300


# --------------------------------------------------------------------------- LSA
def lsa_fit(chunks, k, n_iter=50, seed=42):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(chunks)
    svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=n_iter, random_state=seed)
    svd.fit(X)
    terms = vectorizer.get_feature_names_out()
    return [[terms[i] for i in component.argsort()[::-1]] for component in svd.components_]


def lsa_coherence(chunks, k, top_n=15):
    texts = [chunk.split() for chunk in chunks]
    dictionary = corpora.Dictionary(texts)
    topics = [words[:top_n] for words in lsa_fit(chunks, k)]
    return CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence="c_v").get_coherence()


def lsa_train(chunks, k, out_dir, top_n=100):
    topic2words = {f"Topic{i + 1}": words[:top_n] for i, words in enumerate(lsa_fit(chunks, k))}
    with open(os.path.join(out_dir, "lsa_topics.pkl"), "wb") as writer:
        pickle.dump(topic2words, writer)
    return lsa_coherence(chunks, k)


# --------------------------------------------------------------------------- LDA
def lda_corpus(chunks):
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    tokenize = get_analyzer()
    texts = [[token for token in tokenize(chunk.lower()) if token not in stop_words] for chunk in chunks]
    dictionary = corpora.Dictionary(texts)
    return texts, dictionary, [dictionary.doc2bow(text) for text in texts]


def lda_fit(chunks, k, passes, seed=0):
    from gensim.models import LdaModel

    texts, dictionary, corpus = lda_corpus(chunks)
    model = LdaModel(corpus, num_topics=k, id2word=dictionary, passes=passes, random_state=seed)
    coherence = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence="c_v").get_coherence()
    return model, dictionary, coherence


def lda_train(chunks, k, passes, out_dir):
    model, dictionary, coherence = lda_fit(chunks, k, passes)
    model.save(os.path.join(out_dir, "lda.gensim"))
    dictionary.save(os.path.join(out_dir, "dictionary.gensim"))
    return coherence


# --------------------------------------------------------------------------- BERTopic
def bertopic_chunks(texts):
    """BERTopic is trained on chunks of 300 raw words (NLTK tokenisation, stop words kept)."""
    from nltk.tokenize import word_tokenize
    chunks = []
    for text in texts:
        chunks.extend(chunk_tokens(word_tokenize(text), CHUNK_LENGTH))
    return chunks


def bertopic_train(texts, out_dir, seed=42):
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from nltk.corpus import stopwords
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    chunks = bertopic_chunks(texts)
    model = BERTopic(
        embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
        umap_model=UMAP(n_neighbors=2, n_components=3, min_dist=0.05, random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=10, min_samples=10, gen_min_span_tree=True, prediction_data=True),
        vectorizer_model=CountVectorizer(ngram_range=(1, 2), stop_words=stopwords.words("english")),
        top_n_words=10,
        language="english",
        calculate_probabilities=True,
        verbose=True,
    )
    topics, _ = model.fit_transform(chunks)
    model.save(out_dir, serialization="safetensors", save_ctfidf=True,
               save_embedding_model="sentence-transformers/all-MiniLM-L6-v2")

    # C_v coherence of the topic words, over the chunks tokenised as for the GCN
    tokenize = get_analyzer()
    texts = [tokenize(chunk) for chunk in chunks]
    dictionary = corpora.Dictionary(texts)
    topic_words = [[word for word, _ in model.get_topic(topic)] for topic in range(len(set(topics)) - 1)]
    coherence = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary,
                               coherence="c_v").get_coherence()
    return len(model.topic_labels_) - 1, coherence


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=["lsa", "lda", "bertopic"], required=True)
    parser.add_argument("--k", type=int, help="Number of topics (LSA and LDA; default: the paper's)")
    parser.add_argument("--sweep", action="store_true", help="Print the coherence for every k in --k-range")
    parser.add_argument("--k-range", type=int, nargs="+", default=DEFAULT_K_RANGE)
    parser.add_argument("--passes", type=int, help="LDA passes (default: the paper's, 50 DAIC-WOZ / 10 E-DAIC)")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--models-root", default="models")
    args = parser.parse_args()

    texts, _ = read_texts(dataset_path(args.data_root, args.dataset, "train"))
    chunks = chunk_documents(texts, CHUNK_LENGTH)
    passes = args.passes or PAPER_LDA_PASSES[args.dataset]
    out_dir = os.path.join(args.models_root, args.dataset, args.model)

    if args.sweep:
        if args.model == "bertopic":
            parser.error("BERTopic chooses its own number of topics; run it without --sweep")
        print(f"{args.dataset} | {args.model} | {len(chunks)} chunks of {CHUNK_LENGTH} tokens")
        print(f"{'k':>4} {'C_v':>7}")
        scores = {}
        for k in args.k_range:
            scores[k] = lsa_coherence(chunks, k) if args.model == "lsa" else lda_fit(chunks, k, passes)[2]
            print(f"{k:>4} {scores[k]:7.4f}", flush=True)
        best = max(scores, key=scores.get)
        print(f"Best: k={best} (C_v {scores[best]:.4f})")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "coherence_sweep.json"), "w") as writer:
            json.dump({"best_k": best, "coherence": scores}, writer, indent=2)
        return

    os.makedirs(out_dir, exist_ok=True)
    if args.model == "bertopic":
        k, coherence = bertopic_train(texts, out_dir)
    else:
        k = args.k or PAPER_K[args.model][args.dataset]
        coherence = lsa_train(chunks, k, out_dir) if args.model == "lsa" else lda_train(chunks, k, passes, out_dir)
    with open(os.path.join(out_dir, "coherence.json"), "w") as writer:
        json.dump({"k": k, "coherence": coherence}, writer, indent=2)
    print(f"{args.dataset} | {args.model} | k={k} topics | C_v = {coherence:.4f} | saved to {out_dir}")


if __name__ == "__main__":
    main()
