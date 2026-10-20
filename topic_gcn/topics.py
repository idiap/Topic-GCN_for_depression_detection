# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-License-Identifier: Apache-2.0
#
"""
Adapters that expose the three topic models (LSA, LDA, BERTopic) through the same
interface, so Topic-GCN can build its topic nodes and edges from any of them.

Every adapter provides:
  - `num_topics`
  - `topic_words(i)`:      words whose embeddings are averaged to initialise topic node i;
  - `word_topic(words)`:   (|words| x K) matrix with the word-to-topic edge weights Tw;
  - `doc_topic(docs)`:     (|docs| x K) matrix with the document-to-topic edge weights Td.
"""
import pickle
import numpy as np

from collections import Counter

from .data import get_analyzer

TOPIC_MODELS = ("none", "lsa", "lda", "bertopic")


class LDATopics:
    """
    gensim LDA model. Tw = P(w|t), Td = P(t|d).

    gensim infers P(t|d) from a random initialisation drawn from the model's random
    state, which advances with every call. We reset it before each document so that
    a document always gets the same topics, whatever was inferred before (in the
    paper's runs it was not reset, which made the LDA results depend on the number
    of Optuna trials run before them).
    """

    def __init__(self, model_path, dictionary_path, n_init_words=10, min_word_prob=0.001, seed=0):
        from gensim import corpora
        from gensim.models import LdaModel
        from nltk.corpus import stopwords

        self.model = LdaModel.load(model_path)
        self.dictionary = corpora.Dictionary.load(dictionary_path)
        self.num_topics = self.model.num_topics
        self.n_init_words = n_init_words
        self.min_word_prob = min_word_prob
        self.seed = seed
        self._tokenize = get_analyzer()
        self._stop_words = set(stopwords.words("english"))

    def preprocess(self, doc):
        return [token for token in self._tokenize(doc.lower()) if token not in self._stop_words]

    def topic_words(self, i):
        return [word for word, _ in self.model.show_topic(i, topn=self.n_init_words)]

    def word_topic(self, words):
        matrix = np.zeros((len(words), self.num_topics))
        for i, word in enumerate(words):
            word_id = self.dictionary.token2id.get(word)
            if word_id is not None:
                for j, prob in self.model.get_term_topics(word_id, minimum_probability=self.min_word_prob):
                    matrix[i][j] = prob
        return matrix

    def doc_topic(self, docs):
        matrix = np.zeros((len(docs), self.num_topics))
        for i, doc in enumerate(docs):
            self.model.random_state = np.random.RandomState(self.seed)
            for j, prob in self.model.get_document_topics(self.dictionary.doc2bow(self.preprocess(doc))):
                matrix[i][j] = prob
        return matrix


class LSATopics:
    """
    LSA topics, stored as {topic name: list of its top words}.
    Tw is 1 when the word is among the topic words; Td is the fraction of the
    document tokens that are topic words.
    """

    def __init__(self, model_path):
        with open(model_path, "rb") as reader:
            self.topic2words = pickle.load(reader)
        self.topic_names = list(self.topic2words.keys())
        self.num_topics = len(self.topic_names)
        self._tokenize = get_analyzer()

    def topic_words(self, i):
        return self.topic2words[self.topic_names[i]]

    def word_topic(self, words):
        matrix = np.zeros((len(words), self.num_topics))
        for i, word in enumerate(words):
            for j, topic in enumerate(self.topic_names):
                if word.lower() in self.topic2words[topic]:
                    matrix[i][j] = 1
        return matrix

    def doc_topic(self, docs):
        matrix = np.zeros((len(docs), self.num_topics))
        for i, doc in enumerate(docs):
            tokens = [token.lower() for token in self._tokenize(doc)]
            for j, topic in enumerate(self.topic_names):
                counter = Counter(token for token in tokens if token in self.topic2words[topic])
                matrix[i][j] = sum(counter.values()) / len(tokens)
        return matrix


class BERTopicTopics:
    """
    BERTopic model saved with safetensors serialisation (a folder).
    Tw is the c-TF-IDF weight of the word in the topic representation; Td is the
    topic probability of the document (outlier topic -1 excluded).
    """

    def __init__(self, model_path, embedding_model="all-MiniLM-L6-v2"):
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        self.model = BERTopic.load(model_path, embedding_model=SentenceTransformer(embedding_model))
        self.num_topics = len(self.model.topic_labels_) - 1  # without the outlier topic (-1)

    def topic_words(self, i):
        return [word for word, _ in self.model.topic_representations_[i]]

    def word_topic(self, words):
        matrix = np.zeros((len(words), self.num_topics))
        for j in range(self.num_topics):
            topic = dict(self.model.topic_representations_[j])
            for i, word in enumerate(words):
                if word in topic:
                    matrix[i][j] = topic[word]
        return matrix

    def doc_topic(self, docs):
        _, probs = self.model.transform(docs)
        matrix = np.zeros((len(docs), self.num_topics))
        for i in range(len(docs)):
            matrix[i] = probs[i][1:]
        return matrix


def load_topic_model(name, path, dictionary_path=None):
    if name == "none":
        return None
    if name == "lda":
        return LDATopics(path, dictionary_path)
    if name == "lsa":
        return LSATopics(path)
    if name == "bertopic":
        return BERTopicTopics(path)
    raise ValueError(f"Unknown topic model '{name}' (valid: {TOPIC_MODELS})")
