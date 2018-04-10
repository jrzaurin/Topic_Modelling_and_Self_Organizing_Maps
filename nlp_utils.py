import numpy as np
import os,sys,re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import words
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phraser, Phrases
from sklearn.feature_extraction.text import CountVectorizer


def read_docs(text_data_dir):
    """simple wrapper to read the news documents
    """
    docs = []
    doc_classes = []
    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    f = open(fpath)
                    t = f.read()
                    # skip header
                    i = t.find('\n\n')
                    if 0 < i:
                        t = t[i:]
                    t = BeautifulSoup(t).get_text()
                    t = re.sub("[^a-zA-Z]"," ", t)
                    docs.append(t)
                    doc_classes.append(name)
                    f.close()
    return docs, doc_classes


def SimpleTokenizer(doc):
    """Basic tokenizer using gensim's simple_preprocess

    Parameters:
    ----------
    docs (list): list of documents

    Returns:
    ----------
    tokenized documents
    """
    return [t for t in simple_preprocess(doc, min_len=3) if t not in STOPWORDS]


class StemTokenizer(object):
    """Stem tokens in a document

    Parameters:
    ----------
    docs (list): list of documents

    Returns:
    --------
    list of stemmed tokens
    """
    def __init__(self):
        self.stemmer = PorterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in SimpleTokenizer(doc)]


class LemmaTokenizer(object):
    """Lemmatize tokens in a document

    Parameters:
    ----------
    docs (list): list of documents

    Returns:
    --------
    list of lemmatized tokens
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.lemmatizer.lemmatize(t, pos="v") for t in SimpleTokenizer(doc)]


class Bigram(object):
    """Bigrams to get phrases like artificial_intelligence

    Parameters:
    ----------
    docs (list): list of documents

    Returns:
    --------
    the document with bigrams appended at the end
    """
    def __init__(self):
        self.phraser = Phraser
    def __call__(self, docs):
        phrases = Phrases(docs,min_count=20)
        bigram = self.phraser(phrases)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    docs[idx].append(token)
        return docs


class WordFilter(object):
    """Filter words based on a vocabulary

    Parameters:
    ----------
    vocab: the vocabulary used for filtering
    doc  : the document containing the tokens to be filtered

    Returns:
    -------
    filetered document
    """
    def __init__(self, vocab):
        self.filter = vocab
    def __call__(self, doc):
        return [t for t in doc if t in self.filter]


def get_topic_words(topic_model, feature_names, n_top_words):
    """Helper to get n_top_words per topic

    Parameters:
    ----------
    topic_model: LDA model
    feature_names: vocabulary
    n_top_words: number of top words to retrieve

    Returns:
    -------
    topics: list of tuples with topic index, the most probable words and the scores/probs
    """
    topics = []
    for topic_idx, topic in enumerate(topic_model.components_):
        topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        tot_score = np.sum(topic)
        scores = [topic[i]/tot_score for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics.append([topic_idx, zip(topic_words, scores)])
    return topics


def pipeline(train_set, test_set, vectorizer_, topic_model):
    """Simple helper to save typing the same process again and again.

     Parameters:
    ----------
    train/test_set: train and test datasets
    vectorizer_: vectorizer object (e.g. an instantiated CountVectorizer())
    topic_model: lda topic model

    Returns:
    -------
    perplexity: a meassure of perplexity for the model
    top_words : the result of get_topic_words()
    """

    # 1-Vectorize
    tr_corpus = vectorizer_.fit_transform(train_set)
    te_corpus = vectorizer_.transform(test_set)
    n_words = len(vectorizer_.vocabulary_)

    # 2-train model
    model = topic_model.fit(tr_corpus)

    # 3-compute perplexity
    gamma = model.transform(te_corpus)
    perplexity = model.perplexity(te_corpus, gamma)/n_words

    # get vocabulary and return top N words
    features = vectorizer_.get_feature_names()
    top_words = get_topic_words(model,features,10)
    return perplexity,top_words
