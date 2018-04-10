import numpy as np
import pandas as pd
import os,sys,re
import cPickle as pickle
import argparse
from bs4 import BeautifulSoup
from nltk.corpus import words
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nlp_utils import LemmaTokenizer, Bigram
from nlp_utils import read_docs, get_topic_words

if __name__ == '__main__':

    TEXT_DATA_DIR = '../text_classification/20_newsgroup/'
    MAX_NB_WORDS = 20000
    NB_TOPICS = 10

    # Read documents
    docs, doc_classes = read_docs(TEXT_DATA_DIR)

    # Prepocess based on perplexity results
    tokenizer = LemmaTokenizer()
    phraser = Bigram()
    token_docs = [tokenizer(doc) for doc in docs]
    bigram_docs = phraser(token_docs)
    vectorizer = CountVectorizer(
        min_df=10, max_df=0.5,
        max_features=MAX_NB_WORDS,
        preprocessor = lambda x: x,
        tokenizer = lambda x: x)
    corpus = vectorizer.fit_transform(bigram_docs)

    # Build model and fit
    lda_model = LatentDirichletAllocation(
        n_components=NB_TOPICS,
        learning_method='online',
        max_iter=10,
        batch_size=2000,
        verbose=1,
        max_doc_update_iter=100,
        n_jobs=-1,
        random_state=0)
    lda = lda_model.fit(corpus)
    doc_topics_mtx = lda.transform(corpus)

    # get words per topic
    topic_words = get_topic_words(lda,vectorizer.get_feature_names(),10)

    # save the results
    df = pd.DataFrame({'document_class': doc_classes})
    df.to_csv("data_processed/document_class.txt", index=False)
    pickle.dump(lda, open("data_processed/lda_model.p", "wb"))
    pickle.dump(topic_words, open("data_processed/topic_words_final_model.p", "wb"))
    pickle.dump(doc_topics_mtx, open("data_processed/document_topics_mtx.p", "wb"))
    # so we can read it with R
    np.save("data_processed/document_topics_mtx.npy", doc_topics_mtx)