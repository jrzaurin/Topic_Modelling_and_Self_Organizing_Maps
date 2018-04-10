# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import cPickle as pickle
import argparse
from nltk.corpus import words
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nlp_utils import SimpleTokenizer, StemTokenizer, LemmaTokenizer
from nlp_utils import read_docs, Bigram, WordFilter, pipeline

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_topics", type=int, default=10, help="number of topics")
    args = vars(ap.parse_args())
    NB_TOPICS = args["n_topics"]
    TEXT_DATA_DIR = '../text_classification/20_newsgroup/'
    MAX_NB_WORDS = 20000

    docs, doc_classes = read_docs(TEXT_DATA_DIR)

    # Lda model that will be used through all the experiments
    lda_model = LatentDirichletAllocation(
        n_components=NB_TOPICS,
        learning_method='online',
        max_iter=10,
        batch_size=2000,
        verbose=1,
        max_doc_update_iter=100,
        n_jobs=-1,
        random_state=0)


    ###########################################################################
    # EXPERIMENT 1: CountVectorizer with different tokenizers
    ###########################################################################
    def experiment_1(docs, tokenizer_):

        train_docs, test_docs = train_test_split(docs, test_size=0.25, random_state=0)
        vectorizer = CountVectorizer(min_df=10, max_df=0.5,
            max_features=MAX_NB_WORDS,
            tokenizer = tokenizer_)

        return pipeline(train_docs, test_docs, vectorizer, lda_model)

    basic_pp_exp1, basic_tw_exp1 = experiment_1(docs, SimpleTokenizer)
    stem_pp_exp1, stem_tw_exp1 = experiment_1(docs, StemTokenizer())
    lemma_pp_exp1, lemma_tw_exp1 = experiment_1(docs, LemmaTokenizer())


    ###########################################################################
    # EXPERIMENT 2: CountVectorizer with different tokenizers adding bigrams
    ###########################################################################
    vectorizer = CountVectorizer(
        min_df=10, max_df=0.5,
        max_features=MAX_NB_WORDS,
        preprocessor = lambda x: x,
        tokenizer = lambda x: x)

    def experiment_2(docs, tokenizer_, phraser_):

        tokens = [tokenizer_(doc) for doc in docs]
        ptokens = phraser_(tokens)
        train_docs, test_docs = train_test_split(ptokens, test_size=0.25, random_state=0)

        return pipeline(train_docs, test_docs, vectorizer, lda_model)

    basic_pp_exp2, basic_tw_exp2 = experiment_2(docs, SimpleTokenizer, Bigram())
    stem_pp_exp2, stem_tw_exp2  = experiment_2(docs, StemTokenizer(), Bigram())
    lemma_pp_exp2, lemma_tw_exp2 = experiment_2(docs, LemmaTokenizer(),Bigram())


    ###########################################################################
    # EXPERIMENT 3: CountVectorizer filtering words based on the nltk.words
    # dictionary and the 400k gloveVectors words
    ###########################################################################
    GLOVE_DIR = '../text_classification/glove.6B/'
    glove_words = []
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        glove_words.append(word)
    f.close()

    glovefilter = WordFilter(vocab=set(glove_words))
    wordsfilter = WordFilter(vocab=set(words.words()))
    def experiment_3(docs, wordfilter):

        tokens = [SimpleTokenizer(doc) for doc in docs]
        ftokens = [wordfilter(d) for d in tokens]
        train_docs, test_docs = train_test_split(ftokens, test_size=0.25, random_state=0)

        return pipeline(train_docs, test_docs, vectorizer, lda_model)

    glove_pp, glove_tw = experiment_3(docs,glovefilter)
    words_pp, words_tw = experiment_3(docs,wordsfilter)


    ###########################################################################
    # SAVE
    ###########################################################################
    topic_words = {'basic_exp1': basic_tw_exp1,
                   'stem_exp1': stem_tw_exp1,
                   'lemma_exp1': lemma_tw_exp1,
                   'basic_exp2': basic_tw_exp2,
                   'stem_exp2': stem_tw_exp2,
                   'lemma_exp2': lemma_tw_exp2,
                   'glove': glove_tw,
                   'words': words_tw}
    fname = "_".join(["data_processed/topic_words",str(NB_TOPICS)]) + ".p"
    pickle.dump(topic_words, open(fname, 'wb'))

    topic_words_df = dict()
    for model, result in topic_words.iteritems():
        df = pd.DataFrame()
        for topic in result:
            cols =  [[word[0] for word in topic[1]] for topic in result]
            for i,c in enumerate(cols):
                df["topic_"+str(i)] = c
        topic_words_df[model] = df

    dfname = "_".join(["data_processed/topic_words_df",str(NB_TOPICS)]) + ".p"
    pickle.dump(topic_words_df, open(dfname, 'wb'))

    perplexity = {'basic_exp1': basic_pp_exp1,
                   'stem_exp1': stem_pp_exp1,
                   'lemma_exp1': lemma_pp_exp1,
                   'basic_exp2': basic_pp_exp2,
                   'stem_exp2': stem_pp_exp2,
                   'lemma_exp2': lemma_pp_exp2,
                   'glove': glove_pp,
                   'words': words_pp}
    fname = "_".join(["data_processed/perplexity",str(NB_TOPICS)]) + ".p"
    pickle.dump(perplexity, open(fname, 'wb'))
