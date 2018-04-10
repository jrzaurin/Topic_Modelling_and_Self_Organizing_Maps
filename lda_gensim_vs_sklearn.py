# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import pandas as pd
import time
import random
import gensim,logging
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import LdaMulticore
from gensim.matutils import Sparse2Corpus
from nlp_utils import SimpleTokenizer, read_docs

# logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
# logging.root.level = logging.DEBUG

TEXT_DATA_DIR = '../text_classification/20_newsgroup/'
NB_TOPICS = 10

docs, doc_classes  = read_docs(TEXT_DATA_DIR)
# picking documents at random
random.seed(1981)
rand_docs = random.sample(docs,2500)

# Apply a simple tokenizer based in gensim's simple_preprocess
rand_docs = [SimpleTokenizer(doc) for doc in rand_docs]

#train,test
train_docs, test_docs =  rand_docs[:2000], rand_docs[2000:]

# MODEL PARAMETERS
decay = 0.5
offset = 1.
max_iterations = 10
batch_size = 200
max_e_steps = 100
eval_every = 1
mode = "online"

# PREPARE DATA
# gensim corpus can be prepared using just gensim...
# id2word = gensim.corpora.Dictionary(train_docs)
# id2word.filter_extremes(no_below=20, no_above=0.5)
# gensim_tr_corpus = [id2word.doc2bow(doc) for doc in train_docs]
# gensim_te_corpus = [id2word.doc2bow(doc) for doc in test_docs]

# or using sklearn vectorizer and Sparse2Corpus
vectorizer = CountVectorizer(min_df=20, max_df=0.5,
    preprocessor = lambda x: x, tokenizer=lambda x: x)
sklearn_tr_corpus = vectorizer.fit_transform(train_docs)
sklearn_te_corpus = vectorizer.transform(test_docs)

id2word = dict()
for k, v in vectorizer.vocabulary_.iteritems():
    id2word[v] = k
gensim_tr_corpus = Sparse2Corpus(sklearn_tr_corpus, documents_columns=False)
gensim_te_corpus = Sparse2Corpus(sklearn_te_corpus, documents_columns=False)

# MODELS:
# SKLEARN
lda_sklearn = LatentDirichletAllocation(
    n_components=NB_TOPICS,
    batch_size=batch_size,
    learning_decay=decay,
    learning_offset=offset,
    n_jobs=-1,
    total_samples=len(docs),
    random_state=0,
    verbose=1,
    max_iter=max_iterations,
    learning_method=mode,
    max_doc_update_iter=max_e_steps,
    evaluate_every=eval_every)
start = time.time()
lda_sklearn.fit(sklearn_tr_corpus)
sk_time = time.time() - start

gamma = lda_sklearn.transform(sklearn_te_corpus)
sklearn_perplexity = lda_sklearn.perplexity(sklearn_te_corpus, gamma)

# GENSIM
start = time.time()
lda_gensim_mc = LdaMulticore(
    gensim_tr_corpus,
    id2word=id2word,
    decay=decay,
    offset=offset,
    num_topics=NB_TOPICS,
    passes=max_iterations,
    batch=False,
    chunksize=batch_size,
    iterations=max_e_steps,
    eval_every=eval_every)
gn_time = time.time() - start

log_prep_gensim_mc   = lda_gensim_mc.log_perplexity(gensim_te_corpus)
preplexity_gensim_mc = np.exp(-1.*log_prep_gensim_mc)

print("gensim run time and perplexity: {}, {}".format(gn_time, preplexity_gensim_mc))
print("sklearn run time and perplexity: {}, {}".format(sk_time,sklearn_perplexity))

# Lets have a look to the topics
topic_words = dict()
gensim_topics = lda_gensim_mc.show_topics(formatted=False)
def sklearn_show_topics(model, feature_names, n_top_words):
    sk_topics = []
    for topic_idx, topic in enumerate(model.components_):
        tot_score = np.sum(topic)
        top_words = [(feature_names[i],topic[i]/tot_score)
            for i in topic.argsort()[:-n_top_words - 1:-1]]
        sk_topics.append([topic_idx,top_words])
    return sk_topics
feature_names = vectorizer.get_feature_names()
sklearn_topics = sklearn_show_topics(lda_sklearn, feature_names,10)
topic_words['gensim']  = gensim_topics
topic_words['sklearn'] = sklearn_topics

topic_words_df = dict()
for model, result in topic_words.iteritems():
    df = pd.DataFrame()
    for topic in result:
        cols =  [[word[0] for word in topic[1]] for topic in result]
        for i,c in enumerate(cols):
            df["topic_"+str(i)] = c
    topic_words_df[model] = df

print(topic_words_df['sklearn'])
print(topic_words_df['gensim'])

