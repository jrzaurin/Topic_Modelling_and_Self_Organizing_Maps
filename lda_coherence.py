from __future__ import print_function
import os,logging
import numpy as np
import pandas as pd
import cPickle as pickle
from gensim.models import CoherenceModel, LdaMulticore
from gensim.matutils import Sparse2Corpus
from gensim.corpora import Dictionary
from nlp_utils import SimpleTokenizer, StemTokenizer, LemmaTokenizer
from nlp_utils import Bigram, read_docs

pp_10 = pickle.load(open("data_processed/perplexity_10.p", "rb"))
pp_20 = pickle.load(open("data_processed/perplexity_20.p", "rb"))
pp_50 = pickle.load(open("data_processed/perplexity_50.p", "rb"))
all_perplexities = pp_10.values() + pp_20.values() + pp_50.values()
top10_cut = round(np.percentile(all_perplexities, 10), 2)

top10_models = {}
for n,p in zip([10,20,50], [pp_10,pp_20,pp_50]):
	for model, perplexity in p.iteritems():
		if perplexity <= top10_cut:
			top10_models["_".join([model,str(n)])] = perplexity
top10_models

MAX_NB_WORDS = 20000
TEXT_DATA_DIR = '../text_classification/20_newsgroup/'
docs, doc_classes = read_docs(TEXT_DATA_DIR)

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logging.root.level = logging.DEBUG
def model_builder(docs, tokenizer_, phaser_, nb_topics):

	doc_tokens  = [tokenizer_(doc) for doc in docs]
	doc_tokens  = phaser_(doc_tokens)

	id2word = Dictionary(doc_tokens)
	id2word.filter_extremes(no_below=10, no_above=0.5, keep_n=MAX_NB_WORDS)
	corpus = [id2word.doc2bow(doc) for doc in doc_tokens]

	model = LdaMulticore(
	   corpus=corpus,
	   id2word=id2word,
	   decay=0.7,
	   offset=10.0,
	   num_topics=nb_topics,
	   passes=5,
	   batch=False,
	   chunksize=2000,
	   iterations=50)

	return doc_tokens, id2word, model

stem_10_texts, stem_10_dict, stem_10_model = model_builder(docs, StemTokenizer(), Bigram(), 10)
lemma_10_texts, lemma_10_dict, lemma_10_model = model_builder(docs, LemmaTokenizer(), Bigram(), 10)

stem_10_CM = CoherenceModel(
	model=stem_10_model,
	texts=stem_10_texts,
	dictionary=stem_10_dict,
	coherence='c_v')
stem_10_coherence = stem_10_CM.get_coherence()

lemma_10_CM = CoherenceModel(
	model=lemma_10_model,
	texts=lemma_10_texts,
	dictionary=lemma_10_dict,
	coherence='c_v')
lemma_10_coherence = lemma_10_CM.get_coherence()

print("coherence with stemmization and 10 topics: {}".format(stem_10_coherence))
print("coherence with lemmatization and 10 topics: {}".format(lemma_10_coherence))
