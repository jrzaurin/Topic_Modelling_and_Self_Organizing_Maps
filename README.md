# LDA_and_SOMs
Topic modelling (Latent Dirichlet Allocation) and Self Organizing Maps (in python and R)

Here is the code for a "fun" excercise using LDA and SOMs. For this excercise I have used the 20newsgroups dataset which can be downloaded from [here](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html).

The "flow" of the repo is the following (one could start from 4): 

1. `LDA_perplexity_demo.ipynb`: this is a demo explaining the experiments I have run using different pre-processing for the documents that are latter passed to the LDA model. The companion python script is `lda_perplexity.py`. This script can be run as: 
    
    `python lda_perplexity.py --n_topics N`

     where N is the number of topics. I run the script with 10, 20 and 50 topics and the results are in the directory
    `data_processed/` in the form of pickle files.

2. `LDA_coherence_demo.ipynb`: here I simply explore with the `CoherenceModel` method implemented in `gensim`. The companion script is `lda_coherence.py`

3. `LDA_gensim_vs_sklearn_demo.ipynb`: this is a comparison between the LDA implementation in `sklearn` and `gensim`. The companion script is `lda_gensim_vs_sklearn.py`

4. `lda_build_model.py`: if one is interested in quickly running a model and and move to SOMs, skip 1, 2 and 3 and simply run: 

    `python lda_build_model.py`
    
    just make sure you have downloaded the  20newsgroups group and that the directory names in the script are consistent with     those in your working dir. Also the results are in `data_processed`, so the `topic_SOM` notebooks will run just after         cloning this repo (provided that the neccesary packages are installed. Speaking of which, `ipdb` when using `jupyter` will     only run if the version is 0.8.1 or lower).
    
    The companion `LDA_build_model.ipynb` shows the pyLDAvis plot for that final model 
    
5. `topic_SOMs.ipynb` and `topic_SOMs_R.ipynb`: using self organizing maps to visualize the results from the topic-modeling. I also show how one could build a "semantic fingerprint" based on articles read.  

And that's it. Nothing major. 
