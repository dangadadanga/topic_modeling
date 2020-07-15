from pathlib import Path
import os
from datetime import datetime

import numpy as np
import pandas as pd

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.word2vec import LineSentence

from gensim.models import LdaModel
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel

from gensim.models.wrappers.ldamallet import LdaMallet
dirname = os.path.dirname(__file__)
#assummes mallet exists
mallet_path = os.path.join(dirname, '../mallet-2.0.8/bin/mallet')

from multiprocessing import cpu_count
from ansaro_utils.visualization.topic_model_viz import print_topic_terms

import logging
logger = logging.getLogger(__name__)

#assumes the environment variable exists 
LOCAL_DATA = Path(os.environ['LOCAL_DATA'])
data_dir_local = LOCAL_DATA / 'topic_modeling'

data_dir = data_dir_local
data_dir_training = data_dir / 'training'
data_dir_processed = data_dir_training /'processed' #/ 'topic_model'

data_dir_processed.mkdir(parents=True, exist_ok=True)

response_cols = [
    'resp_whytfa',
]

verbose = True

# for topic modeling
#keep the words that appear at least 5 times
min_absolute_frequency = 5
#and to the maximum of 100% (meaning don't truncate anything)
max_relative_frequency = 1 

#keep a maximum of 100000 entries in the dictionary
# max_features = 100000
max_features = None

multicore = False

def topic_model_gensim_lda(col: str, prefix=None, min_topics=19,max_topics=19,step=2) -> None:
    def trigram_bow_generator(filepath: str):
        '''
        generator function to read docs from a file
        and yield a bag-of-words representation
        '''
        for doc in LineSentence(filepath):
            yield trigram_dictionary.doc2bow(doc)

    if prefix is None:
        prefix = ''
    # for topic modeling
    
    trigram_docs_filepath = data_dir_processed / f'{prefix}{col}_transformed_docs_all.txt'
    print(f'Loading input file {trigram_docs_filepath}')
    trigram_dictionary_filepath = data_dir_processed / f'{prefix}{col}_trigram_dict_all.dict'
    trigram_bow_filepath = data_dir_processed / f'{prefix}{col}_trigram_bow_corpus_all.mm'

    #resp_whytfa_trigram_transformed_docs_all.txt

    # turn to posix filepaths until gensim supports this
    # trigram_docs_filepath = trigram_docs_filepath.as_posix()
    trigram_docs_filepath =  trigram_docs_filepath.as_posix()
    trigram_dictionary_filepath = trigram_dictionary_filepath.as_posix()
    trigram_bow_filepath = trigram_bow_filepath.as_posix()

    # TODO - change 1 == 1 lines to overwrite_interim

    # this is a bit time consuming - make the if statement True
    # if you want to learn the dictionary yourself.
    if 1 == 1:
        trigram_docs = LineSentence(trigram_docs_filepath)
        # learn the dictionary by iterating over all of the docs
        trigram_dictionary = Dictionary(trigram_docs)
        print(trigram_dictionary)
        #for k, v in trigram_dictionary.iteritems():
        #    print (f'{k}, {v}')


        # filter tokens that are very rare or too common from
        # the dictionary (filter_extremes) and reassign integer ids (compactify)
        trigram_dictionary.filter_extremes(no_below=min_absolute_frequency,
                                           no_above=max_relative_frequency,
                                           keep_n=max_features,
                                           )
        trigram_dictionary.compactify()
        print(trigram_dictionary)
        #for k, v in trigram_dictionary.iteritems():
        #    print (f'{k}, {v}')

        if verbose:
            logger.info(f'Saving trigram dictionary: {trigram_dictionary_filepath} {len(trigram_dictionary)}')
        trigram_dictionary.save(trigram_dictionary_filepath)

    # load the finished dictionary from disk
    if verbose:
        logger.info(f'Loading trigram dictionary: {trigram_dictionary_filepath}')
    trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)

    # this is a bit time consuming - make the if statement True
    # if you want to build the bag-of-words corpus yourself.
    if 1 == 1:
        # generate bag-of-words representations for
        # all docs and save them as a matrix
        if verbose:
            print(f'Saving corpus: {trigram_bow_filepath}')
        MmCorpus.serialize(trigram_bow_filepath,
                           trigram_bow_generator(trigram_docs_filepath))
    # load the finished bag-of-words corpus from disk
    if verbose:
        print(f'Loading corpus: {trigram_bow_filepath}')
    trigram_bow_corpus = MmCorpus(trigram_bow_filepath)
    num_topics_range = range(min_topics, max_topics + 1, step)

    #iterations = 2000
    #chunksize = 100  # more than the number of docs?

    passes = 10
    # iterations = 400
    iterations = 100
    # chunksize = len(trigram_bow_corpus)
    chunksize = 100  # more than the number of docs?
    eta = 'auto'
    #eval_every = None  # Don't evaluate model perplexity, takes too much time.
    workers=1
    print(f'cpu_count:{cpu_count()}')
    alpha='auto'
    if multicore:
        # for multicore; one fewer than the number of cores
        workers = cpu_count() - 1
        if verbose:
            print(f'Multiprocessing with {workers} cores (one fewer than the number of cores)')
    else:
        # for singnle core; cannot use in multicore
        alpha = 'auto'

    # now_str = datetime.now(timezone('US/Pacific')).strftime('%Y-%m-%d-%H-%M-%S')
    now_str = ''#datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = data_dir_processed / f'{prefix}{col}_gensim_lda_models_{now_str}'
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    # save_dir_s3 = f'{data_dir_processed_s3}/{prefix}{col}_gensim_lda_models_{now_str}'

    # lm_list = []
    c_v = []
    u_mass = []
    perp = []
    #alg='LDA'
    alg='Mallet'

    for num_topics in num_topics_range:

        if(alg == 'Mallet'):
            logger.info('Using Mallet...')
            #try the Mallet implementation
            ldamallet = LdaMallet(mallet_path, corpus=trigram_bow_corpus, num_topics=num_topics, id2word=trigram_dictionary,workers=workers,iterations=iterations)

            ldamallet_filepath = (save_dir / f'gensim_ldamallet_{num_topics}_topics').as_posix()
            ldamallet.save(ldamallet_filepath)

            for t in ldamallet.show_topics(num_topics=-1, num_words=10, formatted=False):
                words = [w[0] for w in t[1]]
                logger.info('topic {:2d}\t{}'.format(t[0], ' '.join(words)))

            # Show Topics
            #print(ldamallet.show_topics(formatted=False))

            # Compute Coherence Score
            cm = CoherenceModel(model=ldamallet, texts=trigram_docs, dictionary=trigram_dictionary, coherence='c_v')
            c_v.append(cm.get_coherence())
            cm = CoherenceModel(model=ldamallet, corpus=trigram_bow_corpus,
                            dictionary=trigram_dictionary, coherence='u_mass')#, processes=workers)
            u_mass.append(cm.get_coherence())
            #perp_lower_bound = ldamallet.log_perplexity(trigram_bow_corpus)
            #perp.append(2**(-perp_lower_bound))
            perp.append(0)

        else:
            logger.info('Using LDA...')
            #TODO: try with and without alpha
            ldamodel = LdaModel(corpus=trigram_bow_corpus, id2word=trigram_dictionary,
                                num_topics=num_topics, passes=passes, iterations=iterations,
                                chunksize=chunksize, eta=eta, #eval_every=eval_every,
                                alpha=alpha,
                                random_state=np.random.RandomState(seed=10101010),
                                )
            #ldamodel = LdaMulticore(corpus=trigram_bow_corpus, id2word=trigram_dictionary,
            #                     num_topics=num_topics, passes=passes, iterations=iterations,
            #                     chunksize=chunksize, eta=eta, #eval_every=eval_every,
            #                     random_state=np.random.RandomState(seed=10101010),
            #                     workers=workers
            #                     )                                 
             
            ldamodel_filepath = (save_dir / f'gensim_lda_{num_topics}_topics').as_posix()
            ldamodel.save(ldamodel_filepath)

            for t in ldamodel.show_topics(num_topics=-1, num_words=50, formatted=False):
                words = [w[0] for w in t[1]]
                logger.info('topic {:2d}\t{}'.format(t[0], ' '.join(words)))

            cm = CoherenceModel(model=ldamodel, texts=trigram_docs,
                            dictionary=trigram_dictionary, coherence='c_v')#, processes=workers)
            c_v.append(cm.get_coherence())
            cm = CoherenceModel(model=ldamodel, corpus=trigram_bow_corpus,
                            dictionary=trigram_dictionary, coherence='u_mass') #, processes=workers)
            u_mass.append(cm.get_coherence())
            perp_lower_bound = ldamodel.log_perplexity(trigram_bow_corpus)
            perp.append(2**(-perp_lower_bound))

    coh_perp = pd.DataFrame(
        data=np.array([c_v, u_mass, perp]).T,
        columns=['c_v', 'u_mass', 'perp'],
        index=list(num_topics_range))
    coh_perp.index.name = 'num_topics'
    coh_perp_filepath = save_dir / 'coherence_perplexity.csv'
    coh_perp.to_csv(coh_perp_filepath)
    logger.info('coherence_docs={0}, coherence_corpus={1}, perplexity={2}'.format(c_v, u_mass, perp))





def load_topic_model_objects(prefix: str, col: str, now_str: str,
                             num_topics: int,
                             data_dir,
                             df: pd.DataFrame=None,
                             normalized: bool=None,
                             minimum_probability: float=None,
                             num_words: int=None,
                             print_term_probability: bool=None,
                             prefix_with_col: bool=None,
                             ):
    '''Wrapper function to load topic model-related objects
    '''
    # whether to normalize the topic loading (for each document, topics add to 1)
    if normalized is None:
        normalized = True
    if minimum_probability is None:
        minimum_probability = 0.0
    if num_words is None:
        num_words = 10
    # whether to print the word probability for each topic
    if print_term_probability is None:
        print_term_probability = False
    if prefix_with_col is None:
        prefix_with_col = False
    if prefix_with_col:
        tc_prefix = col + ' '
    else:
        tc_prefix = ''

    ldamodel_dir = f'{prefix}{col}_gensim_lda_models_{now_str}'

    save_dir_tm = f'{data_dir_processed}/{ldamodel_dir}'

    # load topic model
    ldamodel = load_gensim_ldamodel(num_topics, save_dir_tm)
    logging.info(f'Processing: {col}')
    print_topic_terms(ldamodel, num_words=num_words,print_term_probability=print_term_probability)                                    


def load_gensim_ldamodel(num_topics: int, save_dir: str):
    '''Loads a Gensim LdaModel, optionally from S3

    Writes locally if saved on S3, because Gensim doesn't currently support loading from S3.
    '''
    ldamodel_filepath = f'{save_dir}/gensim_lda_{num_topics}_topics'
    logging.info(f'Loading model: {ldamodel_filepath}')
    return LdaModel.load(ldamodel_filepath)


if __name__ == '__main__':
    for r_col in response_cols:
        topic_model_gensim_lda(r_col, prefix=None)
