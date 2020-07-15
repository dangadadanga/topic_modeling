import data.preprocessor as preprocessor
import data.phrase_docs as phraser
import model.topic_model as topic_model
import ansaro_utils.features.topic_model_load as model_loader
from pathlib import Path
import os
import numpy as np
import datetime
import pandas as pd

import spacy

import logging
logging.basicConfig(filename="topic_modeling.log",
                            filemode='a',
                            format='{asctime} : {levelname} : {message}', level=logging.INFO, style='{')


LOCAL_DATA = Path(os.environ['LOCAL_DATA'])
data_dir_local = LOCAL_DATA / 'topic_modeling'
data_dir = Path(data_dir_local)
data_dir_processed = data_dir /'processed'

data_dir_training = data_dir / 'training'
data_dir_inference = data_dir / 'inference'
cols = ['resp_whytfa']
as_text=True
as_pickle=False
verbose=True
prefix = ''
now_str = ''
normalized = True
print_term_probability = False
prefix_with_col = True


def training_pipeline(filepath=None, num_words_to_print=10,prefix=None,min_topics=19,max_topics=19,step=2):

    logging.info(f'Started training_pipeline : {min_topics}-{max_topics}')
    start=datetime.datetime.now()

    if filepath is not None:
        filepath = data_dir_local / filepath
    else:
        logging.error("Please enter file name")
        exit()
    if max_topics is None:
        logging.error("Please enter a valid topic number to train model")
        exit()

    logging.info(f'preprocessor.process_data_save: {filepath}')
    #preprocessor.process_data_save(filepath=filepath, as_text=as_text, as_pickle=as_pickle, verbose=verbose)
    logging.info(f'phraser.raw_to_phrased_data_pipeline...')
    #phraser.raw_to_phrased_data_pipeline(to_load='text', verbose=True, overwrite_interim=True, prefix=None)
    phrased_docs = phraser.load_phrased_data_pipeline(to_load='text', verbose=True, overwrite_interim=True, prefix=None, training=True,col='resp_whytfa')

    #default it to min/max topics
    num_topics_range = range(min_topics, max_topics + 1, step)
    #if num_topics is not None:
    #    num_topics_range = range(num_topics, num_topics + 1, step)
    print('Num_topics_range={}'.format(num_topics_range))

    for col in cols:
        logging.info(f'topic_model.topic_model_gensim_lda(col={col})')
        topic_model.topic_model_gensim_lda(col, prefix, min_topics,max_topics,step)
 
        for num_topics in num_topics_range:
            logging.info(f'Running training for num_topics={num_topics}')
            topic_model.load_topic_model_objects(
              prefix, col, now_str,
              num_topics, data_dir,
              normalized=normalized,
              print_term_probability=print_term_probability,
              num_words=num_words_to_print,
              prefix_with_col=prefix_with_col)
    #calculate duration
    dif= datetime.datetime.now() - start
    logging.info(f'Process took {str(dif)}')


def update_model(num_topics: int, filepath):
    if filepath is not None:
        filepath = data_dir_local / filepath
    else:
        logging.error("Please enter file name")
        exit()
    if num_topics is None:
        logging.error("Please enter a valid topic number to train model")
        exit()
    preprocessor.process_data_save(filepath=filepath, as_text=as_text, as_pickle=as_pickle, verbose=verbose,
                                   training=True, update=True)
    phrased_docs = phraser.raw_to_phrased_data_pipeline(to_load='text', verbose=True, overwrite_interim=True, update=True)
    # phrased_docs = model_loader.load_df_phrased(f'{data_dir_training}/processed/resp_whytfa_phrased.csv')
    for col in cols:
        ldamodel_dir = f'{prefix}{col}_gensim_lda_models_{now_str}'
        load_dir = f'{data_dir_training}/processed/{ldamodel_dir}'
        ldamodel = model_loader.load_gensim_ldamodel(num_topics=num_topics, save_dir=load_dir)
        id2word = ldamodel.id2word
        corpus = [id2word.doc2bow(text.split()) for text in phrased_docs[col]]
        ldamodel.update(corpus)
        current = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        ldamodel_dir = f'{prefix}{col}_gensim_lda_models_{current}'
        save_dir = data_dir_training / f'processed/{ldamodel_dir}'
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        ldamodel_filepath = save_dir / f'gensim_lda_{num_topics}_topics'
        ldamodel.save(ldamodel_filepath.as_posix())


def print_dictionary(prefix=None):

     for col in cols:

          trigram_dictionary_filepath = data_dir_processed / f'{prefix}{col}_trigram_dict_all.dict'
          # turn to posix filepaths until gensim supports this
          trigram_dictionary_filepath = trigram_dictionary_filepath.as_posix()
          
          trigram_bow_filepath = data_dir_processed / f'{prefix}{col}_trigram_bow_corpus_all.mm'
          trigram_bow_filepath = trigram_bow_filepath.as_posix()

          #load the data
          trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)
          trigram_bow_corpus = MmCorpus(trigram_bow_filepath)
          
          #sort trigram_bow_corpus descending by frequencea
          #corpus = sorted(trigram_bow_corpus.items(),key=operator.itemgetter(1),reverse=True)
          
          corpus = OrderedDict(sorted(trigram_bow_corpus.items(),
                                  key=lambda kv: kv[1], reverse=True))

          print (len(trigram_bow_corpus)) 
          print([[(trigram_dictionary[id], freq) for id, freq in cp] for cp in corpus[:1]])


def print_topics(num_topics=19,num_words_to_print=10,prefix=None):
    for col in cols:
        topic_model.load_topic_model_objects(
            prefix, col, now_str,
            num_topics, data_dir,
            normalized=normalized,
            print_term_probability=print_term_probability,
            num_words=num_words_to_print,
            prefix_with_col=prefix_with_col)


def inference_pipeline(filename=None, num_topics=None):
    if filename is not None:
        filepath = data_dir_inference / filename
    else:
        print("Please enter file name")
        exit()
    if num_topics is None:
        print("Please enter a valid topic number of trained model")
        exit()
    print('processing the input file...')

    #preprocess the test file 
    #preprocessor.process_data_save(filepath=filepath, as_text=as_text, as_pickle=as_pickle, verbose=verbose, training=False)
    #phrased_docs = phraser.raw_to_phrased_data_pipeline(to_load='text', verbose=True, overwrite_interim=True, prefix=None, training=False)
 
    #if this part has already been done 
    phrased_docs = phraser.load_phrased_data_pipeline(to_load='text', verbose=True, overwrite_interim=True, prefix=None, training=True,col='resp_whytfa')

    for col in cols:
        ldamodel_dir = f'{prefix}{col}_gensim_lda_models_{now_str}'
        save_dir = f'{data_dir_training}/processed/{ldamodel_dir}'
        print(f'Loading model data from {save_dir}') 
        ldamodel = model_loader.load_gensim_ldamodel(num_topics=num_topics, save_dir=save_dir)
        print(ldamodel.print_topics())

        result = [[]] 

        allocation_counts= np.zeros(num_topics,dtype = int)

        row_count = len(phrased_docs.index)
        print(row_count)
        for row in range(1,row_count):
        #for doc in phrased_docs[col]:
            TFAUID=phrased_docs.iloc[[row]]['tfa_master_uid'].values[0]
            doc=phrased_docs.iloc[[row]]['resp_whytfa'].values[0]

            row = [TFAUID] 
            print(f"Phrased doc (TFAUID={TFAUID},doc={doc}... ")
            id2word = ldamodel.id2word
            corpus = id2word.doc2bow(doc.split())
            print(f'Get the topics for the TFAUID {TFAUID}...')
            doc_topics = ldamodel.get_document_topics(corpus, minimum_probability=0)
            for topic in doc_topics:
                row.append(topic[1])
                print(topic)
            result.append(row)

            #get the max for each row
            max_index=row.index(max(row[1:]))-1
            #print("Max: {}, Index: {}".format(max(row[1:]),max_index))
            #print("Row Results: {0} - {1} - {2}".format(row, topic[1], max_index))
            #increase counts for that topic number
            allocation_counts[max_index]=allocation_counts[max_index]+1

        df = pd.DataFrame(result, columns=range(0, num_topics + 1))
        df.to_csv(data_dir_inference / ('inference_result_' + now_str + '_' + filename))

        print("Overall topic distribution={0} out of {1}".format(allocation_counts,row_count))


if __name__ == '__main__':

    #nlp = spacy.load('en', disable=[])
    #print(nlp.Defaults.stop_words)

    #training_pipeline(filepath="2017_2018_whytfa_with_header.csv",num_words_to_print=100,prefix='',min_topics=10,max_topics=10)
    training_pipeline(filepath="whytfa_14_15_16_17_18_19.csv",num_words_to_print=20,prefix='',min_topics=20,max_topics=20)
    #print_topics(30, 20, '')

    #inference_pipeline('2017_2018_whytfa_with_header.csv', 10)
    #inference_pipeline('2019_whytfa.csv', 20)

    #training_pipeline('whytfa_2017_2018.csv', 25)
    #print_topics(10, 50, '')

    #training_pipeline('whytfa_2017_2018.csv', 19)
    #train for just one topic number
    #training_pipeline(filepath="2018_whytfa_with_header.csv",num_words_to_print=10,prefix='',min_topics=13,max_topics=13,step=2,num_topics=19)
    #training_pipeline(filepath="whytfa_14_15_16_17_18_19.csv",num_words_to_print=100,prefix='',min_topics=20,max_topics=20,step=2,num_topics=20)

    #train for a range
    #training_pipeline(filepath="whytfa_14_15_16_17_18_19.csv",num_words_to_print=20,prefix='',min_topics=30,max_topics=30)
    #print_topics(30, 20, '')

    #print_dictionary(prefix='')

