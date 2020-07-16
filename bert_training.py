import data.preprocessor as preprocessor
import data.phrase_docs as phraser
import model.topic_model as topic_model
import ansaro_utils.features.topic_model_load as model_loader
from pathlib import Path
import os
import numpy as np
import datetime
import pandas as pd

import torch
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings, BertEmbeddings
from flair.data import Sentence
from sklearn.decomposition import PCA,IncrementalPCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix

from tqdm import tqdm


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


def training_pipeline_bert(filepath=None, num_words_to_print=10,prefix=None,min_topics=19,max_topics=19,step=2):

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

  #logging.info(f'preprocessor.process_data_save: {filepath}')
  #preprocessor.process_data_save(filepath=filepath, as_text=as_text, as_pickle=as_pickle, verbose=verbose)
  #logging.info(f'phraser.raw_to_phrased_data_pipeline...')
  #phraser.raw_to_phrased_data_pipeline(to_load='text', verbose=True, overwrite_interim=True, prefix=None)
  col=cols[0]
  df = phraser.load_phrased_data_pipeline(to_load='text', verbose=True, overwrite_interim=True, prefix=None, training=True,col='resp_whytfa')

  #if prefix is None:
  #      prefix = ''
    # for topic modeling
  #trigram_docs_filepath = data_dir_processed / f'{prefix}{col}_transformed_docs_all.txt'
  #trigram_docs_filepath = f'/home/watsonrtdev/topic_modeling/input_data/topic_modeling/training/processed/{prefix}{col}_transformed_docs_all.txt'
  #trigram_docs_filepath = f'/home/watsonrtdev/topic_modeling/input_data/topic_modeling/training/processed/processed_dataframe.csv'

  #print(f'Loading input file {trigram_docs_filepath}')
  # turn to posix filepaths until gensim supports this
  #trigram_docs_filepath =  trigram_docs_filepath.as_posix()

  #trigram_docs = LineSentence(trigram_docs_filepath)
  #df = pd.read_csv(trigram_docs_filepath)
  #print(df.columns)

  #default it to min/max topics
  num_topics_range = range(min_topics, max_topics + 1, step)
  #if num_topics is not None:
  #    num_topics_range = range(num_topics, num_topics + 1, step)
  print('Num_topics_range={}'.format(num_topics_range))

  #Contextual string embeddings are powerful embeddings that capture latent syntactic-semantic information that goes beyond standard word embeddings. Key differences are: (1) they are trained without any explicit notion of words and thus fundamentally model words as sequences of characters. And (2) they are contextualized by their surrounding text, meaning that the same word will have different embeddings depending on its contextual use.
  # initialise embedding classes
  flair_embedding_forward = FlairEmbeddings('news-forward')
  flair_embedding_backward = FlairEmbeddings('news-backward')

  bert_embedding = BertEmbeddings('bert-base-uncased')

  # combine word embedding models
  document_embeddings = DocumentPoolEmbeddings([bert_embedding, flair_embedding_backward, flair_embedding_forward])

  # set up empty tensor
  X = torch.empty(size=(len(df.index), 7168)) #.cuda()
  # fill tensor with embeddings

  #  for text in tqdm(df['resp_whytfa']):    #df['text_cl']):
  #from tqdm import tqdm - show smart progress meter
  i=0
  for text in df['resp_whytfa']:  
    sentence = Sentence(text)
    document_embeddings.embed(sentence)
    embedding = sentence.get_embedding()
    X[i] = embedding
    i += 1

    if(i>100): 
        break

  print("before the PCA") 

  #detach the tensor from the GPU and convert it to a NumPy array
  Y = X.cpu().detach().numpy()
  #del(X)
  #torch.cuda.empty_cache()

  #We want to cluster these vectors into topics, and we’ll invoke Agglomerative Clustering with Ward affinity from scikit-learn to do so. 
  #Bottom-up hierarchical clustering algorithms have a memory complexity of O(n²), so we’ll use Principal Component Analysis to speed up this process.
  #As a side note, I did test a number of clustering algorithms (K-means, BIRCH, DBSCAN, Agglomerative with complete/average affinity), but Ward seems to perform the best in most cases

  #reduce the dimensionality of our vectors to length 768  
  pca = IncrementalPCA(copy=False,n_components=768,batch_size=1000)
  #pca = PCA(n_components=768)
  X_red = pca.fit_transform(X)
  
  del(X)
  print("After the fit_transform")

  N_CLUSTERS = 5
  # WARD CLUSTER
  ward = AgglomerativeClustering(n_clusters=N_CLUSTERS,
                               affinity='euclidean',
                               linkage='ward')
  pred_ward = ward.fit_predict(X_red)
  print("After fit_predict")

  df['topic'] = pred_ward
  df.to_csv('bert_withtopic.csv')
  print("Write bert_withtopic.csv")

  #get topic composition
  topic_docs = []
  # group text into topic-documents
  for topic in range(N_CLUSTERS):
    topic_docs.append(' '.join(df[df['cluster']==topic]['text_cl'].values))
  # apply function
  df_tfidf = get_top_words(topic_docs, 10)
  print(f"Top words: df_tfidf")


  #How good are our topics?
  #We find the centroids of the vectors by averaging them across each topic:
  topic_centroids = []
  for topic in tqdm(range(N_CLUSTERS)):
    X_topic = X_red[df.index[df['cluster']==topic]]
    X_mean = np.mean(X_topic, axis=0)
    topic_centroids.append(X_mean)

  #calculate the euclidean distance of each Tweet vector to their respective topic centroid:
  topic_distances = []
  for row in tqdm(df.index):
    topic_centroid = topic_centroids[df.iloc[row]['cluster']]
    X_row = X_red[row]
    topic_distance = euclidean(topic_centroid, X_row)
    topic_distances.append(topic_distance)
    
  df['topic_distance'] = topic_distances
  #visualise the distribution of distances to the topic centroid
  #The closer the distribution to the left of the graph, the more compact the topic is
  df.to_csv('bert_withtopic_distance.csv')
  print('Write bert_withtopic_distance.csv')

  #topic similarity - how similar the topics are to each other
  #We will construct a euclidean distance matrix between the 10 topic centroids to find the distance between the topic averages
  df_dist_matrix = pd.DataFrame(distance_matrix(topic_centroids,
                                              topic_centroids),
                              index=range(N_CLUSTERS),
                              columns=range(N_CLUSTERS))

  print(f"df_dist_matrix={df_dist_matrix}")
  with open('df_dist_matrix', 'w') as fout:
    fout.write(u'#'+'\t'.join(str(e) for e in df_dist_matrix.shape)+'\n')
    df_dist_matrix.tofile(fout)



def get_top_words(documents, top_n):
  '''
  function to get top tf-idf words and phrases
  '''
  vectoriser = TfidfVectorizer(ngram_range=(1, 2),
                               max_df=0.5)
  tfidf_matrix = vectoriser.fit_transform(documents)
  feature_names = vectoriser.get_feature_names()
  df_tfidf = pd.DataFrame()
  for doc in range(len(documents)):
    words = []
    scores = []
    feature_index = tfidf_matrix[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc, x] for x in feature_index])
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
      words.append(w)
      scores.append(s)
    df_temp = pd.DataFrame(data={'word':words, 'score':scores})
    df_temp = df_temp.sort_values('score',ascending=False).head(top_n)
    df_temp['topic'] = doc
    df_tfidf = df_tfidf.append(df_temp)
  return df_tfidf



def training_pipeline_lda(filepath=None, num_words_to_print=10,prefix=None,min_topics=19,max_topics=19,step=2):

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

    training_pipeline_bert(filepath="2017_2018_whytfa_with_header.csv",num_words_to_print=100,prefix='',min_topics=5,max_topics=5)
    #update_model(filepath="2019_whytfa.csv",num_topics=20)

    #training_pipeline(filepath="whytfa_14_15_16_17_18_19.csv",num_words_to_print=20,prefix='',min_topics=20,max_topics=20)
    #print_topics(30, 20, '')

    #inference_pipeline('2017_2018_whytfa_with_header.csv', 5)
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

